"""xbdoc 存储层：持久化、文档 CRUD、切片缓存、会话绑定解析、配置读取。

被主插件多继承（Mixin），不含任何 @filter 装饰方法。
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
    _HAS_DATA_PATH = True
except Exception:
    _HAS_DATA_PATH = False

try:
    from .xbdoc_retrieval import (
        ALLOWED_SUFFIXES,
        chunk_text,
        extract_text_from_bytes,
        tokenize,
    )
except ImportError:
    from xbdoc_retrieval import (
        ALLOWED_SUFFIXES,
        chunk_text,
        extract_text_from_bytes,
        tokenize,
    )

PLUGIN_NAME = "astrbot_plugin_xbdoc"

# 配置唯一来源：与 _conf_schema.json 默认值保持一致，__init__ 不做快照
CONFIG_DEFAULTS: Dict[str, Any] = {
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "top_k": 3,
    "max_inject_chars": 6000,
    "auto_inject": True,
    "allow_private_bind": True,
    "perf_log": False,
}

# 模式强度（隔离级别）：脏 key 合并时高强度胜出，保证结果与遍历顺序无关
_MODE_PRIORITY = {"reference": 0, "system": 1, "workspace": 2}


def _locked(method):
    """串行化写方法：与 _save_lock（RLock）配合，可嵌套，不死锁。"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._save_lock:
            return method(self, *args, **kwargs)
    return wrapper


# ======================================================================
# 存储 Mixin
# ======================================================================

class XbdocStoreMixin:

    def _init_store(self) -> None:
        """初始化持久化存储：路径、缓存、锁、数据加载与标准化（变化才回写）。"""
        # 锁与缓存必须先就绪：后续 _save_json/_load_chunks 依赖它们
        # RLock：读改写关键区（bind/unbind/入库）可与内部的 _save_json 嵌套加锁，不死锁
        self._save_lock = threading.RLock()  # 落盘+绑定锁：防 WebUI 与聊天指令并发写撕裂
        self._chunk_cache: Dict[str, List[str]] = {}  # 内存缓存：doc_id -> chunks
        self._chunk_tokens_cache: Dict[str, List[Counter]] = {}  # 性能优化：doc_id -> 每切片词频
        self._seen_save_ts = 0  # 群记录节流时间戳（仅内存，不落盘）
        self._fulltext_cache: Dict[str, str] = {}  # doc_id -> 全文（system/workspace 免每消息重拼）
        self._bm25_cache: Dict[str, Any] = {}  # BM25 全局量缓存（key 命中即复用）
        # 持久化存储路径
        self._latest_bot = None
        self.data_dir = self._resolve_data_dir()
        self.docs_dir = self.data_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.json"
        self.bindings_path = self.data_dir / "bindings.json"
        self.seen_path = self.data_dir / "seen_groups.json"
        # 崩溃残留的 tmp 先清，避免越积越多（正常 save 结束无残留）
        try:
            for _tmp in self.data_dir.glob("*.tmp"):
                try:
                    _tmp.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        # 加载并自动标准化数据（有变才回写，避免每次启动空转 I/O）
        self._index: Dict[str, Dict[str, Any]] = self._load_json(self.index_path, {})
        self._seen_groups: Dict[str, Dict[str, Any]] = self._load_json(self.seen_path, {})
        _raw_bindings = self._load_json(self.bindings_path, {})
        self._bindings: Dict[str, Dict[str, Any]] = self._normalize_bindings(_raw_bindings)
        if self._bindings != _raw_bindings and self.bindings_path.exists():
            self._save_json(self.bindings_path, self._bindings)


    def save_all(self) -> None:
        """全量落盘（terminate 唯一调用）。"""
        self._save_json(self.index_path, self._index)
        self._save_json(self.bindings_path, self._bindings)
        self._save_seen()


    # ---------- 路径与持久化 ----------
    def _resolve_data_dir(self) -> Path:
        if _HAS_DATA_PATH:
            try:
                return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
            except Exception:
                pass
        _plug_root = Path(__file__).resolve().parent
        for cand in [
            _plug_root.parent.parent.parent / "data" / "plugin_data" / PLUGIN_NAME,
            Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME,
            _plug_root / "data_store",
        ]:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                return cand.resolve()
            except Exception:
                continue
        # 终极兜底：系统临时目录（可写、独立、不污染工程；重启不丢由 AstrBot 数据目录保证）
        fallback = Path(tempfile.gettempdir()) / PLUGIN_NAME
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback.resolve()
        except Exception:
            return fallback


    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 读取 {path.name} 失败: {e}")
            # 损坏文件先备份再丢弃，避免下次保存直接覆盖丢失现场
            try:
                if path.exists() and path.stat().st_size > 0:
                    bak = path.with_name(f"{path.stem}.corrupt-{int(time.time())}.bak")
                    bak.write_bytes(path.read_bytes())
                    logger.warning(f"[{PLUGIN_NAME}] 已备份损坏文件: {bak.name}")
            except Exception:
                pass
        return default


    def _save_json(self, path: Path, data: Any) -> None:
        # 原子写：先落 tmp 再 os.replace，同文件系统下读方永不见半截文件
        try:
            with self._save_lock:
                tmp = path.with_name(f"{path.name}.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp, path)
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 保存 {path.name} 失败: {e}")


    def _save_bytes_atomic(self, path: Path, data: bytes) -> None:
        """二进制原子写（切片缓存/入库原文件），与 _save_json 同策略。"""
        with self._save_lock:
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)


    def _save_seen(self) -> None:
        if len(self._seen_groups) > 500:
            items = sorted(
                self._seen_groups.items(),
                key=lambda kv: kv[1].get("last_seen", 0),
                reverse=True,
            )[:500]
            self._seen_groups = dict(items)
        self._save_json(self.seen_path, self._seen_groups)


    @staticmethod
    def _platform_of(event: Any) -> str:
        """提取事件所属平台（onebot/telegram/...），取不到返回空串，不抛异常。"""
        try:
            platform = str(getattr(event, "platform_id", "") or getattr(event, "platform", "") or "")
            if not platform:
                umo = getattr(event, "unified_msg_origin", "") or ""
                platform = umo.split(":", 1)[0] if ":" in umo else ""
            return platform.strip()
        except Exception:
            return ""

    def _record_seen_group(self, event: AstrMessageEvent) -> None:
        """记录会话基础信息，供 WebUI 模糊搜索/绑定选用（群聊与私聊通用）。"""
        try:
            platform = self._platform_of(event)

            gid = str(event.get_group_id() or "").strip()
            if not gid:
                self._record_seen_private(event, platform)
                return

            group_name = ""
            grp = getattr(event.message_obj, "group", None)
            if grp is not None:
                group_name = str(getattr(grp, "group_name", "") or "").strip()

            now = int(time.time())
            # seen 键同样平台限定（group:plat:gid），跨平台同号不再互相覆盖
            seen_key = f"group:{platform}:{gid}" if platform else gid
            ent = self._seen_groups.setdefault(seen_key, {
                "gid": gid, "group_name": group_name, "platform": platform,
                "kind": "group",
                "first_seen": now, "last_seen": now, "msg_count": 0,
            })
            ent["last_seen"] = now
            ent["msg_count"] = int(ent.get("msg_count", 0)) + 1
            if group_name and group_name != ent.get("group_name"):
                ent["group_name"] = group_name
            if platform and not ent.get("platform"):
                ent["platform"] = platform

            if ent["msg_count"] % 25 == 0 or (now - self._seen_save_ts) > 45:
                self._seen_save_ts = now
                self._save_seen()
        except Exception:
            pass

    def _record_seen_private(self, event: AstrMessageEvent, platform: str) -> None:
        """记录私聊会话（sender 昵称复用 group_name 字段展示，kind 标记区分）。"""
        try:
            uid = ""
            nickname = ""
            try:
                msg_obj = getattr(event, "message_obj", None)
                sender = getattr(msg_obj, "sender", None) if msg_obj is not None else None
                if sender is not None:
                    for a in ("user_id", "id", "qq", "uid"):
                        uid = str(getattr(sender, a, "") or "").strip()
                        if uid:
                            break
                    for a in ("nickname", "remark", "card", "name"):
                        nickname = str(getattr(sender, a, "") or "").strip()
                        if nickname:
                            break
            except Exception:
                pass
            if not uid:
                ck = self._canonical_key(event)
                if ck.startswith("private:"):
                    uid = ck.split(":", 1)[1]
            if not uid:
                return
            uid = re.sub(r"\D", "", uid) or uid
            key = f"private:{platform}:{uid}" if platform else f"private:{uid}"

            now = int(time.time())
            ent = self._seen_groups.setdefault(key, {
                "gid": uid, "group_name": nickname, "platform": platform,
                "kind": "private",
                "first_seen": now, "last_seen": now, "msg_count": 0,
            })
            ent["kind"] = "private"
            ent["last_seen"] = now
            ent["msg_count"] = int(ent.get("msg_count", 0)) + 1
            if nickname and nickname != ent.get("group_name"):
                ent["group_name"] = nickname
            if platform and not ent.get("platform"):
                ent["platform"] = platform

            if ent["msg_count"] % 25 == 0 or (now - self._seen_save_ts) > 45:
                self._seen_save_ts = now
                self._save_seen()
        except Exception:
            pass


    # ---------- 文档管理 ----------
    @staticmethod
    def _safe_filename(name: str) -> str:
        name = (name or "unnamed").strip().replace("\\", "_").replace("/", "_")
        name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name) or "unnamed"
        # ext4 等按字节限长（255B）：中文占 3 字节，超限会写盘崩溃
        return name.encode("utf-8")[:200].decode("utf-8", errors="ignore") or "unnamed"


    @_locked
    def add_document(self, filename: str, data: bytes) -> Dict[str, Any]:
        filename = self._safe_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise RuntimeError(f"不支持的类型 {suffix or '(无后缀)'}，支持格式: {sorted(ALLOWED_SUFFIXES)}")
        if not data:
            raise RuntimeError("空文件，无法入库")
        if len(data) > 50 * 1024 * 1024:
            raise RuntimeError("文件超出 50MB 上限，请拆分后上传")

        text = extract_text_from_bytes(suffix, data).strip()

        if len(text) < 2:
            raise RuntimeError("提取纯文本内容过少，拒绝入库")

        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        doc_id = hashlib.md5(f"{filename}:{len(data)}:{text_hash}".encode("utf-8")).hexdigest()[:10]
        stored_name = f"{doc_id}_{filename}"
        # 同 ID 旧文件残留清理（如改名重传导致文件名变化）
        old_meta = self._index.get(doc_id)
        if old_meta:
            old_stored = str(old_meta.get("stored_name", ""))
            if old_stored and old_stored != stored_name:
                try:
                    old_p = self.docs_dir / old_stored
                    if old_p.exists():
                        old_p.unlink()
                except Exception:
                    pass
        self._save_bytes_atomic(self.docs_dir / stored_name, data)

        chunks = chunk_text(text, self._cfg_int("chunk_size"), self._cfg_int("chunk_overlap"))
        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "stored_name": stored_name,
            "suffix": suffix,
            "size": len(data),
            "text_len": len(text),
            "chunks": len(chunks),
            "updated_at": int(time.time()),
        }
        self._index[doc_id] = meta
        self._save_bytes_atomic(
            self.data_dir / f"chunks_{doc_id}.json",
            json.dumps(chunks, ensure_ascii=False).encode("utf-8"),
        )
        self._chunk_cache[doc_id] = chunks  # 更新内存缓存
        self._chunk_tokens_cache.pop(doc_id, None)  # 词频缓存失效，下次检索重建
        self._fulltext_cache.pop(doc_id, None)  # 全文缓存失效
        # BM25 全局量与绑定集合相关：任何文档变更都可能改变 idf，直接整桶清空最稳
        try:
            _bm25 = getattr(self, "_bm25_cache", None)
            if isinstance(_bm25, dict):
                _bm25.clear()
        except Exception:
            pass
        self._save_json(self.index_path, self._index)
        logger.info(f"[{PLUGIN_NAME}] 入库文档 {filename} id={doc_id} chunks={len(chunks)}")
        return meta


    @_locked
    def delete_document(self, doc_id: str) -> bool:
        meta = self._index.pop(doc_id, None)
        if not meta:
            return False
        stored = str(meta.get("stored_name", ""))
        targets = [self.data_dir / f"chunks_{doc_id}.json"]
        if stored:
            targets.insert(0, self.docs_dir / stored)
        for p in targets:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self._chunk_cache.pop(doc_id, None)  # 清除内存缓存
        self._chunk_tokens_cache.pop(doc_id, None)
        self._fulltext_cache.pop(doc_id, None)
        try:
            _bm25 = getattr(self, "_bm25_cache", None)
            if isinstance(_bm25, dict):
                _bm25.clear()
        except Exception:
            pass

        # 同步清理所有绑定引用；被清空的会话回落模式并尝试删除空条目
        changed = False
        for key, ent in list(self._bindings.items()):
            if doc_id in ent.get("doc_ids", []):
                ent["doc_ids"] = [i for i in ent["doc_ids"] if i != doc_id]
                if not ent["doc_ids"] and str(ent.get("mode") or "") in ("system", "workspace"):
                    ent["mode"] = "reference"
                self._prune_empty_entry(key)
                changed = True
        self._save_json(self.index_path, self._index)
        if changed:
            self._save_json(self.bindings_path, self._bindings)
        return True


    def list_documents(self) -> List[Dict[str, Any]]:
        return sorted(self._index.values(), key=lambda m: m.get("updated_at", 0), reverse=True)


    def _load_chunks(self, doc_id: str) -> List[str]:
        # 内存缓存命中（命中即移到队尾，变 FIFO 为 LRU，热点文档不被挤掉）
        if doc_id in self._chunk_cache:
            try:
                self._chunk_cache[doc_id] = self._chunk_cache.pop(doc_id)
            except Exception:
                pass
            return self._chunk_cache[doc_id]

        cache = self.data_dir / f"chunks_{doc_id}.json"
        try:
            if cache.exists():
                data = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    result = [str(x) for x in data]
                    self._remember_chunks(doc_id, result)
                    return result
        except Exception:
            pass

        meta = self._index.get(doc_id)
        if not meta:
            return []
        try:
            raw = (self.docs_dir / str(meta["stored_name"])).read_bytes()
            text = extract_text_from_bytes(str(meta.get("suffix", "")), raw)
            chunks = chunk_text(text, self._cfg_int("chunk_size"), self._cfg_int("chunk_overlap"))
            self._save_bytes_atomic(cache, json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
            self._remember_chunks(doc_id, chunks)
            return chunks
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 重建切片失败 {doc_id}: {e}")
            return []


    def _remember_chunks(self, doc_id: str, chunks: List[str]) -> None:
        """缓存切片并做简单上限保护，防止文档过多时内存无限增长。"""
        if len(self._chunk_cache) > 200:
            try:
                oldest = next(iter(self._chunk_cache))
                self._chunk_cache.pop(oldest, None)
                self._chunk_tokens_cache.pop(oldest, None)
            except Exception:
                pass
        self._chunk_cache[doc_id] = chunks


    def _get_chunk_counters(self, doc_id: str) -> List[Counter]:
        """获取每切片词频（缓存），避免每次提问重复分词全量切片。"""
        cached = self._chunk_tokens_cache.get(doc_id)
        if cached is not None:
            try:
                self._chunk_tokens_cache[doc_id] = self._chunk_tokens_cache.pop(doc_id)
            except Exception:
                pass
            return self._chunk_tokens_cache[doc_id]
        chunks = self._load_chunks(doc_id)
        counters = [Counter(tokenize(ch)) for ch in chunks]
        self._chunk_tokens_cache[doc_id] = counters
        return counters


    def _get_full_text(self, doc_id: str) -> str:
        """获取文档全文（缓存）：system/workspace 模式每消息复用，入库/删除时失效。"""
        cached = self._fulltext_cache.get(doc_id)
        if cached is not None:
            try:
                self._fulltext_cache[doc_id] = self._fulltext_cache.pop(doc_id)
            except Exception:
                pass
            return self._fulltext_cache[doc_id]
        text = "\n".join(self._load_chunks(doc_id))
        if len(self._fulltext_cache) > 50:
            try:
                self._fulltext_cache.pop(next(iter(self._fulltext_cache)))
            except Exception:
                pass
        self._fulltext_cache[doc_id] = text
        return text


    # ---------- 会话与绑定管理 ----------
    @staticmethod
    def _split_session_key(cks: str):
        """拆会话 key -> (kind, platform|None, ident)。

        新格式 kind:platform:ident 原样拆；老格式 kind:ident 平台为 None；脏串 kind 为空。
        """
        parts = str(cks or "").strip().split(":")
        if len(parts) >= 3 and parts[0].lower() in ("group", "private"):
            return parts[0].lower(), parts[1].strip() or None, ":".join(parts[2:]).strip()
        if len(parts) == 2 and parts[0].lower() in ("group", "private"):
            return parts[0].lower(), None, parts[1].strip()
        return "", None, str(cks or "").strip()

    @staticmethod
    def _canonical_key_str(k: str) -> str:
        s = str(k or "").strip()
        if not s:
            return ""
        # 新规范 kind:platform:ident 直接透传
        kind, plat, ident = XbdocStoreMixin._split_session_key(s)
        if kind and plat and ident:
            return f"{kind}:{plat}:{ident}"
        if s.lower().startswith("group:"):
            tail = s.split(":", 1)[1].strip()
            return f"group:{tail}" if tail else ""
        if s.isdigit():
            return f"group:{s}"
        m = re.search(r"(?:GroupMessage|group)\s*:\s*(\d+)", s, re.IGNORECASE)
        if m:
            return f"group:{m.group(1)}"
        # 私聊不再冒充 group，避免私聊号与群号碰撞；统一归一为 private:xxx
        mp = re.search(r"(?:FriendMessage|PrivateMessage|Private|Friend|User)\s*:\s*(\S+)", s, re.IGNORECASE)
        if mp:
            uid = re.sub(r"\D", "", mp.group(1)) or mp.group(1).strip()
            return f"private:{uid}" if uid else s
        parts = s.split(":")
        if parts and parts[-1].isdigit():
            # 无法判断群/私时保守返回原串，由调用方按群优先处理
            return s
        return s


    def _canonical_key(self, event_or_str: Any) -> str:
        """事件 -> 规范会话 key。新格式 kind:platform:ident（平台隔离，防跨平台同号串台）。

        gid/sender 属权威 id，有平台就限定；umo 纯启发式，原样归一不二次限定；
        平台取不到时回落老格式，保证不断连。
        """
        if isinstance(event_or_str, str):
            return self._canonical_key_str(event_or_str)
        plat = self._platform_of(event_or_str)
        try:
            gid = str(event_or_str.get_group_id() or "").strip()
            if gid:
                # 适配器已返回限定/复合格式（如 group:xxx / group:plat:gid）时直接规范化
                if ":" in gid:
                    return self._canonical_key_str(gid)
                return f"group:{plat}:{gid}" if plat else f"group:{gid}"
        except Exception:
            pass
        umo = str(getattr(event_or_str, "unified_msg_origin", "") or "").strip()
        ck = self._canonical_key_str(umo)
        if ck.startswith("private:") or ck.startswith("group:"):
            return ck
        # 私聊兜底：尝试取 sender id（权威 id，有平台就限定）
        try:
            for attr in ("sender_id", "user_id", "qq", "uid"):
                uid = str(getattr(event_or_str, attr, "") or "").strip()
                if uid:
                    uid = re.sub(r"\D", "", uid) or uid
                    return f"private:{plat}:{uid}" if plat else f"private:{uid}"
            msg_obj = getattr(event_or_str, "message_obj", None)
            sender = getattr(msg_obj, "sender", None) if msg_obj is not None else None
            if sender is not None:
                uid = str(getattr(sender, "user_id", "") or getattr(sender, "id", "") or "").strip()
                if uid:
                    uid = re.sub(r"\D", "", uid) or uid
                    return f"private:{plat}:{uid}" if plat else f"private:{uid}"
        except Exception:
            pass
        return ck or "default"


    def _normalize_bindings(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """标准化绑定：key 归一、模式归一，同 key 条目确定性合并。"""
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            ck = self._canonical_key_str(str(k))
            if not ck or not isinstance(v, dict):
                continue
            new_ent = {
                "doc_ids": [str(i) for i in (v.get("doc_ids") or [])],
                "prompt": str(v.get("prompt") or "").strip(),
                "shield": bool(v.get("shield", False)),
                "mode": self._normalize_mode(v.get("mode")) or "reference",
                "force_system_prompt": bool(v.get("force_system_prompt", False)),
                # 保留未知字段（如 ignore_history），避免打开WebUI就丢配置
                **{kk: vv for kk, vv in v.items() if kk not in (
                    "doc_ids", "prompt", "shield", "mode", "force_system_prompt")},
            }
            if ck not in out:
                out[ck] = new_ent
            else:
                self._merge_entries(out[ck], new_ent)
        return out


    @staticmethod
    def _seen_platforms(kind: str, ident: str, seen: Dict[str, Any]) -> set:
        """seen 中认领该 (kind, id) 的平台集合（恰为 1 个时才可安全限定）"""
        plats = set()
        for meta in (seen or {}).values():
            if not isinstance(meta, dict):
                continue
            is_priv = str(meta.get("kind") or "") == "private"
            if is_priv != (kind == "private"):
                continue
            if str(meta.get("gid") or "").strip() != ident:
                continue
            p = str(meta.get("platform") or "").strip()
            if p:
                plats.add(p)
        return plats


    def _session_keys(self, event: AstrMessageEvent) -> List[str]:
        ck = self._canonical_key(event)
        keys = [ck]
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        for cand in (umo, self._canonical_key_str(umo)):
            if cand and cand not in keys:
                keys.append(cand)
        return keys


    @staticmethod
    def _merge_entries(cur: Dict[str, Any], old: Dict[str, Any]) -> Dict[str, Any]:
        """同会话两条目合并（加载时去重用）：doc 并集保序，开关取或，模式按强度。"""
        merged_ids = list(cur.get("doc_ids", []))
        for d in old.get("doc_ids", []):
            if d not in merged_ids:
                merged_ids.append(d)
        cur["doc_ids"] = merged_ids
        if not str(cur.get("prompt") or "").strip():
            cur["prompt"] = str(old.get("prompt") or "")
        for flag in ("shield", "force_system_prompt", "ignore_history"):
            cur[flag] = bool(cur.get(flag)) or bool(old.get(flag))
        if _MODE_PRIORITY.get(str(old.get("mode") or "reference"), 0) > \
                _MODE_PRIORITY.get(str(cur.get("mode") or "reference"), 0):
            cur["mode"] = old.get("mode")
        if not cur.get("platform") and old.get("platform"):
            cur["platform"] = old.get("platform")
        # 其余未知字段只补不盖（创建时缺省已齐，此处多为 ignore_history 等开关的或合并）
        for kk, vv in old.items():
            if kk not in ("doc_ids", "prompt", "shield", "force_system_prompt",
                          "ignore_history", "mode", "platform"):
                cur.setdefault(kk, vv)
        return cur

    def _qualify_session_key(self, raw_key: str) -> str:
        """WebUI 手填 key 补平台限定：已限定原样；老格式按 seen 单一认领补足，多认领/无认领原样返回。"""
        ck = self._canonical_key_str(raw_key)
        kind, plat, ident = self._split_session_key(ck)
        if not kind or plat or not ident:
            return ck
        plats = self._seen_platforms(kind, ident, self._seen_groups)
        if len(plats) == 1:
            qualified = f"{kind}:{plats.pop()}:{ident}"
            logger.info(f"[{PLUGIN_NAME}] 会话 key 已补平台限定: {ck} -> {qualified}")
            return qualified
        if len(plats) > 1:
            logger.warning(
                f"[{PLUGIN_NAME}] 会话 {ck} 被多平台认领 {sorted(plats)}，"
                f"保持原样，请在 WebUI 用完整 key（kind:platform:id）指定。"
            )
        return ck

    def _get_entry(self, session_key: str) -> Dict[str, Any]:
        ck = self._canonical_key_str(session_key)
        return self._bindings.setdefault(ck, {
            "doc_ids": [], "prompt": "", "shield": False, "mode": "reference", "force_system_prompt": False,
        })


    def _resolve_session(self, event_or_key: Any, create: bool = False) -> Tuple[str, Dict[str, Any]]:
        """全插件统一会话解析：canonical 优先命中，其次兼容历史脏 key。

        返回 (matched_key, entry)。create=False 时未命中返回 {} 且绝不写 bindings，
        调用方拿到的 key 即真正生效的 key，不再各算一遍。
        """
        if isinstance(event_or_key, str):
            cands = [event_or_key]
        else:
            try:
                primary = self._canonical_key(event_or_key)
            except Exception:
                primary = "default"
            cands = [primary]
            try:
                for k in self._session_keys(event_or_key):
                    if k not in cands:
                        cands.append(k)
            except Exception:
                pass
        for k in cands:
            ck = self._canonical_key_str(k)
            if ck and ck in self._bindings:
                return ck, self._bindings[ck]
        primary_ck = self._canonical_key_str(cands[0]) if cands else "default"
        if not primary_ck:
            primary_ck = "default"
        if create:
            return primary_ck, self._get_entry(primary_ck)
        return primary_ck, {}


    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """归一化模式名；未知输入返回空串（调用方自行报错/回落，不静默串改）。"""
        m = str(mode or "").lower().strip()
        if m in ("system", "sys", "s", "强制", "提示词", "1"):
            return "system"
        if m in ("workspace", "ws", "w", "工作区", "沙箱", "3"):
            return "workspace"
        if m in ("reference", "ref", "r", "参考", "仅参考", "资料", "0"):
            return "reference"
        return ""


    @staticmethod
    def _parse_doc_ids(*parts: str) -> List[str]:
        """解析文档 ID：兼容空格/逗号/分号分隔，自动去重保序。"""
        ids: List[str] = []
        for p in parts:
            if not p:
                continue
            for tok in re.split(r"[\s,;，；]+", str(p)):
                t = tok.strip().strip(",;")
                if t and t not in ids:
                    ids.append(t)
        return ids


    def _find_matching_keys(self, event: AstrMessageEvent) -> List[str]:
        """找到本会话所有命中的绑定 Key（解决新旧 Key 并存导致解绑遗漏）。"""
        keys = []
        for k in self._session_keys(event):
            ck = self._canonical_key_str(k)
            if ck in self._bindings and ck not in keys:
                keys.append(ck)
        ck = self._canonical_key(event)
        if ck not in keys:
            keys.append(ck)
        return keys


    def get_bound_doc_ids(self, event: AstrMessageEvent) -> List[str]:
        # 与注入链路共用同一会话解析，避免状态显示与实际生效不一致
        return list(self._effective_session(event).get("doc_ids", []))


    def _effective_session(self, event: AstrMessageEvent) -> Dict[str, Any]:
        """获取本会话综合生效配置（canonical 优先、脏 key 兼容，matched_key 即真实命中）。"""
        key, ent = self._resolve_session(event, create=False)
        doc_ids = [d for d in ent.get("doc_ids", []) if d in self._index]
        return {
            "doc_ids": doc_ids,
            "prompt": str(ent.get("prompt") or "").strip(),
            "shield": bool(ent.get("shield", False)),
            "mode": str(ent.get("mode") or "reference"),
            "force_system_prompt": bool(ent.get("force_system_prompt", False)),
            "ignore_history": bool(ent.get("ignore_history", False)),
            "matched_key": key,
            "has_entry": bool(ent),
        }


    @_locked
    def bind_docs(self, session_key: str, doc_ids: List[str]) -> List[str]:
        """只改内存不落盘，调用方统一 save（避免一次绑定写两次文件）。"""
        valid = [d for d in doc_ids if d in self._index]
        self._get_entry(session_key)["doc_ids"] = valid
        return valid


    def _prune_empty_entry(self, session_key: str) -> bool:
        """解绑后若该会话无文档、无提示词、无屏蔽/强制/断史且为默认模式，则彻底删除条目。

        避免 bindings.json 堆积空壳，会话列表看着像“没解掉”。
        """
        ck = self._canonical_key_str(session_key)
        ent = self._bindings.get(ck)
        if not ent:
            return False
        if (
            not ent.get("doc_ids")
            and not str(ent.get("prompt") or "").strip()
            and not ent.get("shield", False)
            and not ent.get("force_system_prompt", False)
            and not ent.get("ignore_history", False)
            and str(ent.get("mode") or "reference") == "reference"
        ):
            self._bindings.pop(ck, None)
            return True
        return False


    @_locked
    def set_session_prompt(self, session_key: str, prompt: str) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        ent["prompt"] = (prompt or "").strip()
        self._save_json(self.bindings_path, self._bindings)
        return ent


    @_locked
    def set_session_mode(self, session_key: str, mode: str) -> Dict[str, Any]:
        # 文档生效模式必须有绑定文档才能切换，无文档时强制回落 reference
        ent = self._get_entry(session_key)
        if not [d for d in ent.get("doc_ids", []) if d in self._index]:
            ent["mode"] = "reference"
            self._save_json(self.bindings_path, self._bindings)
            return ent
        ent["mode"] = self._normalize_mode(mode) or "reference"
        self._save_json(self.bindings_path, self._bindings)
        return ent


    # ---------- 动态配置读取（唯一来源：CONFIG_DEFAULTS + 实时 config） ----------
    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            if default is None:
                default = CONFIG_DEFAULTS.get(key)
            cfg = getattr(self, "config", None) or {}
            val = cfg.get(key, default) if hasattr(cfg, "get") else default
            return default if val is None else val
        except Exception:
            return CONFIG_DEFAULTS.get(key, default)


    def _cfg_int(self, key: str, default: Optional[int] = None) -> int:
        """读正整数配置：非法值/<=0 时回落（显式 default 优先，其次 CONFIG_DEFAULTS）。"""
        try:
            fallback = int(CONFIG_DEFAULTS.get(key, 0))
        except Exception:
            fallback = 0
        try:
            d = fallback if default is None else int(default)
            v = int(self._cfg(key, d))
            if v > 0:
                return v
            return d if d > 0 else fallback
        except Exception:
            return fallback if fallback > 0 else 0


    def _cfg_no_limit(self, key: str) -> int:
        """读取字符上限：<=0 表示不限制、保证完整注入（与 _cfg_int 强制正数不同）。"""
        try:
            return int(self._cfg(key, CONFIG_DEFAULTS.get(key, 0)))
        except Exception:
            return int(CONFIG_DEFAULTS.get(key, 0))
