"""xbdoc 存储层：持久化、文档 CRUD、切片缓存、会话绑定解析、配置读取。

被主插件多继承（Mixin），不含任何 @filter 装饰方法。
"""

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        format_tavern_to_markdown,
        parse_tavern_card,
        tokenize,
    )
except ImportError:
    from xbdoc_retrieval import (
        ALLOWED_SUFFIXES,
        chunk_text,
        extract_text_from_bytes,
        format_tavern_to_markdown,
        parse_tavern_card,
        tokenize,
    )

PLUGIN_NAME = "astrbot_plugin_xbdoc"
OLD_PLUGIN_NAMES = ("xbdoc", "astrbot_plugin_doc_memory")

# 配置唯一来源：与 _conf_schema.json 默认值保持一致，__init__ 不做快照
CONFIG_DEFAULTS: Dict[str, Any] = {
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "top_k": 3,
    "max_inject_chars": 6000,
    "auto_inject": True,
    "allow_private_bind": True,
}


# ======================================================================
# 存储 Mixin
# ======================================================================

class XbdocStoreMixin:

    def _init_store(self) -> None:
        """初始化持久化存储：路径、缓存、锁、数据加载与标准化（变化才回写）。"""
        # 锁与缓存必须先就绪：后续 _save_json/_load_chunks 依赖它们
        self._save_lock = threading.Lock()  # 落盘锁：防 WebUI 与聊天指令并发写撕裂 tmp 文件
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

        # 加载并自动标准化数据（标准化改了内容才回写，避免每次启动空转 I/O）
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
        # 新插件名优先；若旧目录存在则自动迁移，保证改名不丢数据
        if _HAS_DATA_PATH:
            try:
                base = Path(get_astrbot_data_path()) / "plugin_data"
                new_p = base / PLUGIN_NAME
                if not new_p.exists():
                    # 多个旧目录并存时选最近使用过的迁移，其余仅告警不碰
                    legacy_dirs = [base / n for n in OLD_PLUGIN_NAMES if (base / n).exists()]

                    def _mtime(p: Path) -> float:
                        try:
                            b = p / "bindings.json"
                            return b.stat().st_mtime if b.exists() else p.stat().st_mtime
                        except Exception:
                            return 0.0

                    legacy_dirs.sort(key=_mtime, reverse=True)
                    for idx, old_p in enumerate(legacy_dirs):
                        if idx > 0:
                            logger.warning(f"[{PLUGIN_NAME}] 发现多余旧数据目录未迁移: {old_p}")
                            continue
                        try:
                            import shutil
                            try:
                                # 首选移动：不占双份空间，且删新目录后不会再被复活
                                shutil.move(str(old_p), str(new_p))
                            except Exception:
                                shutil.copytree(old_p, new_p)
                            logger.info(f"[{PLUGIN_NAME}] 已从旧数据目录迁移: {old_p} -> {new_p}")
                        except Exception as e:
                            logger.warning(f"[{PLUGIN_NAME}] 数据迁移失败，将直接使用旧目录: {e}")
                            return old_p
                        break
                return new_p
            except Exception:
                pass
        for cand in [
            Path(__file__).resolve().parent / ".." / ".." / ".." / "data" / "plugin_data" / PLUGIN_NAME,
            Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME,
            Path(__file__).resolve().parent / "data_store",
        ]:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                return cand.resolve()
            except Exception:
                continue
        return Path.cwd()


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


    def _record_seen_group(self, event: AstrMessageEvent) -> None:
        """记录会话基础信息，供 WebUI 模糊搜索/绑定选用（群聊与私聊通用）。"""
        try:
            platform = str(getattr(event, "platform_id", "") or getattr(event, "platform", "") or "")
            if not platform:
                umo = getattr(event, "unified_msg_origin", "") or ""
                platform = umo.split(":", 1)[0] if ":" in umo else ""

            gid = str(event.get_group_id() or "").strip()
            if not gid:
                self._record_seen_private(event, platform)
                return

            group_name = ""
            grp = getattr(event.message_obj, "group", None)
            if grp is not None:
                group_name = str(getattr(grp, "group_name", "") or "").strip()

            now = int(time.time())
            ent = self._seen_groups.setdefault(gid, {
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
            key = f"private:{uid}"

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
        return re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)[:120] or "unnamed"


    def add_document(self, filename: str, data: bytes) -> Dict[str, Any]:
        filename = self._safe_filename(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise RuntimeError(f"不支持的类型 {suffix or '(无后缀)'}，支持格式: {sorted(ALLOWED_SUFFIXES)}")
        if not data:
            raise RuntimeError("空文件，无法入库")
        if len(data) > 50 * 1024 * 1024:
            raise RuntimeError("文件超出 50MB 上限，请拆分后上传")

        # 智能检测酒馆角色卡 / 预设 (PNG / JSON)
        is_tavern = False
        chara_greeting = ""
        t_card = parse_tavern_card(data)
        if t_card:
            is_tavern = True
            d = t_card.get("data") if isinstance(t_card.get("data"), dict) else t_card
            chara_name = str(d.get("name") or t_card.get("name") or "").strip()
            chara_greeting = str(d.get("first_mes") or t_card.get("first_mes") or "").strip()
            if chara_name and chara_name not in ("未命名角色", "未命名酒馆角色"):
                filename = f"【酒馆】{self._safe_filename(chara_name)}.md"
            else:
                stem = Path(filename).stem
                filename = f"【酒馆】{self._safe_filename(stem)}.md"
            text = format_tavern_to_markdown(t_card).strip()
            suffix = ".md"
        else:
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
            "is_tavern": is_tavern,
            "greeting": chara_greeting,
        }
        self._index[doc_id] = meta
        self._save_bytes_atomic(
            self.data_dir / f"chunks_{doc_id}.json",
            json.dumps(chunks, ensure_ascii=False).encode("utf-8"),
        )
        self._chunk_cache[doc_id] = chunks  # 更新内存缓存
        self._chunk_tokens_cache.pop(doc_id, None)  # 词频缓存失效，下次检索重建
        self._fulltext_cache.pop(doc_id, None)  # 全文缓存失效
        self._save_json(self.index_path, self._index)
        logger.info(f"[{PLUGIN_NAME}] 入库文档 {filename} id={doc_id} chunks={len(chunks)} tavern={is_tavern}")
        return meta


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
        # 内存缓存命中
        if doc_id in self._chunk_cache:
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
            if meta.get("is_tavern"):
                card = parse_tavern_card(raw)
                text = format_tavern_to_markdown(card) if card else raw.decode("utf-8", errors="ignore")
            else:
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
            return cached
        chunks = self._load_chunks(doc_id)
        counters = [Counter(tokenize(ch)) for ch in chunks]
        self._chunk_tokens_cache[doc_id] = counters
        return counters


    def _get_full_text(self, doc_id: str) -> str:
        """获取文档全文（缓存）：system/workspace 模式每消息复用，入库/删除时失效。"""
        cached = self._fulltext_cache.get(doc_id)
        if cached is not None:
            return cached
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
    def _canonical_key_str(k: str) -> str:
        s = str(k or "").strip()
        if not s:
            return ""
        # 兼容 group:group:123 这类双重前缀脏数据
        while s.lower().startswith("group:group:"):
            s = s[6:]
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
        if isinstance(event_or_str, str):
            return self._canonical_key_str(event_or_str)
        try:
            gid = str(event_or_str.get_group_id() or "").strip()
            if gid:
                # 防止适配器已返回 group:123 形成双重前缀
                if gid.lower().startswith("group:"):
                    gid = gid.split(":", 1)[1].strip()
                if gid:
                    return f"group:{gid}"
        except Exception:
            pass
        umo = str(getattr(event_or_str, "unified_msg_origin", "") or "").strip()
        ck = self._canonical_key_str(umo)
        if ck.startswith("private:") or ck.startswith("group:"):
            return ck
        # 私聊兜底：尝试取 sender id
        try:
            for attr in ("sender_id", "user_id", "qq", "uid"):
                uid = str(getattr(event_or_str, attr, "") or "").strip()
                if uid:
                    return f"private:{re.sub(r'\\D', '', uid) or uid}"
            msg_obj = getattr(event_or_str, "message_obj", None)
            sender = getattr(msg_obj, "sender", None) if msg_obj is not None else None
            if sender is not None:
                uid = str(getattr(sender, "user_id", "") or getattr(sender, "id", "") or "").strip()
                if uid:
                    return f"private:{re.sub(r'\\D', '', uid) or uid}"
        except Exception:
            pass
        return ck or "default"


    def _normalize_bindings(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """标准化并自动合并同一群聊的历史 Key（如 group:123 与 default:GroupMessage:123）。"""
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            ck = self._canonical_key_str(str(k))
            if not ck:
                continue
            if isinstance(v, list):
                doc_ids = [str(i) for i in v]
                prompt = ""
                shield = False
                mode = "reference"
                force_sys = False
                extra: Dict[str, Any] = {}
            elif isinstance(v, dict):
                doc_ids = [str(i) for i in (v.get("doc_ids") or [])]
                prompt = str(v.get("prompt") or "").strip()
                shield = bool(v.get("shield", False))
                force_sys = bool(v.get("force_system_prompt", False))
                mode = self._normalize_mode(v.get("mode"))
                # 保留未知字段（如 ignore_history / cutoff_timestamp），避免打开WebUI就丢配置
                extra = {kk: vv for kk, vv in v.items() if kk not in (
                    "doc_ids", "prompt", "shield", "mode", "force_system_prompt")}
            else:
                continue

            if ck not in out:
                out[ck] = {"doc_ids": doc_ids, "prompt": prompt, "shield": shield, "mode": mode, "force_system_prompt": force_sys, **extra}
            else:
                cur = out[ck]
                for did in doc_ids:
                    if did not in cur["doc_ids"]:
                        cur["doc_ids"].append(did)
                if prompt and not cur.get("prompt"):
                    cur["prompt"] = prompt
                if shield:
                    cur["shield"] = True
                if force_sys:
                    cur["force_system_prompt"] = True
                if mode in ("system", "workspace"):
                    cur["mode"] = mode
                for kk, vv in extra.items():
                    cur.setdefault(kk, vv)
        return out


    def _session_keys(self, event: AstrMessageEvent) -> List[str]:
        ck = self._canonical_key(event)
        keys = [ck]
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        for cand in (umo, self._canonical_key_str(umo)):
            if cand and cand not in keys:
                keys.append(cand)
        return keys


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
        m = str(mode or "").lower().strip()
        if m in ("system", "sys", "强制", "提示词", "1"):
            return "system"
        if m in ("workspace", "ws", "工作区", "沙箱", "3"):
            return "workspace"
        return "reference"


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


    def set_session_prompt(self, session_key: str, prompt: str) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        ent["prompt"] = (prompt or "").strip()
        self._save_json(self.bindings_path, self._bindings)
        return ent


    def set_session_mode(self, session_key: str, mode: str) -> Dict[str, Any]:
        # 文档生效模式必须有绑定文档才能切换，无文档时强制回落 reference
        ent = self._get_entry(session_key)
        if not [d for d in ent.get("doc_ids", []) if d in self._index]:
            ent["mode"] = "reference"
            self._save_json(self.bindings_path, self._bindings)
            return ent
        ent["mode"] = self._normalize_mode(mode)
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


    def _cfg_int(self, key: str, default: int = 0) -> int:
        try:
            if not default:
                default = int(CONFIG_DEFAULTS.get(key, 0))
            v = int(self._cfg(key, default))
            return v if v > 0 else default
        except Exception:
            return default


    def _cfg_no_limit(self, key: str) -> int:
        """读取字符上限：<=0 表示不限制、保证完整注入（与 _cfg_int 强制正数不同）。"""
        try:
            return int(self._cfg(key, CONFIG_DEFAULTS.get(key, 0)))
        except Exception:
            return int(CONFIG_DEFAULTS.get(key, 0))
