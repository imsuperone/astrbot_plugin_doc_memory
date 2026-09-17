"""文档记忆助手插件：零 Embedding 检索 + 按群会话绑定 + 群独立提示词 + 大模型自动引用"""

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

PLUGIN_NAME = "astrbot_plugin_xbdoc"
OLD_PLUGIN_NAMES = ("xbdoc", "astrbot_plugin_doc_memory")

# 可选 Web API 依赖
try:
    from astrbot.api.web import (
        error_response,
        file_response,
        json_response,
        request,
    )
    _HAS_WEB_API = True
except Exception:
    _HAS_WEB_API = False

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
    _HAS_DATA_PATH = True
except Exception:
    _HAS_DATA_PATH = False

try:
    from astrbot.core.agent.message import TextPart
    _HAS_TEXT_PART = True
except Exception:
    _HAS_TEXT_PART = False
    TextPart = None  # type: ignore

# 子模块导入：优先包内相对导入，失败时回退到文件目录直载（兼容各类加载器）
try:
    from .xbdoc_retrieval import (
        ALLOWED_SUFFIXES,
        chunk_text,
        extract_text_from_bytes,
        format_tavern_to_markdown,
        parse_tavern_card,
        score_chunk_tf,
        tokenize,
    )
    from .xbdoc_inject import apply_system_prompt, build_system_text, build_workspace_text
except ImportError:
    import sys as _sys
    _plug_dir = str(Path(__file__).resolve().parent)
    if _plug_dir not in _sys.path:
        _sys.path.insert(0, _plug_dir)
    from xbdoc_retrieval import (
        ALLOWED_SUFFIXES,
        chunk_text,
        extract_text_from_bytes,
        format_tavern_to_markdown,
        parse_tavern_card,
        score_chunk_tf,
        tokenize,
    )
    from xbdoc_inject import apply_system_prompt, build_system_text, build_workspace_text


# ======================================================================
# 插件主体
# ======================================================================

class XbdocPlugin(Star):
    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        if config is not None:
            try:
                self.config = config
            except Exception:
                pass
        cfg = getattr(self, "config", None) or config or {}

        self.chunk_size = int(cfg.get("chunk_size", 1500) or 1500)
        self.chunk_overlap = int(cfg.get("chunk_overlap", 200) or 200)
        self.top_k = int(cfg.get("top_k", 3) or 3)
        self.max_inject_chars = int(cfg.get("max_inject_chars", 6000) or 6000)
        self.auto_inject = bool(cfg.get("auto_inject", True))
        self.allow_private_bind = bool(cfg.get("allow_private_bind", True))

        # 持久化存储路径
        self._latest_bot = None
        self.data_dir = self._resolve_data_dir()
        self.docs_dir = self.data_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.json"
        self.bindings_path = self.data_dir / "bindings.json"
        self.seen_path = self.data_dir / "seen_groups.json"

        # 加载并自动标准化数据
        self._index: Dict[str, Dict[str, Any]] = self._load_json(self.index_path, {})
        self._seen_groups: Dict[str, Dict[str, Any]] = self._load_json(self.seen_path, {})
        self._bindings: Dict[str, Dict[str, Any]] = self._normalize_bindings(
            self._load_json(self.bindings_path, {})
        )
        self._chunk_cache: Dict[str, List[str]] = {}  # 内存缓存：doc_id -> chunks
        self._chunk_tokens_cache: Dict[str, List[Counter]] = {}  # 性能优化：doc_id -> 每切片词频
        self._seen_save_ts = 0  # 群记录节流时间戳（仅内存，不落盘）

        if _HAS_WEB_API:
            try:
                self._register_web_apis()
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] 注册 Web API 异常: {e}")

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
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 保存 {path.name} 失败: {e}")

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
        """记录群聊基础信息，供 WebUI 模糊搜索使用。"""
        try:
            gid = str(event.get_group_id() or "").strip()
            if not gid:
                return
            platform = str(getattr(event, "platform_id", "") or getattr(event, "platform", "") or "")
            if not platform:
                umo = getattr(event, "unified_msg_origin", "") or ""
                platform = umo.split(":", 1)[0] if ":" in umo else ""

            group_name = ""
            grp = getattr(event.message_obj, "group", None)
            if grp is not None:
                group_name = str(getattr(grp, "group_name", "") or "").strip()

            now = int(time.time())
            ent = self._seen_groups.setdefault(gid, {
                "gid": gid, "group_name": group_name, "platform": platform,
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
                filename = f"【酒馆】{chara_name}.md"
            else:
                stem = Path(filename).stem
                filename = f"【酒馆】{stem}.md"
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
        (self.docs_dir / stored_name).write_bytes(data)

        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
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
        (self.data_dir / f"chunks_{doc_id}.json").write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
        )
        self._chunk_cache[doc_id] = chunks  # 更新内存缓存
        self._chunk_tokens_cache.pop(doc_id, None)  # 词频缓存失效，下次检索重建
        self._save_json(self.index_path, self._index)
        logger.info(f"[{PLUGIN_NAME}] 入库文档 {filename} id={doc_id} chunks={len(chunks)} tavern={is_tavern}")
        return meta

    def delete_document(self, doc_id: str) -> bool:
        meta = self._index.pop(doc_id, None)
        if not meta:
            return False
        for p in [self.docs_dir / str(meta.get("stored_name", "")), self.data_dir / f"chunks_{doc_id}.json"]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self._chunk_cache.pop(doc_id, None)  # 清除内存缓存
        self._chunk_tokens_cache.pop(doc_id, None)

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
            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
            cache.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
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

    def _peek_entry(self, session_key: str) -> Dict[str, Any]:
        """只读不创建，避免每次对话污染 bindings。"""
        return self._bindings.get(self._canonical_key_str(session_key)) or {}

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
        """获取本会话综合生效配置（严格群唯一化）。"""
        ck = self._canonical_key(event)
        ent = self._bindings.get(ck)
        if not ent:
            for k in self._session_keys(event):
                if k in self._bindings:
                    ent = self._bindings[k]
                    break
        ent = ent or {}
        doc_ids = [d for d in ent.get("doc_ids", []) if d in self._index]
        return {
            "doc_ids": doc_ids,
            "prompt": str(ent.get("prompt") or "").strip(),
            "shield": bool(ent.get("shield", False)),
            "mode": str(ent.get("mode") or "reference"),
            "force_system_prompt": bool(ent.get("force_system_prompt", False)),
            "matched_key": ck,
            "has_entry": bool(ent),
        }

    def bind_docs(self, session_key: str, doc_ids: List[str]) -> List[str]:
        valid = [d for d in doc_ids if d in self._index]
        self._get_entry(session_key)["doc_ids"] = valid
        self._save_json(self.bindings_path, self._bindings)
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

    def set_session_shield(self, session_key: str, shield: bool) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        ent["shield"] = bool(shield)
        self._save_json(self.bindings_path, self._bindings)
        return ent

    def set_session_mode(self, session_key: str, mode: str) -> Dict[str, Any]:
        # 允许预设模式（无文档也可设置，绑定后自动生效），与 WebUI 保持一致
        ent = self._get_entry(session_key)
        ent["mode"] = self._normalize_mode(mode)
        self._save_json(self.bindings_path, self._bindings)
        return ent

    # ---------- 动态配置读取 ----------
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            cfg = getattr(self, "config", None) or {}
            val = cfg.get(key, default) if hasattr(cfg, "get") else default
            return default if val is None else val
        except Exception:
            return default

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            v = int(self._cfg(key, default))
            return v if v > 0 else default
        except Exception:
            return default

    # ---------- 检索与上下文注入 ----------
    def retrieve(self, query: str, doc_ids: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self._cfg_int("top_k", self.top_k)
        clean_q = re.sub(r"@\S+", "", query or "").strip()
        qtokens = tokenize(clean_q)
        if not doc_ids:
            return []
        scored: List[Dict[str, Any]] = []
        q_lower = clean_q.lower()

        for did in doc_ids:
            meta = self._index.get(did)
            if not meta:
                continue
            fname = str(meta.get("filename", "")).lower()
            fstem = Path(fname).stem.lower()
            fname_hit = bool((fstem and fstem in q_lower) or (fname and fname in q_lower))

            chunks = self._load_chunks(did)
            counters = self._get_chunk_counters(did)
            for idx, ch in enumerate(chunks):
                tf = counters[idx] if idx < len(counters) else Counter()
                s = score_chunk_tf(qtokens, tf) if qtokens else 0.0
                if fname_hit:
                    s += 10.0
                if s > 0:
                    scored.append({
                        "doc_id": did, "filename": meta.get("filename", did),
                        "chunk_idx": idx, "text": ch, "score": s,
                    })

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:max(1, top_k)]

        # 保底策略：若未命中具体切片关键词，但本群确有绑定文档，保底载入首切片，确保模型拥有文档认知
        if not results and doc_ids:
            for did in doc_ids[:2]:
                chunks = self._load_chunks(did)
                if chunks:
                    meta = self._index.get(did, {})
                    results.append({
                        "doc_id": did, "filename": meta.get("filename", did),
                        "chunk_idx": 0, "text": chunks[0], "score": 0.5,
                    })

        return results

    def build_inject_text(self, query: str, doc_ids: List[str]) -> str:
        max_chars = self._cfg_int("max_inject_chars", self.max_inject_chars)
        hits = self.retrieve(query, doc_ids, self._cfg_int("top_k", self.top_k))
        if not hits:
            return ""
        parts = []
        total = 0
        for h in hits:
            seg = f"{h['filename']} (片段{h['chunk_idx']+1}):\n{h['text']}"
            if total + len(seg) > max_chars:
                remain = max_chars - total
                if remain > 100:
                    parts.append(seg[:remain] + "\n…(截断)")
                break
            parts.append(seg)
            total += len(seg)
        return "\n\n".join(parts)

    # ---------- LLM 钩子 ----------
    @filter.on_llm_request()
    async def _inject_docs(self, event: AstrMessageEvent, req):
        try:
            try:
                is_private = not event.get_group_id()
            except Exception:
                is_private = False
            if is_private and not bool(self._cfg("allow_private_bind", True)):
                return

            # 只读不创建：避免每次对话凭空制造空绑定（曾导致解绑后仍残留空条目）
            sess = self._effective_session(event)
            c_key = str(sess.get("matched_key") or self._canonical_key(event))
            doc_ids = [d for d in sess.get("doc_ids", []) if d in self._index]
            has_bound = bool(doc_ids)
            shield = bool(sess.get("shield", False))
            mode = str(sess.get("mode") or "reference").lower()
            custom_prompt = str(sess.get("prompt") or "").strip()
            ent_raw = self._bindings.get(c_key) or {}
            for k in self._session_keys(event):
                if k in self._bindings:
                    ent_raw = self._bindings[k]
                    break
            ignore_history = bool(ent_raw.get("ignore_history", False))

            # 0. /doc no 指令支持：彻底清空此前所有历史消息，不再读取与记忆
            if ignore_history:
                if hasattr(req, "contexts") and isinstance(req.contexts, list):
                    req.contexts.clear()
                if hasattr(req, "messages") and isinstance(req.messages, list):
                    req.messages = [
                        m for m in req.messages
                        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "system"
                    ]
                for h_attr in ("history", "chat_history"):
                    h_val = getattr(req, h_attr, None)
                    if isinstance(h_val, list):
                        h_val.clear()

            # -------------------------------------------------------------
            # 提示词通道：shield 或 force 任一开启即替换原人格，否则追加。
            # 有绑定文档时由各模式统一拼入提示词（仅一次），避免重复注入。
            # -------------------------------------------------------------
            force_sys = bool(sess.get("force_system_prompt", False))
            replace_all = bool(shield or force_sys)
            max_chars = self._cfg_int("max_inject_chars", self.max_inject_chars)
            if not has_bound and custom_prompt:
                apply_system_prompt(req, custom_prompt, replace=replace_all)
                logger.info(f"[{PLUGIN_NAME}] [专属系统词模式] 无文档，专属系统提示词独立生效 (会话: {c_key})")
                return

            # 无文档且无提示词时的空载响应
            if not has_bound:
                if shield:
                    apply_system_prompt(req, "", replace=True)
                return

            # -------------------------------------------------------------
            # 模式 1：⚡ 强制遵守文档 (文档直接作为系统提示词，专属提示词作为额外附加提示词)
            # -------------------------------------------------------------
            if mode == "system" and has_bound:
                doc_texts = ["\n".join(self._load_chunks(did)) for did in doc_ids]
                apply_system_prompt(
                    req, build_system_text(doc_texts, custom_prompt, max_chars), replace=replace_all
                )

                # 保持 req.prompt 纯净，绝不向用户发言拼入文档正文
                logger.info(f"[{PLUGIN_NAME}] [强制遵守模式] 文档已作为系统提示词载入 (会话: {c_key})")
                return

            # -------------------------------------------------------------
            # 模式 3：💻 模拟工作区模式 (需有绑定文档，否则回落到参考资料逻辑)
            # -------------------------------------------------------------
            if mode == "workspace" and has_bound:
                files = []
                for did in doc_ids:
                    meta = self._index.get(did, {})
                    files.append((meta.get("filename", did), "\n".join(self._load_chunks(did))))
                apply_system_prompt(
                    req, build_workspace_text(files, custom_prompt, max_chars), replace=replace_all
                )

                # 保持 req.prompt 纯净，绝不向用户发言拼入工作区文件
                logger.info(f"[{PLUGIN_NAME}] [工作区模式] 纯净挂载工作区文件 (会话: {c_key})")
                return

            # -------------------------------------------------------------
            # 模式 2：📖 仅作参考资料 (提示词按 shield/force 决定替换或追加，必生效一次)
            # -------------------------------------------------------------
            apply_system_prompt(req, custom_prompt, replace=replace_all)

            if not has_bound:
                return

            # auto_inject 关闭时不再自动检索文档（专属提示词/屏蔽仍生效）
            if not bool(self._cfg("auto_inject", True)):
                return

            query = str(getattr(req, "prompt", "") or event.message_str or "").strip()
            inject = self.build_inject_text(query, doc_ids)
            if not inject:
                return

            # 写入 extra_user_content_parts 官方标准通道；缺失时回退拼接到 prompt
            injected = False
            parts = getattr(req, "extra_user_content_parts", None)
            if parts is not None and _HAS_TEXT_PART and TextPart is not None:
                try:
                    tp = TextPart(text=f"【参考资料】\n{inject}")
                    if hasattr(tp, "mark_as_temp"):
                        tp = tp.mark_as_temp()
                    parts.append(tp)
                    injected = True
                except Exception:
                    injected = False
            if not injected:
                try:
                    base = str(getattr(req, "prompt", "") or "")
                    req.prompt = f"{base}\n\n【参考资料】\n{inject}".strip()
                except Exception:
                    pass

            logger.info(f"[{PLUGIN_NAME}] [参考资料模式] 纯净载入文档记忆 (会话: {c_key})")
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 上下文注入异常: {e}")

    # ---------- 消息捕获与群记录 ----------
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def _seen_collector(self, event: AstrMessageEvent):
        try:
            bot = getattr(event, "bot", None)
            if bot is not None:
                self._latest_bot = bot
        except Exception:
            pass
        self._record_seen_group(event)

    # ---------- LLM Tool ----------
    @filter.llm_tool(name="doc_memory_search")
    async def _tool_search(self, event: AstrMessageEvent, query: str):
        """搜索本会话绑定的文档内容。

        Args:
            query(string): 要在文档中搜索的关键词或问题
        """
        doc_ids = self.get_bound_doc_ids(event)
        if not doc_ids:
            yield event.plain_result("本会话尚未绑定任何文档。")
            return
        hits = self.retrieve(query or event.message_str or "", doc_ids, self._cfg_int("top_k", self.top_k))
        if not hits:
            yield event.plain_result("在绑定文档中未匹配到相关内容。")
            return
        out = "\n---\n".join(f"【{h['filename']}#片段{h['chunk_idx']+1}】\n{h['text'][:1200]}" for h in hits)
        yield event.plain_result(out[:3500])

    @filter.llm_tool(name="doc_memory_list")
    async def _tool_list(self, event: AstrMessageEvent):
        """列出本会话已绑定的文档清单。"""
        doc_ids = self.get_bound_doc_ids(event)
        if not doc_ids:
            yield event.plain_result("本会话尚未绑定任何文档。")
            return
        lines = [f"- {self._index[d]['filename']} (id={d})" for d in doc_ids if d in self._index]
        yield event.plain_result("本会话绑定的文档：\n" + "\n".join(lines))

    # ---------- 聊天指令 ----------
    @filter.command("doc")
    async def doc_cmd(self, event: AstrMessageEvent):
        """文档记忆助手统一指令入口 /doc [子指令]"""
        raw = (event.message_str or "").strip()
        tokens = [t for t in re.split(r"\s+", raw) if t]
        sub = tokens[1].lower() if len(tokens) > 1 else ""
        sub_args = tokens[2:] if len(tokens) > 2 else []

        if not sub or sub in ("help", "h", "?", "帮助", "菜单"):
            async for res in self._cmd_help(event):
                yield res
            return

        # 管理员指令校验（私聊全功能放行，群聊校验管理员）
        admin_subs = {"bind", "unbind", "mode", "shield", "force", "prompt_set", "prompt_clear", "no"}
        if sub in admin_subs:
            try:
                if event.get_group_id() and not event.is_admin():
                    yield event.plain_result("⚠️ 权限不足：该指令在群聊中仅限群主或管理员使用。")
                    return
            except Exception:
                pass

        if sub == "list":
            async for res in self.doc_list(event):
                yield res
        elif sub == "status":
            async for res in self.doc_status(event):
                yield res
        elif sub == "bind":
            async for res in self.doc_bind(event, *sub_args):
                yield res
        elif sub == "unbind":
            async for res in self.doc_unbind(event, *sub_args):
                yield res
        elif sub == "mode":
            async for res in self.doc_mode(event, *sub_args):
                yield res
        elif sub in ("workspace", "ws", "工作区"):
            async for res in self.doc_workspace(event):
                yield res
        elif sub == "shield":
            async for res in self.doc_shield(event, *sub_args):
                yield res
        elif sub == "force":
            async for res in self.doc_force(event, *sub_args):
                yield res
        elif sub in ("greeting", "greet", "intro", "开场白", "问候"):
            async for res in self.doc_greeting(event):
                yield res
        elif sub == "search":
            kw = " ".join(sub_args)
            async for res in self.doc_search(event, kw):
                yield res
        elif sub == "read":
            did = sub_args[0] if sub_args else ""
            n = sub_args[1] if len(sub_args) > 1 else "1"
            async for res in self.doc_read(event, did, n):
                yield res
        elif sub == "prompt":
            async for res in self.doc_prompt(event):
                yield res
        elif sub == "prompt_set":
            async for res in self.doc_prompt_set(event):
                yield res
        elif sub == "prompt_clear":
            async for res in self.doc_prompt_clear(event):
                yield res
        elif sub in ("no", "forget", "clear_history", "重置记忆"):
            async for res in self.doc_no(event, *sub_args):
                yield res
        else:
            yield event.plain_result(f"❓ 未知子指令「{sub}」，发送 /doc 可查看可用指令菜单。")

    async def _cmd_help(self, event: AstrMessageEvent):
        menu = (
            "📚 文档记忆助手 · 指令菜单\n\n"
            "📖 查看\n"
            "• /doc status — 本群状态；/doc list — 文档列表\n"
            "• /doc workspace — 工作区挂载；/doc greeting — 角色开场白\n"
            "• /doc search <词> — 检索绑定文档；/doc read <ID> [n] — 预览切片\n\n"
            "🔗 绑定（管理员）\n"
            "• /doc bind <ID...> — 追加绑定，自动合并\n"
            "• /doc unbind [ID...] — 解绑，留空全清（含提示词/屏蔽）\n\n"
            "🎛️ 模式（管理员）\n"
            "• /doc mode system|workspace|reference — 切换生效模式\n"
            "• /doc shield on|off — 清空/保留原人格\n"
            "• /doc force on|off — 专属提示词唯一生效\n\n"
            "🏷️ 提示词（管理员）\n"
            "• /doc prompt — 查看；/doc prompt_set <内容> — 设置\n"
            "• /doc prompt_clear — 清除\n\n"
            "🧹 历史（管理员）\n"
            "• /doc no [off] — 忘掉此前消息 / 恢复\n\n"
            "💡 模式：system 强制遵守 · workspace 工作区 · reference 仅参考"
        )
        yield event.plain_result(menu)

    async def doc_list(self, event: AstrMessageEvent):
        """查看知识库中所有文档 /doc list"""
        docs = self.list_documents()
        if not docs:
            yield event.plain_result(
                "📚 知识库当前暂无入库文档。\n\n"
                "请在 WebUI 后端管理台上传文档后再进行绑定。"
            )
            return

        lines = [f"📚 知识库文档列表（共 {len(docs)} 篇）\n"]
        for idx, m in enumerate(docs[:30], 1):
            lines.append(f"[{idx}] {m['filename']}")
            lines.append(f"• 文档ID：{m['doc_id']}")
            lines.append(f"• 规模：{m['chunks']} 切片 · {m['text_len']:,} 字\n")

        lines.append("💡 绑定到本群：/doc bind <文档ID>")
        yield event.plain_result("\n".join(lines).strip())

    async def doc_status(self, event: AstrMessageEvent):
        """查看本会话绑定的文档 /doc status"""
        ids = self.get_bound_doc_ids(event)
        keys = self._session_keys(event)
        sess = self._effective_session(event)
        curr_key = keys[0] if keys else "(未知)"
        shield_txt = "🛡️ 开启（已清空原人格）" if sess.get("shield") else "👤 关闭（保留原人格）"
        mode_txt = "💻 模拟工作区（仅限工作区文档）" if sess.get("mode") == "workspace" else ("⚡ 强制遵守（系统提示词模式）" if sess.get("mode") == "system" else "📖 仅作参考资料（按需检索）")
        has_prompt = bool(sess.get("prompt"))
        prompt_txt = f"已设置（{len(sess['prompt'])}字）" if has_prompt else "未设置"
        force_sys = bool(sess.get("force_system_prompt"))
        force_txt = "⚡ 开启（清空其他提示词，专属提示词唯一生效）" if force_sys else "关闭"

        lines = [
            "📌 本群文档记忆状态\n",
            f"• 会话标识：{curr_key}",
            f"• 生效模式：{mode_txt}",
            f"• 人格屏蔽：{shield_txt}",
            f"• 强制系统词：{force_txt}",
            f"• 专属提示词：{prompt_txt}\n",
        ]
        if not ids:
            if has_prompt:
                lines.append("💡 当前未绑定文档，专属系统提示词正常独立生效中。")
                if force_sys:
                    lines.append("⚡ 强制注入模式已激活：已清空其他提示词，专属提示词作为底层唯一系统词。")
            else:
                lines.append("⚠️ 本群当前未绑定任何文档。")
                lines.append("💡 发送 /doc list 查看可用文档，或发送 /doc bind <ID> 快速绑定。")
        else:
            lines.append(f"📖 已绑定文档（共 {len(ids)} 篇）：")
            for idx, d in enumerate(ids, 1):
                meta = self._index.get(d, {})
                fname = meta.get("filename", d)
                lines.append(f"{idx}. {fname}（ID: {d}）")
            lines.append("")
            if sess.get("mode") == "workspace":
                lines.append("💻 说明：当前会话处于独立工作区沙箱，大模型仅对工作区内的挂载文档进行严谨分析与回答。")
            elif sess.get("mode") == "system":
                lines.append("⚡ 说明：大模型已将文档作为最高系统设定执行，强制遵守文档规则与设定。")
            else:
                lines.append("📖 说明：群内提问相关内容时，AI 将检索片段作为参考资料引用回答。")
            has_tavern = any(self._index.get(d, {}).get("is_tavern") for d in ids)
            if has_tavern:
                lines.append("🍷 提示：当前包含酒馆角色卡，可发送 /doc greeting 查看角色开场白。")
            lines.append("\n💡 切换模式：/doc mode workspace / system / reference")
            lines.append("💡 查看工作区：/doc workspace")
            lines.append("💡 切换屏蔽：/doc shield on / off")
        yield event.plain_result("\n".join(lines).strip())

    async def doc_workspace(self, event: AstrMessageEvent):
        """查看当前模拟工作区状态与文件清单 /doc workspace"""
        ids = self.get_bound_doc_ids(event)
        sess = self._effective_session(event)
        key = str(sess.get("matched_key") or self._canonical_key(event))
        mode = str(sess.get("mode") or "reference")

        if not ids:
            yield event.plain_result(
                f"💻 模拟工作区详情（{key}）\n\n"
                f"• 当前模式：{'💻 模拟工作区模式 (生效中)' if mode == 'workspace' else '📖 普通模式'}\n"
                "⚠️ 当前工作区尚未挂载用户文档。\n"
                "💡 发送 /doc list 查看可用文档，使用 /doc bind <ID> 挂载文件到工作区。"
            )
            return

        lines = [
            f"💻 模拟工作区详情（{key}）\n",
            f"• 当前模式：{'💻 模拟工作区模式 (生效中)' if mode == 'workspace' else '📖 普通模式 (发送 /doc mode workspace 切换为工作区)'}",
            f"• 挂载文件数量：共 {len(ids)} 篇文档\n",
            "📁 工作区根目录 [/workspace] 文件清单：",
        ]
        total_len = 0
        for idx, did in enumerate(ids, 1):
            meta = self._index.get(did, {})
            fname = meta.get("filename", did)
            tlen = meta.get("text_len", 0)
            total_len += tlen
            lines.append(f"{idx}. /workspace/{fname}")
            lines.append(f"   ├─ ID: {did}")
            lines.append(f"   └─ 大小: {meta.get('chunks', 1)} 切片 · {tlen:,} 字符")

        lines.append(f"\n📊 工作区总文本容量：{total_len:,} 字符")
        if mode != "workspace":
            lines.append("\n💡 发送 /doc mode workspace 可切换为工作区模式。")
        yield event.plain_result("\n".join(lines).strip())

    async def doc_greeting(self, event: AstrMessageEvent):
        """查看已绑定酒馆角色卡的开场白 /doc greeting"""
        ids = self.get_bound_doc_ids(event)
        if not ids:
            yield event.plain_result("⚠️ 本群当前未绑定任何文档或酒馆角色卡。")
            return
        greetings = []
        for did in ids:
            meta = self._index.get(did, {})
            g = str(meta.get("greeting") or "").strip()
            if g:
                fname = meta.get("filename", did)
                greetings.append(f"🍷《{fname}》角色开场白：\n\n{g}")
        if not greetings:
            yield event.plain_result("💡 本群当前绑定的文档未包含酒馆角色开场白数据（first_mes）。")
            return
        yield event.plain_result("\n\n────────────────────────\n\n".join(greetings))

    async def doc_bind(self, event: AstrMessageEvent, doc_id: str = "", *rest: str):
        """绑定文档 /doc bind <id1> [id2...]（追加到本群已有绑定，管理员）"""
        raw_tokens = [t for t in re.split(r"\s+", (event.message_str or "").strip()) if t][2:]
        ids = self._parse_doc_ids(doc_id, " ".join(raw_tokens))
        if not ids:
            yield event.plain_result(
                "❌ 用法错误：/doc bind <文档ID1> [文档ID2...]\n"
                "💡 可先发送 /doc list 查看知识库中可用的文档 ID。"
            )
            return
        bad = [i for i in ids if i not in self._index]
        if bad:
            yield event.plain_result(f"❌ 绑定失败：以下 ID 不存在于知识库中：\n{', '.join(bad)}\n\n💡 请发送 /doc list 查看可用 ID。")
            return
        key = self._canonical_key(event)
        existed = [d for d in self._peek_entry(key).get("doc_ids", []) if d in self._index]
        added = [i for i in ids if i not in existed]
        dup = [i for i in ids if i in existed]
        self.bind_docs(key, existed + added)
        ent = self._peek_entry(key)
        m = str(ent.get("mode") or "reference")
        mode_txt = "⚡ 强制遵守（系统提示词）" if m == "system" else ("💻 模拟工作区" if m == "workspace" else "📖 仅作参考资料")

        lines = [
            f"✅ 绑定成功！已关联到本群（{key}）：\n",
        ]
        for did in added:
            lines.append(f"• 新增：{self._index[did]['filename']}（ID: {did}）")
        for did in dup:
            lines.append(f"• 已在绑定中：{self._index[did]['filename']}（ID: {did}）")
        lines.append(f"\n本群共绑定 {len(existed) + len(added)} 篇，当前模式：{mode_txt}")
        lines.append("💡 切换为强制遵守模式：/doc mode system")
        lines.append("💡 切换为参考资料模式：/doc mode reference")
        yield event.plain_result("\n".join(lines).strip())

    async def doc_unbind(self, event: AstrMessageEvent, doc_id: str = "", *rest: str):
        """解绑文档 /doc unbind [id...]，留空则清空绑定（管理员）"""
        keys = self._find_matching_keys(event)
        main_key = keys[0]
        targets = [k for k in keys if (self._bindings.get(k) or {}).get("doc_ids")]
        if not targets:
            yield event.plain_result(f"⚠️ 本群（{main_key}）当前未绑定任何文档。")
            return
        raw_tokens = [t for t in re.split(r"\s+", (event.message_str or "").strip()) if t][2:]
        # 留空 = 清空全部（含历史重复 Key）：文档、提示词、屏蔽、强制注入一并清除，回到未配置状态
        if not doc_id and not raw_tokens:
            for k in targets:
                ent = self._bindings.get(k) or {}
                ent["doc_ids"] = []
                ent["prompt"] = ""
                ent["shield"] = False
                ent["force_system_prompt"] = False
                ent["mode"] = "reference"
            pruned = sum(1 for k in targets if self._prune_empty_entry(k))
            self._save_json(self.bindings_path, self._bindings)
            tail = "相关配置已彻底移除。" if pruned else "已回到默认配置。"
            yield event.plain_result(f"✅ 已清空本群（{main_key}）的所有文档绑定，专属提示词、屏蔽与强制注入已一并清除。{tail}")
            return
        tokens = self._parse_doc_ids(doc_id, " ".join(raw_tokens))
        removed: List[str] = []
        not_found: List[str] = []
        for t in tokens:
            hit = False
            for k in targets:
                ent = self._bindings.get(k) or {}
                if t in ent.get("doc_ids", []):
                    ent["doc_ids"] = [d for d in ent["doc_ids"] if d != t]
                    hit = True
            (removed if hit else not_found).append(t)
        # 若解绑后已无文档，视为彻底解绑：提示词/屏蔽/强制一并清除，与清空解绑保持一致
        remaining = sum(len((self._bindings.get(k) or {}).get("doc_ids", [])) for k in targets)
        if remaining == 0:
            for k in targets:
                ent = self._bindings.get(k) or {}
                ent["prompt"] = ""
                ent["shield"] = False
                ent["force_system_prompt"] = False
                ent["mode"] = "reference"
            for k in targets:
                self._prune_empty_entry(k)
        self._save_json(self.bindings_path, self._bindings)
        msg = f"✅ 解绑完成（{main_key}）："
        if removed:
            msg += f"\n• 已移除：{', '.join(removed)}"
        if not_found:
            msg += f"\n• 未绑定/不存在：{', '.join(not_found)}"
        if remaining == 0:
            msg += "\n• 本群已无绑定文档，提示词与屏蔽已一并清除。"
        else:
            msg += f"\n本群当前剩余：{remaining} 篇文档。"
        yield event.plain_result(msg)

    async def doc_search(self, event: AstrMessageEvent, keyword: str = ""):
        """检索绑定文档 /doc search <关键词>"""
        q = (keyword or re.sub(r"^/doc\s+search\s*", "", event.message_str or "")).strip()
        if not q:
            yield event.plain_result("❌ 用法错误：/doc search <关键词或提问内容>")
            return
        ids = self.get_bound_doc_ids(event)
        if not ids:
            yield event.plain_result("⚠️ 本群尚未绑定任何文档，请先使用 /doc bind <ID> 绑定。")
            return
        hits = self.retrieve(q, ids, self._cfg_int("top_k", self.top_k))
        if not hits:
            yield event.plain_result(f"🔍 未在已绑定文档中检索到与「{q}」相关的片段，可尝试更换搜索词。")
            return
        out = [f"🔍 检索结果（关键词：{q}，匹配 {len(hits)} 处）\n"]
        for idx, h in enumerate(hits, 1):
            out.append(f"【{idx}】《{h['filename']}》片段{h['chunk_idx']+1}（相关度: {h['score']}）")
            out.append(f"{h['text'][:400]}\n")
        yield event.plain_result("\n".join(out)[:3500].strip())

    async def doc_read(self, event: AstrMessageEvent, doc_id: str = "", num: str = "1", *rest: str):
        """预览文档切片 /doc read <id> [片段号]"""
        if not doc_id or doc_id not in self._index:
            yield event.plain_result("❌ 用法错误：/doc read <文档ID> [片段号]，ID 可用 /doc list 查看。")
            return
        try:
            n = max(1, int(num or "1"))
        except Exception:
            n = 1
        chunks = self._load_chunks(doc_id)
        if not chunks:
            yield event.plain_result("⚠️ 该文档暂无可用文本切片。")
            return
        n = min(n, len(chunks))
        meta = self._index[doc_id]
        yield event.plain_result(
            f"📄 预览《{meta['filename']}》（ID: {doc_id}）\n"
            f"进度：片段 {n} / {len(chunks)}\n\n"
            f"{chunks[n-1][:1500]}"
        )

    async def doc_prompt(self, event: AstrMessageEvent):
        """查看本群提示词、生效模式与屏蔽状态 /doc prompt"""
        sess = self._effective_session(event)
        doc_ids = sess.get("doc_ids", [])
        eff_prompt = str(sess.get("prompt") or "").strip()
        eff_shield = bool(sess.get("shield", False))
        mode = str(sess.get("mode") or "reference")
        preview = (eff_prompt[:260] + "…") if len(eff_prompt) > 260 else eff_prompt
        shield_desc = "🛡️ 已开启（清空原人格）" if eff_shield else "👤 已关闭（保留原人格）"
        if mode == "system":
            mode_desc = "⚡ 强制AI遵守文档（系统提示词）"
        elif mode == "workspace":
            mode_desc = "💻 模拟工作区（工作区文件挂载）"
        else:
            mode_desc = "📖 仅作参考资料（按需检索）"

        yield event.plain_result(
            "🧩 本群配置详情\n\n"
            f"• 生效模式：{mode_desc}\n"
            f"• 人格屏蔽：{shield_desc}\n"
            f"• 绑定文档：{len(doc_ids)} 篇\n"
            f"• 专属提示词：\n{preview or '（未设置）'}\n\n"
            "⚙️ 管理指令：\n"
            "• /doc mode system | reference\n"
            "• /doc shield on | off\n"
            "• /doc prompt_set <内容>\n"
            "• /doc prompt_clear"
        )

    async def doc_mode(self, event: AstrMessageEvent, mode: str = "", *rest: str):
        """设置本群文档生效模式 /doc mode workspace|system|reference（管理员）"""
        key = self._canonical_key(event)
        ent = self._peek_entry(key)
        raw = (mode or re.sub(r"^/doc\s+mode\s*", "", event.message_str or "")).strip().lower()
        norm = self._normalize_mode(raw) if raw else ""
        if norm == "workspace":
            self.set_session_mode(key, "workspace")
            yield event.plain_result(
                f"💻 本群模式已切换为【模拟工作区】！\n\n"
                f"当前会话已挂载进入独立工作区沙箱 (/workspace)，上下文中【仅包含】绑定的文档文件，模型将严格基于工作区文件进行专业分析、开发与问答。\n"
                f"💡 可发送 /doc workspace 查看工作区挂载清单。"
            )
        elif norm == "system":
            self.set_session_mode(key, "system")
            yield event.plain_result(
                f"⚡ 本群模式已切换为【强制遵守文档】！\n\n"
                f"文档将直接作为最高优先级系统提示词载入大模型，AI 将严格遵循文档中的一切角色设定、语言规范与指令要求。"
            )
        elif norm == "reference" and raw:
            self.set_session_mode(key, "reference")
            yield event.plain_result(
                f"📖 本群模式已切换为【仅作参考资料】！\n\n"
                f"文档将作为外部知识库，仅在群友提问相关内容时检索片段供 AI 参考回答。"
            )
        else:
            cur = "💻 模拟工作区" if ent.get("mode") == "workspace" else ("⚡ 强制遵守（系统提示词）" if ent.get("mode") == "system" else "📖 仅作参考资料")
            yield event.plain_result(
                f"📌 当前群生效模式：{cur}\n\n"
                "切换指令：\n"
                "• /doc mode workspace（模拟工作区，仅限工作区文档）\n"
                "• /doc mode system（强制遵守文档，角色与指令模式）\n"
                "• /doc mode reference（仅作参考资料，知识库问答）"
            )

    async def doc_prompt_set(self, event: AstrMessageEvent):
        """设置本群专属提示词 /doc prompt_set <内容>（管理员）"""
        text = re.sub(r"^/doc\s+prompt_set\s*", "", event.message_str or "").strip()
        if len(text) < 2:
            yield event.plain_result("用法：/doc prompt_set <本群专属提示词内容>，至少2个字。")
            return
        if len(text) > 4000:
            yield event.plain_result("提示词超出 4000 字上限，请精简后重试。")
            return
        key = self._canonical_key(event)
        self.set_session_prompt(key, text)
        yield event.plain_result(
            f"✅【本群专属提示词已生效】\n"
            f"会话标识：{key}\n"
            f"提示词字数：{len(text)} 字\n\n"
            f"💡 可发送 /doc prompt 查看详情，发送 /doc prompt_clear 可清除。"
        )

    async def doc_prompt_clear(self, event: AstrMessageEvent):
        """清空本群提示词 /doc prompt_clear（管理员）"""
        key = self._canonical_key(event)
        ent = self._peek_entry(key)
        if not ent:
            yield event.plain_result(f"⚠️ 本群（{key}）当前未设置专属提示词。")
            return
        ent["prompt"] = ""
        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        yield event.plain_result(f"✅ 已清空本群（{key}）专属提示词。")

    async def doc_shield(self, event: AstrMessageEvent, mode: str = "", *rest: str):
        """本群屏蔽 AstrBot 原人格开关 /doc shield on|off（管理员）"""
        raw = (mode or re.sub(r"^/doc\s+shield\s*", "", event.message_str or "")).strip().lower()
        key = self._canonical_key(event)
        cur_shield = bool(self._peek_entry(key).get("shield", False))

        if raw in ("on", "开", "1", "true"):
            target_shield = True
        elif raw in ("off", "关", "0", "false"):
            target_shield = False
        elif not raw:
            target_shield = not cur_shield
        else:
            yield event.plain_result("❌ 用法错误：/doc shield on（开启） | off（关闭）")
            return

        self.set_session_shield(key, target_shield)
        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        if target_shield:
            yield event.plain_result(f"🛡️ 本群已开启人格屏蔽！已彻底清空 AstrBot 自带人格，进入纯文档/提示词模式。")
        else:
            yield event.plain_result(f"👤 本群已关闭人格屏蔽！已恢复 AstrBot 原有人格。")

    async def doc_force(self, event: AstrMessageEvent, *args):
        """切换强制注入系统提示词开关 /doc force on|off（管理员）"""
        raw = " ".join(args).strip().lower()
        key = self._canonical_key(event)
        cur = bool(self._peek_entry(key).get("force_system_prompt", False))

        if raw in ("on", "开", "1", "true"):
            target = True
        elif raw in ("off", "关", "0", "false"):
            target = False
        elif not raw:
            target = not cur
        else:
            yield event.plain_result("用法：/doc force on (开启强制注入) | off (关闭)")
            return

        ent = self._get_entry(key)
        ent["force_system_prompt"] = target
        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        if target:
            yield event.plain_result(f"⚡【强制注入系统提示词已开启】\n会话（{key}）：将清空其他一切提示词，强制本群专属提示词为唯一底层系统提示词。")
        else:
            yield event.plain_result(f"✅【强制注入系统提示词已关闭】\n会话（{key}）：已恢复正常模式。")

    async def doc_no(self, event: AstrMessageEvent, *args):
        """清空历史记忆并停止读取此指令之前的消息 /doc no [off]"""
        raw = " ".join(args).strip().lower()
        key = self._canonical_key(event)

        if raw in ("off", "恢复", "false", "0", "no_off", "reset", "yes"):
            ent = self._peek_entry(key)
            if ent:
                ent["ignore_history"] = False
                self._save_json(self.bindings_path, self._bindings)
            yield event.plain_result(f"✅ 已恢复读取历史消息上下文（会话：{key}）。")
            return

        ent = self._get_entry(key)
        ent["ignore_history"] = True
        ent["cutoff_timestamp"] = int(time.time())
        self._save_json(self.bindings_path, self._bindings)

        # 同步重置当前底层对话会话 ID（彻底隔离历史轮次）
        try:
            for s_attr in ("session", "_session", "conversation"):
                sess_obj = getattr(event, s_attr, None)
                if sess_obj is not None:
                    for cid_attr in ("cid", "curr_cid", "conversation_id"):
                        if hasattr(sess_obj, cid_attr):
                            setattr(sess_obj, cid_attr, hashlib.md5(f"{key}:{time.time()}".encode()).hexdigest()[:8])
        except Exception:
            pass

        yield event.plain_result(
            f"🧹【已清空历史消息记忆】\n\n"
            f"本群（{key}）已彻底清空并停止读取此指令之前的所有消息！\n"
            "此前哪怕有聊天记录也会全部忘掉，后续仅响应当前提问与绑定文档。\n\n"
            "💡 如需恢复读取历史聊天：/doc no off"
        )

    # ---------- WebUI 后端 API ----------
    def _register_web_apis(self) -> None:
        assert _HAS_WEB_API
        reg = self.context.register_web_api
        reg(f"/{PLUGIN_NAME}/docs", self._api_list_docs, ["GET"], "列出文档")
        reg(f"/{PLUGIN_NAME}/docs/upload", self._api_upload_doc, ["POST"], "上传文档")
        reg(f"/{PLUGIN_NAME}/docs/delete", self._api_delete_doc, ["POST"], "删除文档")
        reg(f"/{PLUGIN_NAME}/docs/content", self._api_doc_content, ["GET"], "预览文档")
        reg(f"/{PLUGIN_NAME}/docs/download", self._api_doc_download, ["GET"], "下载文档")
        reg(f"/{PLUGIN_NAME}/groups", self._api_list_groups, ["GET"], "列出已见群聊")
        reg(f"/{PLUGIN_NAME}/groups/fetch", self._api_fetch_groups, ["POST", "GET"], "主动拉取机器人所在群")
        reg(f"/{PLUGIN_NAME}/bindings", self._api_list_bindings, ["GET"], "列出绑定")
        reg(f"/{PLUGIN_NAME}/bindings/save", self._api_save_binding, ["POST"], "保存绑定")

    async def _api_list_docs(self):
        return json_response({"docs": self.list_documents()})

    async def _api_upload_doc(self):
        import asyncio
        import base64

        filename = ""
        data = b""

        # 1. 尝试从 base64 JSON 直传（兼容 iframe 桥接及各种跨域环境）
        try:
            payload = await request.json(default={})
            if isinstance(payload, dict):
                b64 = str(payload.get("file_base64") or payload.get("data") or "").strip()
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                if b64:
                    data = base64.b64decode(b64)
                    filename = str(payload.get("filename") or "unnamed").strip()
        except Exception:
            data = b""

        # 2. 若无 base64，尝试从 multipart/form-data 解析
        if not data:
            upload = None
            try:
                files = await request.files()
                if isinstance(files, dict):
                    upload = files.get("file") or files.get("files") or files.get("upload") or (next(iter(files.values())) if files else None)
                elif hasattr(files, "filename") or hasattr(files, "read"):
                    upload = files
            except Exception:
                pass

            if not upload:
                try:
                    form = await request.form()
                    if isinstance(form, dict):
                        upload = form.get("file") or form.get("files") or form.get("upload") or (next(iter(form.values())) if form else None)
                except Exception:
                    pass

            if upload is not None:
                filename = getattr(upload, "filename", None) or getattr(upload, "name", None) or "unnamed"
                val = upload.read() if hasattr(upload, "read") else bytes(upload)
                if asyncio.iscoroutine(val):
                    data = await val
                else:
                    data = bytes(val) if val is not None else b""

        if not data:
            return error_response("未读取到上传文件内容，请重试", status_code=400)

        try:
            meta = self.add_document(filename, data)
            return json_response({"ok": True, "doc": meta})
        except RuntimeError as e:
            return error_response(str(e), status_code=400)
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 上传入库异常: {e}")
            return error_response(f"入库失败: {e}", status_code=500)

    async def _api_delete_doc(self):
        payload = await request.json(default={})
        doc_id = str(payload.get("doc_id", "")).strip()
        if not doc_id:
            return error_response("缺少 doc_id", status_code=400)
        if not self.delete_document(doc_id):
            return error_response("文档不存在", status_code=404)
        return json_response({"ok": True})

    async def _api_doc_content(self):
        doc_id = request.query.get("doc_id", "", type=str)
        chunk = request.query.get("chunk", 1, type=int)
        meta = self._index.get(doc_id)
        if not meta:
            return error_response("文档不存在", status_code=404)
        chunks = self._load_chunks(doc_id)
        n = max(1, min(int(chunk or 1), max(1, len(chunks))))
        preview = chunks[n - 1][:3000] if chunks else ""
        return json_response({"meta": meta, "chunk": n, "total": len(chunks), "preview": preview})

    async def _api_doc_download(self):
        doc_id = request.query.get("doc_id", "", type=str)
        meta = self._index.get(doc_id)
        if not meta:
            return error_response("文档不存在", status_code=404)
        path = self.docs_dir / str(meta.get("stored_name", ""))
        if not path.exists():
            return error_response("原文件丢失", status_code=404)
        return file_response(path, filename=str(meta.get("filename", "file")))

    async def _api_list_bindings(self):
        # 自动执行规范化去重合并，彻底清理不同端产生的重复 Key
        self._bindings = self._normalize_bindings(self._bindings)
        enriched = {}
        for k, ent in self._bindings.items():
            ids = ent.get("doc_ids", [])
            ks = str(k).strip()
            if ks.startswith("private:"):
                gid = ks.split(":", 1)[1]
            elif ks.startswith("group:"):
                gid = ks.split(":", 1)[1]
            else:
                gid = ks if ks.isdigit() else ks.split(":")[-1]
            gname = ""
            if gid and gid in self._seen_groups:
                gname = str(self._seen_groups[gid].get("group_name") or "").strip()

            enriched[k] = {
                "gid": gid,
                "group_name": gname,
                "docs": [{"doc_id": d, "filename": self._index.get(d, {}).get("filename", d)} for d in ids],
                "prompt": ent.get("prompt", ""),
                "shield": bool(ent.get("shield", False)),
                "force_system_prompt": bool(ent.get("force_system_prompt", False)),
                "mode": str(ent.get("mode") or "reference"),
            }
        return json_response({"bindings": enriched, "docs": self.list_documents()})

    async def _api_save_binding(self):
        payload = await request.json(default={})
        raw_key = str(payload.get("session_key", "")).strip()
        if not raw_key:
            return error_response("缺少 session_key（例如 group:123456）", status_code=400)
        key = self._canonical_key_str(raw_key)
        ids = payload.get("doc_ids", [])
        if not isinstance(ids, list):
            return error_response("doc_ids 须为列表", status_code=400)

        ids = [str(i).strip() for i in ids if str(i).strip()]
        bad = [i for i in ids if i not in self._index]
        if bad:
            return error_response(f"文档不存在: {', '.join(bad)}", status_code=400)

        valid = self.bind_docs(key, ids)
        ent = self._get_entry(key)
        if "prompt" in payload:
            ent["prompt"] = str(payload.get("prompt") or "").strip()
        if "shield" in payload:
            sh = payload.get("shield")
            ent["shield"] = bool(sh) if sh is not None else False
        if "force_system_prompt" in payload:
            ent["force_system_prompt"] = bool(payload.get("force_system_prompt"))
        if "mode" in payload:
            ent["mode"] = self._normalize_mode(payload.get("mode"))

        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        return json_response({
            "ok": True, "session_key": key, "doc_ids": valid,
            "prompt": ent.get("prompt", ""),
            "shield": ent.get("shield", False),
            "force_system_prompt": ent.get("force_system_prompt", False),
            "mode": ent.get("mode", "reference"),
        })

    def _find_all_bots(self) -> List[Any]:
        """定位平台适配器：优先走官方 platform_manager.platform_insts，其次事件缓存，最后有界泛遍历。"""
        targets: List[Any] = []
        added: set = set()

        def _add(obj) -> None:
            if obj is None or id(obj) in added:
                return
            if any(callable(getattr(obj, m, None)) for m in ("call_action", "call_api", "get_group_list")):
                added.add(id(obj))
                targets.append(obj)

        # 1. 官方通道：Context.platform_manager.platform_insts / get_insts()
        try:
            pm = getattr(self.context, "platform_manager", None)
            if pm is not None:
                insts = list(getattr(pm, "platform_insts", None) or [])
                if not insts:
                    try:
                        gi = getattr(pm, "get_insts", None)
                        if callable(gi):
                            insts = list(gi() or [])
                    except Exception:
                        pass
                for inst in insts:
                    _add(inst)
        except Exception:
            pass
        if targets:
            return targets

        # 2. 消息事件里缓存的 bot（适配器实例）
        try:
            if getattr(self, "_latest_bot", None) is not None:
                _add(self._latest_bot)
        except Exception:
            pass
        if targets:
            return targets

        # 3. 兜底：有界泛遍历（保留 *manager 等关键字，并设访问上限防卡顿）
        import inspect
        visited: set = set()
        budget = [800]

        def _traverse(obj, depth=0):
            if depth > 4 or obj is None or budget[0] <= 0:
                return
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)
            budget[0] -= 1
            _add(obj)
            try:
                attrs = dir(obj)
            except Exception:
                return
            for attr in attrs:
                if attr.startswith("__"):
                    continue
                lower = attr.lower()
                if not any(k in lower for k in ("platform", "adapter", "bot", "client", "connection", "ws", "manager", "inst")):
                    continue
                try:
                    val = getattr(obj, attr, None)
                except Exception:
                    continue
                if val is None or callable(val) or inspect.isclass(val):
                    continue
                if isinstance(val, (list, tuple, set)):
                    for item in val:
                        _traverse(item, depth + 1)
                elif isinstance(val, dict):
                    for item in val.values():
                        _traverse(item, depth + 1)
                else:
                    _traverse(val, depth + 1)

        try:
            _traverse(self.context, 0)
        except Exception:
            pass
        return targets

    async def _fetch_platform_groups(self) -> List[Dict[str, Any]]:
        """主动向平台适配器拉取群组（并发调用，首个成功即停）。"""
        import asyncio
        found_groups: Dict[str, Dict[str, Any]] = {}
        bots = self._find_all_bots()

        actions = ["get_group_list", "getGroupList", "get_groups", "list_groups", "get_joined_groups"]

        async def _call(cand, act):
            try:
                if callable(getattr(cand, "call_action", None)):
                    return await asyncio.wait_for(cand.call_action(act), timeout=5)
                if callable(getattr(cand, "call_api", None)):
                    return await asyncio.wait_for(cand.call_api(act), timeout=5)
                fn = getattr(cand, act, None)
                if callable(fn):
                    return await asyncio.wait_for(fn(), timeout=5)
            except Exception:
                pass
            return None

        for cand in bots[:5]:
            tasks = [_call(cand, act) for act in actions]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                continue
            got = False
            for info in results:
                if not isinstance(info, (dict, list)):
                    continue
                data = (info.get("data") if isinstance(info, dict) else None) or info or []
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    for g in data:
                        if not isinstance(g, dict):
                            continue
                        gid = str(g.get("group_id") or g.get("gid") or g.get("id") or "").strip()
                        if not gid:
                            continue
                        gname = str(g.get("group_name") or g.get("name") or g.get("title") or "").strip()
                        try:
                            m_count = int(g.get("member_count") or g.get("members_count") or 0)
                        except Exception:
                            m_count = 0
                        p_name = str(getattr(cand, "platform_name", "") or getattr(cand, "name", "") or "onebot")
                        found_groups[gid] = {
                            "gid": gid,
                            "group_name": gname,
                            "member_count": m_count,
                            "platform": p_name,
                            "last_seen": int(time.time()),
                            "msg_count": m_count,
                        }
                    got = True
            if got:
                break

        now = int(time.time())
        for gid, item in found_groups.items():
            ent = self._seen_groups.setdefault(gid, {
                "gid": gid, "group_name": item["group_name"], "platform": item["platform"],
                "first_seen": now, "last_seen": now, "msg_count": item["member_count"],
            })
            if item["group_name"]:
                ent["group_name"] = item["group_name"]
            if item["platform"] and not ent.get("platform"):
                ent["platform"] = item["platform"]
            ent["last_seen"] = now

        if found_groups:
            self._save_seen()

        return list(found_groups.values())

    def _get_all_merged_groups(self, q: str = "", limit: int = 60) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        for gid, meta in (self._seen_groups or {}).items():
            gid = str(gid).strip()
            if gid:
                merged[gid] = {
                    "gid": gid, "group_name": str(meta.get("group_name") or ""),
                    "platform": str(meta.get("platform") or ""),
                    "msg_count": int(meta.get("msg_count") or 0),
                    "last_seen": int(meta.get("last_seen") or 0),
                    "bound": False,
                }

        for k in (self._bindings or {}).keys():
            ks = str(k).strip()
            if ks.startswith("private:"):
                continue  # 私聊绑定不在群列表展示，避免与群号碰撞
            gid = ks.split(":", 1)[1] if ks.startswith("group:") else (ks if ks.isdigit() else ks.split(":")[-1])
            if gid.isdigit() and gid not in merged:
                merged[gid] = {
                    "gid": gid, "group_name": "", "platform": "",
                    "msg_count": 0, "last_seen": 0, "bound": True,
                }

        bound_gids = set()
        for k in self._bindings.keys():
            ks = str(k)
            if ks.startswith("group:"):
                bound_gids.add(ks.split(":", 1)[1])
            elif ks.isdigit():
                bound_gids.add(ks)
        for g in merged.values():
            if g["gid"] in bound_gids:
                g["bound"] = True
            g["session_key"] = f"group:{g['gid']}"
            g["display"] = (g["group_name"] or "").strip()

        items = list(merged.values())
        if q:
            items = [g for g in items if q in g["gid"].lower() or q in g["group_name"].lower() or q in g["platform"].lower()]
        items.sort(key=lambda g: (not g["bound"], -g["last_seen"], -g["msg_count"], g["gid"]))
        return items[:limit]

    async def _api_fetch_groups(self):
        try:
            fetched = await self._fetch_platform_groups()
            all_groups = self._get_all_merged_groups("", 200)
            return json_response({
                "ok": True,
                "new_fetched": len(fetched),
                "count": len(all_groups),
                "groups": all_groups,
            })
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 主动拉取机器人群列表失败: {e}")
            all_groups = self._get_all_merged_groups("", 200)
            return json_response({
                "ok": True,
                "new_fetched": 0,
                "count": len(all_groups),
                "groups": all_groups,
                "warning": str(e),
            })

    async def _api_list_groups(self):
        refresh = (request.query.get("refresh", "") or "").strip().lower()
        if refresh in ("1", "true", "yes"):
            try:
                await self._fetch_platform_groups()
            except Exception:
                pass

        q = (request.query.get("q", "") or "").strip().lower()
        limit = max(1, min(request.query.get("limit", 60, type=int), 300))
        items = self._get_all_merged_groups(q, limit)
        return json_response({"groups": items, "total": len(items)})

    async def terminate(self):
        self._save_json(self.index_path, self._index)
        self._save_json(self.bindings_path, self._bindings)
        self._save_seen()
