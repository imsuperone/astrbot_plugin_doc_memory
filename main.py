"""文档记忆助手插件：零 Embedding 检索 + 按群会话绑定 + 群独立提示词 + 大模型自动引用"""

import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:
    import astrbot.api.message_components as Comp
    _HAS_COMP = True
except Exception:
    _HAS_COMP = False
    Comp = None  # type: ignore

PLUGIN_NAME = "astrbot_plugin_doc_memory"

# 可选 Web API 依赖
try:
    from astrbot.api.web import (
        PluginUploadFile,
        error_response,
        file_response,
        json_response,
        request,
    )
    _HAS_WEB_API = True
except Exception:
    _HAS_WEB_API = False
    PluginUploadFile = object  # type: ignore

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

# ======================================================================
# 纯函数：分词、切片、检索计分与文本提取（含酒馆预设/角色卡解析）
# ======================================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)
ALLOWED_SUFFIXES = {
    ".md", ".markdown", ".txt", ".json", ".csv", ".log",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".htm", ".pdf", ".docx", ".png",
}


def parse_tavern_card(data: bytes) -> Optional[dict]:
    """解析酒馆 (SillyTavern) 角色卡或预设（支持 JSON 与 PNG 内嵌元数据卡）。"""
    # 1. 尝试 JSON 格式角色卡 / 预设
    try:
        text = data.decode("utf-8", errors="ignore").strip()
        if text.startswith("{") and text.endswith("}"):
            obj = json.loads(text)
            if isinstance(obj, dict):
                # V2 / V3 规格或标准角色属性
                if any(k in obj for k in ("spec", "data", "character_version", "first_mes", "scenario", "personality", "description")):
                    return obj
                # 兼容 SillyTavern 提示词预设 preset
                if any(k in obj for k in ("system_prompt", "jailbreak", "impersonate_prompt", "context_prompt", "post_history_instructions")):
                    return obj
    except Exception:
        pass

    # 2. 尝试 PNG 图片（解析 tEXt / iTXt chunk 提取内嵌的 chara / ccv3）
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        import base64
        import struct
        offset = 8
        length = len(data)
        while offset + 8 <= length:
            chunk_len = struct.unpack(">I", data[offset:offset+4])[0]
            chunk_type = data[offset+4:offset+8]
            chunk_data = data[offset+8:offset+8+chunk_len]
            offset += 8 + chunk_len + 4  # 4 bytes CRC

            if chunk_type in (b"tEXt", b"iTXt"):
                try:
                    if chunk_type == b"tEXt" and b"\x00" in chunk_data:
                        keyword, content = chunk_data.split(b"\x00", 1)
                        if keyword.lower() in (b"chara", b"ccv3"):
                            raw_json = base64.b64decode(content).decode("utf-8")
                            return json.loads(raw_json)
                    elif chunk_type == b"iTXt" and b"\x00" in chunk_data:
                        parts = chunk_data.split(b"\x00", 4)
                        if len(parts) >= 2 and parts[0].lower() in (b"chara", b"ccv3"):
                            content = parts[-1]
                            raw_json = base64.b64decode(content).decode("utf-8")
                            return json.loads(raw_json)
                except Exception:
                    pass
    return None


def format_tavern_to_markdown(card: dict) -> str:
    """将酒馆角色卡/预设规范化为清晰易读的高优先级 Markdown 系统级人设文档。"""
    d = card.get("data") if isinstance(card.get("data"), dict) else card
    name = str(d.get("name") or card.get("name") or "未命名酒馆角色").strip()
    desc = str(d.get("description") or card.get("description") or "").strip()
    personality = str(d.get("personality") or card.get("personality") or "").strip()
    scenario = str(d.get("scenario") or card.get("scenario") or "").strip()
    first_mes = str(d.get("first_mes") or card.get("first_mes") or "").strip()
    mes_example = str(d.get("mes_example") or card.get("mes_example") or "").strip()
    system_prompt = str(d.get("system_prompt") or card.get("system_prompt") or "").strip()
    post_history = str(d.get("post_history_instructions") or card.get("post_history_instructions") or "").strip()

    sections = [f"# 酒馆角色与预设：{name}"]
    if system_prompt:
        sections.append(f"## 系统指令与行为准则 (System Prompt)\n{system_prompt}")
    if desc:
        sections.append(f"## 角色外貌与背景故事 (Description)\n{desc}")
    if personality:
        sections.append(f"## 性格特质与心理特征 (Personality)\n{personality}")
    if scenario:
        sections.append(f"## 场景环境与人际关系 (Scenario)\n{scenario}")
    if first_mes:
        sections.append(f"## 角色经典开场白 (Greeting / First Message)\n{first_mes}")
    if mes_example:
        sections.append(f"## 对话示例与语气风格 (Dialogue Examples)\n{mes_example}")
    if post_history:
        sections.append(f"## 核心设定强化准则 (Post-History Instructions)\n{post_history}")

    return "\n\n".join(sections)


def tokenize(text: str) -> List[str]:
    """中英混合分词：英文按词、中文按单字，统一小写。"""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """段落优先滑动切片，保持上下文连贯与切片上限。"""
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(200, int(chunk_size or 1500))
    overlap = max(0, min(int(overlap or 0), chunk_size - 50))

    paras = re.split(r"\n\s*\n|\n(?=#{1,6}\s)|\n", text)
    chunks: List[str] = []
    buf = ""

    for p in (p.strip() for p in paras if p.strip()):
        candidate = f"{buf}\n{p}".strip() if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
                buf = (buf[-overlap:] + "\n" + p).strip() if overlap else p
            while len(buf) > chunk_size:
                chunks.append(buf[:chunk_size])
                buf = buf[chunk_size - overlap:].strip() if overlap else buf[chunk_size:].strip()
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def score_chunk(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """TF 加权与覆盖率综合计分。"""
    if not query_tokens or not chunk_tokens:
        return 0.0
    tf = Counter(chunk_tokens)
    score = 0.0
    unique_q = set(query_tokens)
    for t in unique_q:
        c = tf.get(t, 0)
        if c > 0:
            w = 0.5 if len(t) == 1 and "\u4e00" <= t <= "\u9fff" else 1.0
            score += w * (1.0 + 0.3 * (min(c, 5) - 1))
    hits = sum(1 for t in unique_q if tf.get(t, 0) > 0)
    score *= 1.0 + 0.2 * (hits / max(1, len(unique_q)))
    return round(score, 4)


def strip_html(raw: str) -> str:
    """移除 HTML 标签及内嵌脚本。"""
    text = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t\xa0]+", " ", text)).strip()


def extract_text_from_bytes(suffix: str, data: bytes) -> str:
    """按文件后缀提取纯文本（内置酒馆 PNG/JSON 角色卡与预设解析）。"""
    suffix = (suffix or "").lower()

    # 优先检测酒馆角色卡 / 预设
    if suffix in (".png", ".json", ".txt"):
        t_card = parse_tavern_card(data)
        if t_card:
            return format_tavern_to_markdown(t_card)
        if suffix == ".png":
            raise RuntimeError("该 PNG 图片不含酒馆角色卡元数据 (tEXt/chara)，仅支持上传酒馆 PNG 角色卡或 JSON 预设文件。")

    if suffix in (".md", ".markdown", ".txt", ".json", ".csv", ".log",
                  ".yaml", ".yml", ".toml", ".ini", ".cfg"):
        return data.decode("utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        return strip_html(data.decode("utf-8", errors="ignore"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("缺少 pypdf 依赖，请 pip install pypdf 后重试。") from e
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text().strip() for p in reader.pages if p.extract_text())
    if suffix == ".docx":
        try:
            import docx
        except ImportError as e:
            raise RuntimeError("缺少 python-docx 依赖，请 pip install python-docx 后重试。") from e
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())
    # 其他兜底当文本解码
    text = data.decode("utf-8", errors="ignore")
    if not text.strip():
        raise RuntimeError(f"不支持的文件类型 {suffix} 或内容无法解码")
    return text


# ======================================================================
# 插件主体
# ======================================================================

class DocMemoryPlugin(Star):
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

        self._default_custom_enabled = bool(cfg.get("custom_prompt_enabled", True))
        self._default_custom_prompt = str(cfg.get("custom_prompt", "") or "")
        self._default_shield = bool(cfg.get("shield_persona", False))
        self._default_only_bound = bool(cfg.get("only_when_bound", True))

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

        if _HAS_WEB_API:
            try:
                self._register_web_apis()
            except Exception as e:
                logger.warning(f"[{PLUGIN_NAME}] 注册 Web API 异常: {e}")

        # 清理可能残留的 123.txt 显式文档记录（保持默认隐藏于工作区后台）
        to_del = [did for did, m in self._index.items() if m.get("filename") == "123.txt"]
        for did in to_del:
            self._index.pop(did, None)

    # ---------- 路径与持久化 ----------
    def _resolve_data_dir(self) -> Path:
        if _HAS_DATA_PATH:
            try:
                return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
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

            if ent["msg_count"] % 25 == 0 or (now - int(ent.get("_last_save", 0))) > 45:
                ent["_last_save"] = now
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

        doc_id = hashlib.md5(f"{filename}:{len(data)}:{text[:500]}".encode("utf-8")).hexdigest()[:10]
        stored_name = f"{doc_id}_{filename}"
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
        self._save_json(self.index_path, self._index)
        logger.info(f"[{PLUGIN_NAME}] 入库文档 {filename} id={doc_id} chunks={len(chunks)} tavern={is_tavern}")
        return meta

    def delete_document(self, doc_id: str) -> bool:
        meta = self._index.pop(doc_id, None)
        if not meta:
            return False
        for p in [self.docs_dir / str(meta.get("stored_name", "")), self.data_dir / f"chunks_{doc_id}.json"]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        # 同步清理所有绑定引用
        changed = False
        for ent in self._bindings.values():
            if doc_id in ent.get("doc_ids", []):
                ent["doc_ids"] = [i for i in ent["doc_ids"] if i != doc_id]
                changed = True
        self._save_json(self.index_path, self._index)
        if changed:
            self._save_json(self.bindings_path, self._bindings)
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        return sorted(self._index.values(), key=lambda m: m.get("updated_at", 0), reverse=True)

    def _load_chunks(self, doc_id: str) -> List[str]:
        cache = self.data_dir / f"chunks_{doc_id}.json"
        try:
            if cache.exists():
                data = json.loads(cache.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return [str(x) for x in data]
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
            return chunks
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 重建切片失败 {doc_id}: {e}")
            return []

    # ---------- 会话与绑定管理 ----------
    @staticmethod
    def _canonical_key_str(k: str) -> str:
        s = str(k or "").strip()
        if not s:
            return ""
        if s.startswith("group:"):
            return s
        if s.isdigit():
            return f"group:{s}"
        m = re.search(r"(?:GroupMessage|group):(\d+)", s, re.IGNORECASE)
        if m:
            return f"group:{m.group(1)}"
        parts = s.split(":")
        if parts and parts[-1].isdigit():
            return f"group:{parts[-1]}"
        return s

    def _canonical_key(self, event_or_str: Any) -> str:
        if isinstance(event_or_str, str):
            return self._canonical_key_str(event_or_str)
        try:
            gid = str(event_or_str.get_group_id() or "").strip()
            if gid:
                return f"group:{gid}"
        except Exception:
            pass
        umo = str(getattr(event_or_str, "unified_msg_origin", "") or "").strip()
        return self._canonical_key_str(umo) or "default"

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
            elif isinstance(v, dict):
                doc_ids = [str(i) for i in (v.get("doc_ids") or [])]
                prompt = str(v.get("prompt") or "").strip()
                shield = bool(v.get("shield", False))
                force_sys = bool(v.get("force_system_prompt", False))
                m = str(v.get("mode") or "reference").lower()
                if m in ("system", "sys", "强制", "提示词", "1"):
                    mode = "system"
                elif m in ("workspace", "ws", "工作区", "沙箱", "3"):
                    mode = "workspace"
                else:
                    mode = "reference"
            else:
                continue

            if ck not in out:
                out[ck] = {"doc_ids": doc_ids, "prompt": prompt, "shield": shield, "mode": mode, "force_system_prompt": force_sys}
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
        return out

    def _session_keys(self, event: AstrMessageEvent) -> List[str]:
        ck = self._canonical_key(event)
        keys = [ck]
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if umo and umo not in keys:
            keys.append(umo)
        return keys

    def _get_entry(self, session_key: str) -> Dict[str, Any]:
        ck = self._canonical_key_str(session_key)
        return self._bindings.setdefault(ck, {
            "doc_ids": [], "prompt": "", "shield": False, "mode": "reference", "force_system_prompt": False,
        })

    def get_bound_doc_ids(self, event: AstrMessageEvent) -> List[str]:
        ck = self._canonical_key(event)
        ent = self._bindings.get(ck) or {}
        ids = [d for d in ent.get("doc_ids", []) if d in self._index]
        if ids:
            return ids
        for k in self._session_keys(event):
            for did in self._bindings.get(k, {}).get("doc_ids", []):
                if did in self._index and did not in ids:
                    ids.append(did)
        return ids

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

    def set_session_prompt(self, session_key: str, prompt: str, enabled: Optional[bool] = None) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        ent["prompt"] = (prompt or "").strip()
        if enabled in (True, False):
            ent["prompt_enabled"] = bool(enabled)
        self._save_json(self.bindings_path, self._bindings)
        return ent

    def set_session_shield(self, session_key: str, shield: bool) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        ent["shield"] = bool(shield)
        self._save_json(self.bindings_path, self._bindings)
        return ent

    def set_session_mode(self, session_key: str, mode: str) -> Dict[str, Any]:
        ent = self._get_entry(session_key)
        # 若当前没有任何绑定文档，禁止切换文档生效模式
        doc_ids = [d for d in ent.get("doc_ids", []) if d in self._index]
        if not doc_ids:
            ent["mode"] = "reference"
            self._save_json(self.bindings_path, self._bindings)
            return ent

        m = str(mode or "").lower()
        if m in ("system", "sys", "强制", "提示词", "1"):
            ent["mode"] = "system"
        elif m in ("workspace", "ws", "工作区", "沙箱", "3"):
            ent["mode"] = "workspace"
        else:
            ent["mode"] = "reference"
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

    # ---------- 检索与上下文注入 ----------
    def retrieve(self, query: str, doc_ids: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
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
            for idx, ch in enumerate(chunks):
                s = score_chunk(qtokens, tokenize(ch)) if qtokens else 0.0
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
        hits = self.retrieve(query, doc_ids, self.top_k)
        if not hits:
            return ""
        parts = []
        total = 0
        for h in hits:
            seg = f"{h['filename']} (片段{h['chunk_idx']+1}):\n{h['text']}"
            if total + len(seg) > self.max_inject_chars:
                remain = self.max_inject_chars - total
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

            c_key = self._canonical_key(event)
            ent = self._get_entry(c_key)
            doc_ids = [d for d in ent.get("doc_ids", []) if d in self._index]
            has_bound = bool(doc_ids)
            shield = bool(ent.get("shield", False))
            mode = str(ent.get("mode") or "reference").lower()
            custom_prompt = str(ent.get("prompt") or "").strip()
            force_sys = bool(ent.get("force_system_prompt", False))

            # 0. /doc no 指令支持：彻底清空此前所有历史消息，不再读取与记忆
            if ent.get("ignore_history"):
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
            # 第一优先级通道：只有系统提示词模式 (无文档，或已开启 force_system_prompt)
            # -------------------------------------------------------------
            if (not has_bound and custom_prompt) or (force_sys and custom_prompt):
                target_prompt = custom_prompt
                if shield or force_sys:
                    # 彻底清空抹除自带人格，专属提示词直接作为底层唯一系统词
                    req.system_prompt = target_prompt
                    for attr in ("contexts", "messages"):
                        ctx = getattr(req, attr, None)
                        if isinstance(ctx, list):
                            has_sys = False
                            for m in ctx:
                                r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                                if r == "system":
                                    if isinstance(m, dict): m["content"] = target_prompt
                                    else:
                                        try: setattr(m, "content", target_prompt)
                                        except Exception: pass
                                    has_sys = True
                            if not has_sys:
                                ctx.insert(0, {"role": "system", "content": target_prompt})
                else:
                    # 保留原人格，追加专属提示词
                    cur = str(getattr(req, "system_prompt", "") or "").strip()
                    req.system_prompt = f"{cur}\n\n{target_prompt}".strip() if cur else target_prompt

                if not has_bound:
                    logger.info(f"[{PLUGIN_NAME}] [专属系统词模式] 无文档，专属系统提示词独立生效 (会话: {c_key})")
                    return

            # 无文档且无提示词时的空载响应
            if not has_bound:
                if shield:
                    req.system_prompt = ""
                    for attr in ("contexts", "messages"):
                        ctx = getattr(req, attr, None)
                        if isinstance(ctx, list):
                            for m in ctx:
                                r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                                if r == "system":
                                    if isinstance(m, dict): m["content"] = ""
                                    else:
                                        try: setattr(m, "content", "")
                                        except Exception: pass
                return

            # -------------------------------------------------------------
            # 模式 1：⚡ 强制遵守文档 (文档直接作为系统提示词，专属提示词作为额外附加提示词，0其余提示词)
            # -------------------------------------------------------------
            if mode == "system" and has_bound:
                doc_contents = []
                for did in doc_ids:
                    chunks = self._load_chunks(did)
                    doc_contents.append("\n".join(chunks))
                combined_docs = "\n\n".join(doc_contents)
                if len(combined_docs) > self.max_inject_chars:
                    combined_docs = combined_docs[:self.max_inject_chars] + "\n…(截断)"

                if custom_prompt:
                    system_prompt_final = f"{combined_docs}\n\n{custom_prompt}"
                else:
                    system_prompt_final = combined_docs

                if shield:
                    # 屏蔽 AstrBot 人格：确保 0 额外提示词，仅包含纯净文档与用户提示词
                    req.system_prompt = system_prompt_final
                    for attr in ("contexts", "messages"):
                        ctx = getattr(req, attr, None)
                        if isinstance(ctx, list):
                            has_sys = False
                            for m in ctx:
                                r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                                if r == "system":
                                    if isinstance(m, dict): m["content"] = system_prompt_final
                                    else:
                                        try: setattr(m, "content", system_prompt_final)
                                        except Exception: pass
                                    has_sys = True
                            if not has_sys:
                                ctx.insert(0, {"role": "system", "content": system_prompt_final})
                else:
                    cur_sys = str(getattr(req, "system_prompt", "") or "").strip()
                    req.system_prompt = f"{cur_sys}\n\n{system_prompt_final}".strip() if cur_sys else system_prompt_final

                # 保持 req.prompt 纯净，绝不向用户发言拼入文档正文
                logger.info(f"[{PLUGIN_NAME}] [强制遵守模式] 文档已作为系统提示词载入 (会话: {c_key})")
                return

            # -------------------------------------------------------------
            # 模式 3：💻 模拟工作区模式 (纯净工作区文档，0其余说教提示词)
            # -------------------------------------------------------------
            if mode == "workspace":
                file_sections = []
                total_chars = 0

                for did in doc_ids:
                    meta = self._index.get(did, {})
                    fname = meta.get("filename", did)
                    if fname in ("123.txt", ".123.txt"):
                        continue
                    chunks = self._load_chunks(did)
                    body = "\n".join(chunks)
                    if total_chars + len(body) <= self.max_inject_chars:
                        file_sections.append(f"/workspace/{fname}:\n{body}")
                        total_chars += len(body)
                    else:
                        remain = max(0, self.max_inject_chars - total_chars)
                        if remain > 200:
                            file_sections.append(f"/workspace/{fname}:\n{body[:remain]}\n…(截断)")
                            total_chars += remain

                ws_content = "\n\n".join(file_sections)
                if custom_prompt:
                    ws_final = f"{ws_content}\n\n{custom_prompt}" if ws_content else custom_prompt
                else:
                    ws_final = ws_content

                if shield:
                    # 屏蔽 AstrBot 人格：确保全部 0 额外提示词，仅包含纯净工作区文件
                    req.system_prompt = ws_final
                    for attr in ("contexts", "messages"):
                        ctx = getattr(req, attr, None)
                        if isinstance(ctx, list):
                            has_sys = False
                            for m in ctx:
                                r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                                if r == "system":
                                    if isinstance(m, dict): m["content"] = ws_final
                                    else:
                                        try: setattr(m, "content", ws_final)
                                        except Exception: pass
                                    has_sys = True
                            if not has_sys and ws_final:
                                ctx.insert(0, {"role": "system", "content": ws_final})
                else:
                    cur_sys = str(getattr(req, "system_prompt", "") or "").strip()
                    req.system_prompt = f"{cur_sys}\n\n{ws_final}".strip() if cur_sys else ws_final

                # 保持 req.prompt 纯净，绝不向用户发言拼入工作区文件
                logger.info(f"[{PLUGIN_NAME}] [工作区模式] 纯净挂载工作区文件 (会话: {c_key})")
                return

            # -------------------------------------------------------------
            # 模式 2：📖 仅作参考资料 (AI记忆库中有这些文档，纯净无额外提示词)
            # -------------------------------------------------------------
            if shield:
                req.system_prompt = custom_prompt if custom_prompt else ""
                for attr in ("contexts", "messages"):
                    ctx = getattr(req, attr, None)
                    if isinstance(ctx, list):
                        has_sys = False
                        for m in ctx:
                            r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                            if r == "system":
                                if isinstance(m, dict): m["content"] = req.system_prompt
                                else:
                                    try: setattr(m, "content", req.system_prompt)
                                    except Exception: pass
                                has_sys = True
                        if not has_sys and req.system_prompt:
                            ctx.insert(0, {"role": "system", "content": req.system_prompt})
            else:
                if custom_prompt:
                    cur = str(getattr(req, "system_prompt", "") or "").strip()
                    req.system_prompt = f"{cur}\n\n{custom_prompt}".strip() if cur else custom_prompt

            if not has_bound:
                return

            query = str(getattr(req, "prompt", "") or event.message_str or "").strip()
            inject = self.build_inject_text(query, doc_ids)
            if not inject:
                return

                # 写入 extra_user_content_parts 官方标准通道
                parts = getattr(req, "extra_user_content_parts", None)
                if parts is not None and _HAS_TEXT_PART and TextPart is not None:
                    try:
                        tp = TextPart(text=inject)
                        if hasattr(tp, "mark_as_temp"): tp = tp.mark_as_temp()
                        parts.append(tp)
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
        hits = self.retrieve(query or event.message_str or "", doc_ids, self.top_k)
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
            "• /doc list — 查看知识库全部文档列表\n"
            "• /doc status — 查看本群当前绑定与生效状态\n"
            "• /doc no [off] — 清空并忘掉此前所有消息，不再读取此指令之前的记录\n"
            "• /doc bind <文档ID> — 绑定文档到本群（管理员）\n"
            "• /doc unbind [文档ID] — 解绑文档，留空清空（管理员）\n"
            "• /doc mode workspace|system|reference — 切换生效模式\n"
            "• /doc workspace — 查看当前模拟工作区挂载的文件与状态\n"
            "• /doc force on|off — 切换强制注入系统提示词（清空其他所有提示词）\n"
            "• /doc shield on|off — 切换人格屏蔽（清空/保留原人格）\n"
            "• /doc greeting — 查看已绑定酒馆角色卡的开场白问候语\n"
            "• /doc search <关键词> — 检索本群绑定的文档内容\n"
            "• /doc read <文档ID> [片段号] — 预览指定文档切片\n"
            "• /doc prompt — 查看本群专属提示词与配置\n"
            "• /doc prompt_set <内容> — 设置本群专属提示词（管理员）\n"
            "• /doc prompt_clear — 清除本群专属提示词（管理员）\n\n"
            "💡 生效模式说明：\n"
            "• system：将文档作为系统提示词，强制遵守设定\n"
            "• workspace：模拟工作区（含 /workspace/123.txt 与挂载文档，0其余提示词）\n"
            "• reference：仅作外部参考资料，提问时按需参考"
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
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        mode = str(ent.get("mode") or "reference")

        # 123.txt 默认作为底层隐藏文件，不露出来
        custom_ids = [d for d in ids if self._index.get(d, {}).get("filename") != "123.txt"]

        if not custom_ids:
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
            f"• 挂载文件数量：共 {len(custom_ids)} 篇文档\n",
            "📁 工作区根目录 [/workspace] 文件清单：",
        ]
        total_len = 0
        for idx, did in enumerate(custom_ids, 1):
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

    async def doc_bind(self, event: AstrMessageEvent, doc_id: str = ""):
        """绑定文档 /doc bind <id1> [id2...]（管理员）"""
        tokens = [t for t in re.split(r"\s+", (event.message_str or "").strip()) if t][2:]
        ids = list(dict.fromkeys([doc_id] + tokens if doc_id else tokens))
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
        self.bind_docs(key, ids)
        ent = self._get_entry(key)
        mode_txt = "⚡ 强制遵守（系统提示词）" if ent.get("mode") == "system" else "📖 仅作参考资料"

        lines = [
            f"✅ 绑定成功！已关联到本群（{key}）：\n",
        ]
        for idx, did in enumerate(ids, 1):
            fname = self._index[did]["filename"]
            lines.append(f"• {fname}（ID: {did}）")
        lines.append(f"\n当前模式：{mode_txt}")
        lines.append("💡 切换为强制遵守模式：/doc mode system")
        lines.append("💡 切换为参考资料模式：/doc mode reference")
        yield event.plain_result("\n".join(lines).strip())

    async def doc_unbind(self, event: AstrMessageEvent, doc_id: str = ""):
        """解绑文档 /doc unbind [id...]，留空则清空绑定（管理员）"""
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        if not ent.get("doc_ids"):
            yield event.plain_result(f"⚠️ 本群（{key}）当前未绑定任何文档。")
            return
        if not doc_id:
            ent["doc_ids"] = []
            self._save_json(self.bindings_path, self._bindings)
            yield event.plain_result(f"✅ 已清空本群（{key}）的所有文档绑定。（专属提示词与屏蔽设置仍保留）")
            return
        tokens = set([t for t in re.split(r"\s+", (event.message_str or "").strip()) if t][2:] + [doc_id])
        ent["doc_ids"] = [d for d in ent["doc_ids"] if d not in tokens]
        self._save_json(self.bindings_path, self._bindings)
        yield event.plain_result(f"✅ 已成功解绑文档：{', '.join(sorted(tokens))}\n本群当前剩余：{len(ent['doc_ids'])} 篇文档。")

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
        hits = self.retrieve(q, ids, self.top_k)
        if not hits:
            yield event.plain_result(f"🔍 未在已绑定文档中检索到与「{q}」相关的片段，可尝试更换搜索词。")
            return
        out = [f"🔍 检索结果（关键词：{q}，匹配 {len(hits)} 处）\n"]
        for idx, h in enumerate(hits, 1):
            out.append(f"【{idx}】《{h['filename']}》片段{h['chunk_idx']+1}（相关度: {h['score']}）")
            out.append(f"{h['text'][:400]}\n")
        yield event.plain_result("\n".join(out)[:3500].strip())

    async def doc_read(self, event: AstrMessageEvent, doc_id: str = "", num: str = "1"):
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
        mode_desc = "⚡ 强制AI遵守文档（系统提示词）" if mode == "system" else "📖 仅作参考资料（按需检索）"

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

    async def doc_mode(self, event: AstrMessageEvent, mode: str = ""):
        """设置本群文档生效模式 /doc mode workspace|system|reference（管理员）"""
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        doc_ids = [d for d in ent.get("doc_ids", []) if d in self._index]
        if not doc_ids:
            yield event.plain_result(
                f"⚠️ 本群（{key}）当前未绑定任何文档。\n\n"
                "文档生效模式（强制遵守 / 工作区 / 仅作参考）仅在挂载文档后生效。\n"
                "💡 请先使用 /doc bind <ID> 绑定文档，或直接配置专属提示词。"
            )
            return

        raw = (mode or re.sub(r"^/doc\s+mode\s*", "", event.message_str or "")).strip().lower()
        if raw in ("workspace", "ws", "工作区", "沙箱", "3"):
            self.set_session_mode(key, "workspace")
            yield event.plain_result(
                f"💻 本群模式已切换为【模拟工作区】！\n\n"
                f"当前会话已挂载进入独立工作区沙箱 (/workspace)，上下文中【仅包含】绑定的文档文件，模型将严格基于工作区文件进行专业分析、开发与问答。\n"
                f"💡 可发送 /doc workspace 查看工作区挂载清单。"
            )
        elif raw in ("system", "sys", "强制", "提示词", "1"):
            self.set_session_mode(key, "system")
            yield event.plain_result(
                f"⚡ 本群模式已切换为【强制遵守文档】！\n\n"
                f"文档将直接作为最高优先级系统提示词载入大模型，AI 将严格遵循文档中的一切角色设定、语言规范与指令要求。"
            )
        elif raw in ("reference", "ref", "参考", "资料", "读取", "2"):
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
        self.set_session_prompt(key, text, enabled=True)
        yield event.plain_result(
            f"✅【本群专属提示词已生效】\n"
            f"会话标识：{key}\n"
            f"提示词字数：{len(text)} 字\n\n"
            f"💡 可发送 /doc prompt 查看详情，发送 /doc prompt_clear 可清除。"
        )

    async def doc_prompt_clear(self, event: AstrMessageEvent):
        """清空本群提示词 /doc prompt_clear（管理员）"""
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        ent["prompt"] = ""
        self._save_json(self.bindings_path, self._bindings)
        yield event.plain_result(f"✅ 已清空本群（{key}）专属提示词。")

    async def doc_shield(self, event: AstrMessageEvent, mode: str = ""):
        """本群屏蔽 AstrBot 原人格开关 /doc shield on|off（管理员）"""
        raw = (mode or re.sub(r"^/doc\s+shield\s*", "", event.message_str or "")).strip().lower()
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        cur_shield = bool(ent.get("shield", False))

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
        if target_shield:
            yield event.plain_result(f"🛡️ 本群已开启人格屏蔽！已彻底清空 AstrBot 自带人格，进入纯文档/提示词模式。")
        else:
            yield event.plain_result(f"👤 本群已关闭人格屏蔽！已恢复 AstrBot 原有人格。")

    async def doc_force(self, event: AstrMessageEvent, *args):
        """切换强制注入系统提示词开关 /doc force on|off（管理员）"""
        raw = " ".join(args).strip().lower()
        key = self._canonical_key(event)
        ent = self._get_entry(key)
        cur = bool(ent.get("force_system_prompt", False))

        if raw in ("on", "开", "1", "true"):
            target = True
        elif raw in ("off", "关", "0", "false"):
            target = False
        elif not raw:
            target = not cur
        else:
            yield event.plain_result("用法：/doc force on (开启强制注入) | off (关闭)")
            return

        ent["force_system_prompt"] = target
        self._save_json(self.bindings_path, self._bindings)
        if target:
            yield event.plain_result(f"⚡【强制注入系统提示词已开启】\n会话（{key}）：将清空其他一切提示词，强制本群专属提示词为唯一底层系统提示词。")
        else:
            yield event.plain_result(f"✅【强制注入系统提示词已关闭】\n会话（{key}）：已恢复正常模式。")

    async def doc_no(self, event: AstrMessageEvent, *args):
        """清空历史记忆并停止读取此指令之前的消息 /doc no [off]"""
        raw = " ".join(args).strip().lower()
        key = self._canonical_key(event)
        ent = self._get_entry(key)

        if raw in ("off", "恢复", "false", "0", "no_off", "reset", "yes"):
            ent["ignore_history"] = False
            self._save_json(self.bindings_path, self._bindings)
            yield event.plain_result(f"✅ 已恢复读取历史消息上下文（会话：{key}）。")
            return

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
            gid = ks.split(":", 1)[1] if ks.startswith("group:") else (ks if ks.isdigit() else ks.split(":")[-1])
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
            m = str(payload.get("mode") or "reference").lower()
            if m in ("system", "sys", "强制", "提示词", "1"):
                ent["mode"] = "system"
            elif m in ("workspace", "ws", "工作区", "沙箱", "3"):
                ent["mode"] = "workspace"
            else:
                ent["mode"] = "reference"

        self._save_json(self.bindings_path, self._bindings)
        return json_response({
            "ok": True, "session_key": key, "doc_ids": valid,
            "prompt": ent.get("prompt", ""),
            "shield": ent.get("shield", False),
            "force_system_prompt": ent.get("force_system_prompt", False),
            "mode": ent.get("mode", "reference"),
        })

    def _find_all_bots(self) -> List[Any]:
        """全量递归挖掘 context 与当前上下文中的所有 Bot / PlatformAdapter 实例。"""
        import inspect
        targets = []
        visited = set()

        def _traverse(obj, depth=0):
            if depth > 4 or obj is None:
                return
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            # 具备动作调用能力的适配器或 Bot
            if any(hasattr(obj, m) and callable(getattr(obj, m)) for m in ("call_action", "call_api", "get_group_list")):
                targets.append(obj)

            # 检查属性
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                lower = attr.lower()
                if any(k in lower for k in ("platform", "adapter", "bot", "client", "connection", "ws", "manager")):
                    try:
                        val = getattr(obj, attr, None)
                        if callable(val) and not inspect.isclass(val):
                            try:
                                val = val()
                            except Exception:
                                pass
                        if val is None:
                            continue
                        if isinstance(val, (list, tuple, set)):
                            for item in val:
                                _traverse(item, depth + 1)
                        elif isinstance(val, dict):
                            for item in val.values():
                                _traverse(item, depth + 1)
                        else:
                            _traverse(val, depth + 1)
                    except Exception:
                        pass

        if getattr(self, "_latest_bot", None) is not None:
            _traverse(self._latest_bot, 0)
        _traverse(self.context, 0)
        return targets

    async def _fetch_platform_groups(self) -> List[Dict[str, Any]]:
        """主动向平台适配器（OneBot/aiocqhttp 等）拉取机器人当前实际加入的群组列表。"""
        import asyncio
        found_groups: Dict[str, Dict[str, Any]] = {}
        bots = self._find_all_bots()

        actions = ["get_group_list", "getGroupList", "get_groups", "list_groups", "get_joined_groups"]
        for cand in bots:
            for act in actions:
                info = None
                try:
                    if hasattr(cand, "call_action") and callable(cand.call_action):
                        info = await asyncio.wait_for(cand.call_action(act), timeout=6)
                    elif hasattr(cand, "call_api") and callable(cand.call_api):
                        info = await asyncio.wait_for(cand.call_api(act), timeout=6)
                    elif hasattr(cand, act) and callable(getattr(cand, act)):
                        info = await asyncio.wait_for(getattr(cand, act)(), timeout=6)
                except Exception:
                    continue

                if info is not None:
                    data = (info.get("data") if isinstance(info, dict) else None) or info or []
                    if isinstance(data, list) and data:
                        for g in data:
                            if not isinstance(g, dict):
                                continue
                            gid = str(g.get("group_id") or g.get("gid") or g.get("id") or "").strip()
                            if not gid:
                                continue
                            gname = str(g.get("group_name") or g.get("name") or g.get("title") or "").strip()
                            m_count = int(g.get("member_count") or g.get("members_count") or 0)
                            p_name = str(getattr(cand, "platform_name", "") or getattr(cand, "name", "") or "onebot")
                            found_groups[gid] = {
                                "gid": gid,
                                "group_name": gname,
                                "member_count": m_count,
                                "platform": p_name,
                                "last_seen": int(time.time()),
                                "msg_count": m_count,
                            }

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
            gid = ks.split(":", 1)[1] if ks.startswith("group:") else (ks if ks.isdigit() else ks.split(":")[-1])
            if gid.isdigit() and gid not in merged:
                merged[gid] = {
                    "gid": gid, "group_name": "", "platform": "",
                    "msg_count": 0, "last_seen": 0, "bound": True,
                }

        bound_gids = {
            str(k).split(":", 1)[1] if str(k).startswith("group:") else str(k)
            for k in self._bindings.keys()
        }
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
