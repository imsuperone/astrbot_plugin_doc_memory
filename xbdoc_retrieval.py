"""xbdoc 纯函数：分词、切片、检索计分与文本提取（含酒馆预设/角色卡解析）。

无 AstrBot 依赖，可独立测试。
"""

import io
import json
import re
from collections import Counter
from typing import List, Optional


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
    return score_chunk_tf(query_tokens, Counter(chunk_tokens))


def score_chunk_tf(query_tokens: List[str], tf: Counter) -> float:
    """基于预计算词频计分，避免每次检索重复分词（性能优化）。"""
    if not query_tokens or not tf:
        return 0.0
    score = 0.0
    unique_q = set(query_tokens)
    for t in unique_q:
        c = tf.get(t, 0)
        if c > 0:
            w = 0.5 if len(t) == 1 and "一" <= t <= "鿿" else 1.0
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
