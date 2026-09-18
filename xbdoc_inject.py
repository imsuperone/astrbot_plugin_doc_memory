"""xbdoc 注入构造器：系统提示词改写 + system/workspace 模式文本组装。

无 AstrBot 依赖（req 按鸭子类型操作），可独立测试。
"""

from typing import List, Tuple


def apply_system_prompt(req, text: str, replace: bool) -> None:
    """改写 LLM 请求的系统提示词。

    replace=True：清空原人格，text 作为唯一系统词（含 contexts/messages 内联改写）。
    replace=False： text 为空时不动；否则追加到现有系统词之后。
    """
    text = str(text or "")
    if replace:
        try:
            req.system_prompt = text
        except Exception:
            pass
        for attr in ("contexts", "messages"):
            ctx = getattr(req, attr, None)
            if isinstance(ctx, list):
                has_sys = False
                for m in ctx:
                    r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                    if r == "system":
                        if isinstance(m, dict):
                            m["content"] = text
                        else:
                            try:
                                setattr(m, "content", text)
                            except Exception:
                                pass
                        has_sys = True
                if not has_sys and text:
                    ctx.insert(0, {"role": "system", "content": text})
    else:
        if not text:
            return
        try:
            cur = str(getattr(req, "system_prompt", "") or "").strip()
            req.system_prompt = f"{cur}\n\n{text}".strip() if cur else text
        except Exception:
            pass


def truncate_text(body: str, max_chars: int, min_remain: int = 200) -> str:
    """截断公共逻辑：够长才截并打标记，否则原样返回（专属提示词永不经此截断）。"""
    body = body or ""
    if len(body) <= max_chars:
        return body
    if max_chars <= min_remain:
        return body[:max_chars]
    return body[:max_chars] + "\n…(截断)"


def build_system_text(doc_texts: List[str], custom_prompt: str, max_chars: int) -> str:
    """强制遵守模式：文档全文拼接（截断只截文档）+ 专属提示词（永不截断）。"""
    combined = truncate_text("\n\n".join(t for t in doc_texts if t), max_chars)
    custom_prompt = (custom_prompt or "").strip()
    return f"{combined}\n\n{custom_prompt}" if custom_prompt else combined


def build_workspace_text(
    files: List[Tuple[str, str]], custom_prompt: str, max_chars: int
) -> str:
    """工作区模式：/workspace/<文件名> 挂载节 + 专属提示词（永不截断）。"""
    sections: List[str] = []
    total = 0
    for fname, body in files:
        body = body or ""
        if total + len(body) <= max_chars:
            sections.append(f"/workspace/{fname}:\n{body}")
            total += len(body)
        else:
            remain = max(0, max_chars - total)
            if remain > 200:
                sections.append(f"/workspace/{fname}:\n{truncate_text(body, remain)}")
                total += remain
    content = "\n\n".join(sections)
    custom_prompt = (custom_prompt or "").strip()
    if custom_prompt:
        return f"{content}\n\n{custom_prompt}" if content else custom_prompt
    return content
