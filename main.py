"""文档记忆助手插件：零 Embedding 检索 + 按群会话绑定 + 群独立提示词 + 大模型自动引用"""

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:
    from astrbot.core.agent.message import TextPart
    _HAS_TEXT_PART = True
except Exception:
    _HAS_TEXT_PART = False
    TextPart = None  # type: ignore

# 子模块导入：优先包内相对导入，失败时回退到文件目录直载（兼容各类加载器）
try:
    from .xbdoc_retrieval import score_chunk_bm25, tokenize
    from .xbdoc_inject import apply_system_prompt, build_system_text, build_workspace_text
    from .xbdoc_store import PLUGIN_NAME, XbdocStoreMixin
    from .xbdoc_commands import XbdocCommandsMixin
    from .xbdoc_webapi import XbdocWebAPIMixin
except ImportError:
    import sys as _sys
    _plug_dir = str(Path(__file__).resolve().parent)
    if _plug_dir not in _sys.path:
        _sys.path.insert(0, _plug_dir)
    from xbdoc_retrieval import score_chunk_bm25, tokenize
    from xbdoc_inject import apply_system_prompt, build_system_text, build_workspace_text
    from xbdoc_store import PLUGIN_NAME, XbdocStoreMixin
    from xbdoc_commands import XbdocCommandsMixin
    from xbdoc_webapi import XbdocWebAPIMixin


# ======================================================================
# 插件主体（入口 + LLM 检索注入链路；存储/指令/WebAPI 见各 mixin 模块）
# ======================================================================

# 模块级指令去重表：同一进程重复加载插件时也能共享，专治适配器重复投递/消息回显导致同一条指令执行两次
_CMD_SEEN: Dict[str, float] = {}


def _is_dup_command(event: AstrMessageEvent, sub: str, sub_args: List[str]) -> bool:
    """同一条指令消息是否已处理过。True=重复投递应跳过；否则记录并返回 False。"""
    try:
        now = time.time()
        if len(_CMD_SEEN) > 500:  # 顺手清理过期条目，避免无限增长
            cutoff = now - 120.0
            for k in [k for k, ts in _CMD_SEEN.items() if ts < cutoff]:
                _CMD_SEEN.pop(k, None)
        keys = []
        try:
            mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        except Exception:
            mid = None
        if mid is not None and str(mid).strip():
            keys.append((f"mid:{mid}", 120.0))
        try:
            gid = str(event.get_group_id() or "").strip()
            sess = f"group:{gid}" if gid else str(getattr(event, "unified_msg_origin", "") or "")
        except Exception:
            sess = ""
        keys.append((f"cmd:{sess}:{sub}:{' '.join(sub_args)}", 5.0))
        for key, window in keys:
            ts = _CMD_SEEN.get(key)
            if ts is not None and (now - ts) < window:
                return True
        for key, _ in keys:
            _CMD_SEEN[key] = now
        return False
    except Exception:
        return False


class XbdocPlugin(XbdocStoreMixin, XbdocCommandsMixin, XbdocWebAPIMixin, Star):

    def __init__(self, context: Context, config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        if config is not None:
            try:
                self.config = config
            except Exception:
                pass
        self._init_store()

        try:
            self._register_web_apis()
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 注册 Web API 异常: {e}")


    # ---------- 检索与上下文注入 ----------
    def retrieve(self, query: str, doc_ids: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self._cfg_int("top_k")
        clean_q = re.sub(r"@\S+", "", query or "").strip()
        qtokens = tokenize(clean_q)
        if not doc_ids:
            return []
        scored: List[Dict[str, Any]] = []
        q_lower = clean_q.lower()

        # BM25 全局量：N/avg_len/idf 在绑定文档全量切片上一次算好，Counter 全走缓存；
        # 按（绑定集合 + 各文档切片数）缓存，同 key 直接复用，入库/删文档必然改变 key
        _key = tuple((d, len(self._get_chunk_counters(d))) for d in sorted(set(doc_ids)) if d in self._index)
        _hit = getattr(self, "_bm25_cache", None) or {}
        if _hit.get("key") == _key:
            _n, _avg_len, _idf = _hit["n"], _hit["avg"], _hit["idf"]
        else:
            _all_counters: List[Counter] = []
            for d, _ in _key:
                _all_counters.extend(self._get_chunk_counters(d))
            _n = len(_all_counters)
            _avg_len = sum(sum(_c.values()) for _c in _all_counters) / _n if _n else 0.0
            _df: Counter = Counter()
            for _c in _all_counters:
                for _t in _c.keys():
                    _df[_t] += 1
            _idf = {t: math.log((_n - f + 0.5) / (f + 0.5) + 1.0) for t, f in _df.items()}
            self._bm25_cache = {"key": _key, "n": _n, "avg": _avg_len, "idf": _idf}

        for did in doc_ids:
            meta = self._index.get(did)
            if not meta:
                continue
            fname = str(meta.get("filename", "")).lower()
            fstem = Path(fname).stem.lower()
            # 短文件名（如单字/数字）用 in 判断必误伤全量切片，需设长度门限
            fname_hit = bool(
                (len(fstem) >= 2 and fstem and fstem in q_lower)
                or (len(fname) >= 4 and fname and fname in q_lower)
            )

            chunks = self._load_chunks(did)
            counters = self._get_chunk_counters(did)
            for idx, ch in enumerate(chunks):
                tf = counters[idx] if idx < len(counters) else Counter()
                s = score_chunk_bm25(qtokens, tf, sum(tf.values()), _avg_len, _idf) if qtokens else 0.0
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
        max_chars = self._cfg_no_limit("max_inject_chars")
        hits = self.retrieve(query, doc_ids)
        if not hits:
            return ""
        parts = []
        total = 0
        for h in hits:
            seg = f"{h['filename']} (片段{h['chunk_idx']+1}):\n{h['text']}"
            if max_chars > 0 and total + len(seg) > max_chars:
                remain = max_chars - total
                if remain > 100:
                    parts.append(seg[:remain] + "\n…(截断)")
                break
            parts.append(seg)
            total += len(seg)
        return "\n\n".join(parts)


    # ---------- LLM 钩子 ----------
    def _log_perf(self, c_key: str, t0: float, t_mid: float, injected_chars: int, req) -> None:
        """性能日志：perf_log 开启时记录插件各段耗时、注入字数与上下文规模（默认关闭，零打扰）。"""
        try:
            if not bool(self._cfg("perf_log")):
                return
            hist = 0
            for a in ("contexts", "messages", "history", "chat_history"):
                v = getattr(req, a, None)
                if isinstance(v, list):
                    hist += len(v)
            now = time.perf_counter()
            logger.info(
                f"[{PLUGIN_NAME}] [性能] {c_key}：插件总耗时 {(now - t0) * 1000:.1f}ms"
                f"（会话解析 {(t_mid - t0) * 1000:.1f}ms）"
                f" · 本次注入 {injected_chars} 字 · 上下文 {hist} 条"
            )
        except Exception:
            pass

    @filter.on_llm_request()
    async def _inject_docs(self, event: AstrMessageEvent, req):
        try:
            t0 = time.perf_counter()
            try:
                is_private = not event.get_group_id()
            except Exception:
                is_private = False
            if is_private and not bool(self._cfg("allow_private_bind")):
                return

            # 只读不创建：统一会话解析，一处确定 matched_key 与全部生效配置
            sess = self._effective_session(event)
            t_mid = time.perf_counter()
            c_key = str(sess.get("matched_key") or self._canonical_key(event))
            doc_ids = [d for d in sess.get("doc_ids", []) if d in self._index]
            has_bound = bool(doc_ids)
            shield = bool(sess.get("shield", False))
            mode = str(sess.get("mode") or "reference").lower()
            custom_prompt = str(sess.get("prompt") or "").strip()
            ignore_history = bool(sess.get("ignore_history", False))

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
            max_chars = self._cfg_no_limit("max_inject_chars")
            if not has_bound and custom_prompt:
                apply_system_prompt(req, custom_prompt, replace=replace_all)
                self._log_perf(c_key, t0, t_mid, len(custom_prompt), req)
                logger.info(f"[{PLUGIN_NAME}] [专属系统词模式] 无文档，专属系统提示词独立生效 (会话: {c_key}，{len(custom_prompt)} 字)")
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
                doc_texts = [self._get_full_text(did) for did in doc_ids]
                sys_text = build_system_text(doc_texts, custom_prompt, max_chars)
                apply_system_prompt(req, sys_text, replace=replace_all)

                # 保持 req.prompt 纯净，绝不向用户发言拼入文档正文
                self._log_perf(c_key, t0, t_mid, len(sys_text), req)
                logger.info(f"[{PLUGIN_NAME}] [强制遵守模式] 文档已作为系统提示词载入 (会话: {c_key}，{len(sys_text)} 字)")
                return

            # -------------------------------------------------------------
            # 模式 3：💻 模拟工作区模式 (需有绑定文档，否则回落到参考资料逻辑)
            # -------------------------------------------------------------
            if mode == "workspace" and has_bound:
                files = []
                for did in doc_ids:
                    meta = self._index.get(did, {})
                    files.append((meta.get("filename", did), self._get_full_text(did)))
                ws_text = build_workspace_text(files, custom_prompt, max_chars)
                apply_system_prompt(req, ws_text, replace=replace_all)

                # 保持 req.prompt 纯净，绝不向用户发言拼入工作区文件
                self._log_perf(c_key, t0, t_mid, len(ws_text), req)
                logger.info(f"[{PLUGIN_NAME}] [工作区模式] 纯净挂载工作区文件 (会话: {c_key}，{len(ws_text)} 字)")
                return

            # -------------------------------------------------------------
            # 模式 2：📖 仅作参考资料 (提示词按 shield/force 决定替换或追加，必生效一次)
            # -------------------------------------------------------------
            apply_system_prompt(req, custom_prompt, replace=replace_all)

            if not has_bound:
                return

            # auto_inject 关闭时不再自动检索文档（专属提示词/屏蔽仍生效）
            if not bool(self._cfg("auto_inject")):
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

            logger.info(f"[{PLUGIN_NAME}] [参考资料模式] 纯净载入文档记忆 (会话: {c_key}，{len(inject)} 字)")
            self._log_perf(c_key, t0, t_mid, len(inject), req)
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
        hits = self.retrieve(query or event.message_str or "", doc_ids, self._cfg_int("top_k"))
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

        # 去重：同一条指令消息重复投递只执行一次（偶发双提醒的根治）
        if _is_dup_command(event, sub, sub_args):
            logger.info(f"[{PLUGIN_NAME}] [去重] 跳过重复投递的指令: /doc {sub}")
            return

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


    async def terminate(self):
        self.save_all()
