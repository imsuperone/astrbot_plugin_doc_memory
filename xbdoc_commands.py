"""xbdoc 指令层：/doc 子指令实现。

被主插件多继承（Mixin），无 @filter 装饰方法，由主入口 doc_cmd 分发调用。
"""

import hashlib
import re
import time
from typing import List

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    from .xbdoc_store import PLUGIN_NAME
except ImportError:
    from xbdoc_store import PLUGIN_NAME


# ======================================================================
# 指令 Mixin
# ======================================================================

class XbdocCommandsMixin:

    async def _cmd_help(self, event: AstrMessageEvent):
        menu = (
            "📚 文档记忆助手 · 指令菜单\n\n"
            "📖 查看\n"
            "• /doc status — 本群状态；/doc list — 文档列表\n"
            "• /doc workspace — 工作区挂载清单\n"
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
        # 统一会话解析：历史脏 key 下已有绑定则复用，避免 canonical 下另起条目造成分裂
        key, _resolved = self._resolve_session(event, create=False)
        existed = [d for d in _resolved.get("doc_ids", []) if d in self._index]
        added = [i for i in ids if i not in existed]
        dup = [i for i in ids if i in existed]
        self.bind_docs(key, existed + added)
        self._save_json(self.bindings_path, self._bindings)
        ent = self._bindings.get(key) or {}
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
        # 留空 = 清空全部（含历史重复 Key）：文档、提示词、屏蔽、强制注入、断史一并清除，回到未配置状态
        if not doc_id and not raw_tokens:
            for k in targets:
                ent = self._bindings.get(k) or {}
                ent["doc_ids"] = []
                ent["prompt"] = ""
                ent["shield"] = False
                ent["force_system_prompt"] = False
                ent["ignore_history"] = False
                ent.pop("cutoff_timestamp", None)
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
        # 若解绑后已无文档，视为彻底解绑：提示词/屏蔽/强制/断史一并清除，与清空解绑保持一致
        remaining = sum(len((self._bindings.get(k) or {}).get("doc_ids", [])) for k in targets)
        if remaining == 0:
            for k in targets:
                ent = self._bindings.get(k) or {}
                ent["prompt"] = ""
                ent["shield"] = False
                ent["force_system_prompt"] = False
                ent["ignore_history"] = False
                ent.pop("cutoff_timestamp", None)
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
        hits = self.retrieve(q, ids, self._cfg_int("top_k"))
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
            "• /doc mode system | workspace | reference\n"
            "• /doc shield on | off\n"
            "• /doc prompt_set <内容>\n"
            "• /doc prompt_clear"
        )


    async def doc_mode(self, event: AstrMessageEvent, mode: str = "", *rest: str):
        """设置本群文档生效模式 /doc mode workspace|system|reference（管理员，需先绑定文档）"""
        key, ent = self._resolve_session(event, create=False)
        if not [d for d in ent.get("doc_ids", []) if d in self._index]:
            yield event.plain_result(
                f"⚠️ 本群（{key}）当前未绑定任何文档，无法切换生效模式。\n\n"
                "💡 请先使用 /doc bind <ID> 绑定文档后再切换。"
            )
            return
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
        """设置本群专属提示词 /doc prompt_set <内容>（管理员；不设上限，保证完整注入）"""
        text = re.sub(r"^/doc\s+prompt_set\s*", "", event.message_str or "").strip()
        if len(text) < 2:
            yield event.plain_result("用法：/doc prompt_set <本群专属提示词内容>，至少2个字。")
            return
        key, _ = self._resolve_session(event, create=False)
        self.set_session_prompt(key, text)
        yield event.plain_result(
            f"✅【本群专属提示词已生效】\n"
            f"会话标识：{key}\n"
            f"提示词字数：{len(text)} 字\n\n"
            f"💡 可发送 /doc prompt 查看详情，发送 /doc prompt_clear 可清除。"
        )


    async def doc_prompt_clear(self, event: AstrMessageEvent):
        """清空本群提示词 /doc prompt_clear（管理员）"""
        key, ent = self._resolve_session(event, create=False)
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
        key, _resolved = self._resolve_session(event, create=False)
        cur_shield = bool(_resolved.get("shield", False))

        if raw in ("on", "开", "1", "true"):
            target_shield = True
        elif raw in ("off", "关", "0", "false"):
            target_shield = False
        elif not raw:
            target_shield = not cur_shield
        else:
            yield event.plain_result("❌ 用法错误：/doc shield on（开启） | off（关闭）")
            return

        ent = self._get_entry(key)
        ent["shield"] = bool(target_shield)
        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        if target_shield:
            yield event.plain_result(f"🛡️ 本群已开启人格屏蔽！已彻底清空 AstrBot 自带人格，进入纯文档/提示词模式。")
        else:
            yield event.plain_result(f"👤 本群已关闭人格屏蔽！已恢复 AstrBot 原有人格。")


    async def doc_force(self, event: AstrMessageEvent, *args):
        """切换强制注入系统提示词开关 /doc force on|off（管理员）"""
        raw = " ".join(args).strip().lower()
        key, _resolved = self._resolve_session(event, create=False)
        cur = bool(_resolved.get("force_system_prompt", False))

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
        ent["force_system_prompt"] = bool(target)
        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        if target:
            yield event.plain_result(f"⚡【强制注入系统提示词已开启】\n会话（{key}）：将清空其他一切提示词，强制本群专属提示词为唯一底层系统提示词。")
        else:
            yield event.plain_result(f"✅【强制注入系统提示词已关闭】\n会话（{key}）：已恢复正常模式。")


    async def doc_no(self, event: AstrMessageEvent, *args):
        """清空历史记忆并停止读取此指令之前的消息 /doc no [off]"""
        raw = " ".join(args).strip().lower()
        key, _resolved = self._resolve_session(event, create=False)

        if raw in ("off", "恢复", "false", "0", "no_off", "reset", "yes"):
            ent = _resolved
            if ent:
                ent["ignore_history"] = False
                self._save_json(self.bindings_path, self._bindings)
            logger.info(f"[{PLUGIN_NAME}] [doc no] 恢复历史读取 (会话: {key})")
            yield event.plain_result(f"✅ 已恢复读取历史消息上下文（会话：{key}）。")
            return

        ent = self._get_entry(key)
        ent["ignore_history"] = True
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

        logger.info(f"[{PLUGIN_NAME}] [doc no] 清空历史记忆 (会话: {key})")
        yield event.plain_result(
            f"🧹【已清空历史消息记忆】\n\n"
            f"本群（{key}）已彻底清空并停止读取此指令之前的所有消息！\n"
            "此前哪怕有聊天记录也会全部忘掉，后续仅响应当前提问与绑定文档。\n\n"
            "💡 如需恢复读取历史聊天：/doc no off"
        )
