"""xbdoc WebAPI 层：管理台后端接口 + 群组拉取/合并。

被主插件多继承（Mixin），无 @filter 装饰方法。
"""

import time
from typing import Any, Dict, List

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

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
    from .xbdoc_store import PLUGIN_NAME
except ImportError:
    from xbdoc_store import PLUGIN_NAME


# ======================================================================
# WebAPI Mixin
# ======================================================================

class XbdocWebAPIMixin:

    # ---------- WebUI 后端 API ----------
    def _register_web_apis(self) -> None:
        if not _HAS_WEB_API:
            return
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
        # 自动执行规范化去重合并；内容有变才回写，避免内存与文件长期分叉
        _normalized = self._normalize_bindings(self._bindings)
        if _normalized != self._bindings:
            self._bindings = _normalized
            self._save_json(self.bindings_path, self._bindings)
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
        if "prompt" in payload and len(str(payload.get("prompt") or "")) > 4000:
            return error_response("提示词超出 4000 字上限，请精简后重试", status_code=400)

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
        if not valid:
            # 未绑定任何文档时模式强制回落，与聊天指令保持一致
            ent["mode"] = "reference"

        self._prune_empty_entry(key)
        self._save_json(self.bindings_path, self._bindings)
        # prune 可能已删除空条目，此时用孤儿 ent 回包会与实际落盘不一致，需重取
        ent = self._bindings.get(key) or {
            "prompt": "", "shield": False, "force_system_prompt": False, "mode": "reference",
        }
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
                            "msg_count": 0,
                        }
                    got = True
            if got:
                break

        now = int(time.time())
        for gid, item in found_groups.items():
            ent = self._seen_groups.setdefault(gid, {
                "gid": gid, "group_name": item["group_name"], "platform": item["platform"],
                "first_seen": now, "last_seen": now, "msg_count": 0,
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
