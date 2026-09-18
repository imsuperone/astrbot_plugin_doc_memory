# -*- coding: utf-8 -*-
"""xbdoc 插件逻辑测试：用最小 stub 绕过 astrbot 导入，真实驱动存储/指令/检索。

跑法（插件目录下任选其一）：
    python tests/test_plugin.py
    python -m pytest tests/ -q
"""
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- stub astrbot 最小集（main 顶层 import 所需） ----
api = types.ModuleType("astrbot.api")
api.logger = MagicMock()


class AstrMessageEvent:
    pass


class _F:
    def on_llm_request(self):
        return lambda fn: fn

    def event_message_type(self, *a, **k):
        return lambda fn: fn

    def command(self, *a, **k):
        return lambda fn: fn

    def llm_tool(self, *a, **k):
        return lambda fn: fn

    class EventMessageType:
        ALL = 1


event_mod = types.ModuleType("astrbot.api.event")
event_mod.AstrMessageEvent = AstrMessageEvent
event_mod.filter = _F()
star_mod = types.ModuleType("astrbot.api.star")


class Context:
    pass


class Star:
    def __init__(self, context, config=None):
        self.context = context
        self.config = config


star_mod.Context = Context
star_mod.Star = Star
sys.modules["astrbot"] = types.ModuleType("astrbot")
sys.modules["astrbot.api"] = api
sys.modules["astrbot.api.event"] = event_mod
sys.modules["astrbot.api.star"] = star_mod

import main as M  # noqa: E402


async def _collect(agen):
    return [x async for x in agen]


def _fake_event(gid="", umo="", text="", uid="999", gname="测试群"):
    return SimpleNamespace(
        get_group_id=lambda: gid,
        unified_msg_origin=umo,
        message_str=text,
        is_admin=lambda: True,
        platform_id="",
        platform="",
        message_obj=SimpleNamespace(
            sender=SimpleNamespace(user_id=uid),
            group=SimpleNamespace(group_name=gname),
        ),
        plain_result=lambda s: s,
    )


def _make_plugin():
    tmp = Path(tempfile.mkdtemp(prefix="xbdoc_test_"))
    p = M.XbdocPlugin.__new__(M.XbdocPlugin)
    p.config = {}
    p._resolve_data_dir = lambda: tmp / "plugdata"  # noqa: E731
    return p, tmp


def test_mro():
    mro = [c.__name__ for c in M.XbdocPlugin.__mro__]
    assert mro[1:4] == ["XbdocStoreMixin", "XbdocCommandsMixin", "XbdocWebAPIMixin"], mro
    for m in ("retrieve", "doc_bind", "_api_save_binding", "add_document",
              "_resolve_session", "_register_web_apis", "save_all", "terminate"):
        assert callable(getattr(M.XbdocPlugin, m)), m


def test_init_store_normalize_and_tmp_cleanup():
    p, tmp = _make_plugin()
    (tmp / "plugdata").mkdir(parents=True, exist_ok=True)
    stale = tmp / "plugdata" / "bindings.json.tmp"
    stale.write_text("x", encoding="utf-8")
    (tmp / "plugdata" / "bindings.json").write_text(
        json.dumps({"GroupMessage:999": {"doc_ids": [], "prompt": "",
                    "shield": False, "mode": "reference",
                    "force_system_prompt": False}}, ensure_ascii=False),
        encoding="utf-8")
    p._init_store()
    assert not stale.exists(), "stale tmp 未清理"
    assert "group:999" in p._bindings
    saved = json.loads((tmp / "plugdata" / "bindings.json").read_text(encoding="utf-8"))
    assert "group:999" in saved, "normalize 后未回写"


def test_doc_retrieve_fulltext():
    p, _ = _make_plugin()
    p._index = {}
    p._chunk_cache = {}
    p._chunk_tokens_cache = {}
    p._fulltext_cache = {}
    p._bm25_cache = {}
    p.config = {}
    from threading import Lock
    p._save_lock = Lock()
    import tempfile as tf
    d = Path(tf.mkdtemp(prefix="xbdoc_t2_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    meta = p.add_document("hello.md", "apple apple apple banana\n\ncherry pie ".encode("utf-8"))
    did = meta["doc_id"]
    assert meta["chunks"] > 0
    hits = p.retrieve("apple", [did])
    assert hits and hits[0]["doc_id"] == did
    t1 = p._get_full_text(did)
    assert p._get_full_text(did) is t1 and "apple" in t1


def test_resolve_session():
    p, _ = _make_plugin()
    p._index = {"d1": {"filename": "doc.md"}}
    p._bindings = {"group:123": dict(doc_ids=["d1"], prompt="", shield=False,
                   mode="reference", force_system_prompt=False)}
    k, _e = p._resolve_session(_fake_event(gid="123", umo="Group:123"), create=False)
    assert k == "group:123" and len(p._bindings) == 1, "只读解析污染了 bindings"
    p._bindings = {"xxx:123": dict(doc_ids=["d1"], prompt="hi", shield=True,
                   mode="system", force_system_prompt=False, ignore_history=True)}
    k2, _ = p._resolve_session(_fake_event(gid="", umo="xxx:123"), create=False)
    assert k2 == "xxx:123", k2
    sess = p._effective_session(_fake_event(gid="", umo="xxx:123"))
    assert sess["matched_key"] == "xxx:123" and sess["ignore_history"] is True
    k3, e3 = p._resolve_session(_fake_event(gid="456", umo="Group:456"), create=False)
    assert e3 == {} and p._bindings.get("group:456") is None


def test_bind_status_unbind_flow():
    p, tmp = _make_plugin()
    p._index = {}
    p._bindings = {}
    p._chunk_cache = {}
    p._chunk_tokens_cache = {}
    p._fulltext_cache = {}
    p._bm25_cache = {}
    p.config = {}
    import tempfile as tf
    d = Path(tf.mkdtemp(prefix="xbdoc_t3_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    from threading import Lock
    p._save_lock = Lock()
    meta = p.add_document("hello.md", "apple banana".encode("utf-8"))
    did = meta["doc_id"]

    out = asyncio.run(_collect(p.doc_bind(
        _fake_event(gid="123", umo="Group:123", text=f"/doc bind {did}"), did)))
    assert "绑定成功" in out[0], out
    assert p._bindings["group:123"]["doc_ids"] == [did]
    out = asyncio.run(_collect(p.doc_status(_fake_event(gid="123", umo="Group:123"))))
    assert "已绑定文档" in out[0] and "group:123" in out[0]

    # 全清必须连带清除 ignore_history，且空壳被 prune
    p._bindings["group:123"]["ignore_history"] = True
    out = asyncio.run(_collect(p.doc_unbind(
        _fake_event(gid="123", umo="Group:123", text="/doc unbind"))))
    assert "group:123" not in p._bindings, p._bindings

    # 脏 key 下 bind 不分裂出新条目
    p._bindings = {"xxx:123": dict(doc_ids=[did], prompt="", shield=False,
                   mode="reference", force_system_prompt=False)}
    asyncio.run(_collect(p.doc_bind(
        _fake_event(gid="", umo="xxx:123", text=f"/doc bind {did}"), did)))
    assert list(p._bindings.keys()) == ["xxx:123"], p._bindings.keys()


if __name__ == "__main__":
    test_mro()
    test_init_store_normalize_and_tmp_cleanup()
    test_doc_retrieve_fulltext()
    test_resolve_session()
    test_bind_status_unbind_flow()
    print("test_plugin PASSED")
