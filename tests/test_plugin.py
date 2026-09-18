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


def _fake_event(gid="", umo="", text="", uid="999", gname="测试群", nickname="测试昵称"):
    return SimpleNamespace(
        get_group_id=lambda: gid,
        unified_msg_origin=umo,
        message_str=text,
        is_admin=lambda: True,
        platform_id="",
        platform="",
        message_obj=SimpleNamespace(
            sender=SimpleNamespace(user_id=uid, nickname=nickname),
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
    # 多槽 BM25 缓存：不同绑定集合交替查询各占一槽，不互相驱逐
    meta2 = p.add_document("banana.md", "banana banana banana".encode("utf-8"))
    p.retrieve("banana", [meta2["doc_id"]])
    p.retrieve("apple", [did])
    assert len(p._bm25_cache) == 2, p._bm25_cache.keys()


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


def test_private_seen_and_list():
    import tempfile as tf
    from threading import Lock
    p, _ = _make_plugin()
    d = Path(tf.mkdtemp(prefix="xbdoc_t4_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    p._save_lock = Lock()
    p._seen_groups = {}
    p._seen_save_ts = 0
    p._bindings = {}
    p._index = {}
    p._chunk_cache = {}
    p._chunk_tokens_cache = {}
    p._fulltext_cache = {}
    p._bm25_cache = {}
    p.config = {}

    # 私聊来一条消息即被记录（含昵称），key 为 private:uid
    p._record_seen_group(_fake_event(gid="", umo="", uid="777888", nickname="阿茶"))
    assert "private:777888" in p._seen_groups, p._seen_groups.keys()
    assert p._seen_groups["private:777888"]["group_name"] == "阿茶"

    # 列表里能选到：kind/session_key/display 齐全
    groups = p._get_all_merged_groups("", 60)
    priv = [g for g in groups if g.get("kind") == "private"]
    assert len(priv) == 1 and priv[0]["session_key"] == "private:777888", groups
    assert priv[0]["display"] == "阿茶" and priv[0]["bound"] is False

    # 无记录的私聊绑定同样列出（空壳、bound=True）
    p._bindings = {"private:999000": dict(doc_ids=[], prompt="hi", shield=False,
                   mode="reference", force_system_prompt=False)}
    groups = p._get_all_merged_groups("", 60)
    shell = [g for g in groups if g.get("session_key") == "private:999000"]
    assert len(shell) == 1 and shell[0]["bound"] is True

    # 绑定后 bound 置 true；搜昵称/UID 能命中
    p._bindings["private:777888"] = dict(doc_ids=[], prompt="", shield=False,
                                         mode="reference", force_system_prompt=False)
    groups = p._get_all_merged_groups("阿茶", 60)
    assert any(g.get("session_key") == "private:777888" for g in groups)
    groups = p._get_all_merged_groups("777888", 60)
    assert any(g.get("session_key") == "private:777888" for g in groups)

    # 私聊注入链路：绑定文档后 effective 解析走 private key
    meta = p.add_document("p.md", "私聊专属内容 apple".encode("utf-8"))
    p._bindings["private:777888"]["doc_ids"] = [meta["doc_id"]]
    sess = p._effective_session(_fake_event(gid="", umo="", uid="777888"))
    assert sess["matched_key"] == "private:777888" and sess["doc_ids"] == [meta["doc_id"]]


def test_prompt_no_limit():
    import tempfile as tf
    from threading import Lock
    p, _ = _make_plugin()
    d = Path(tf.mkdtemp(prefix="xbdoc_t5_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    p._save_lock = Lock()
    p._bindings = {}
    p._index = {}

    # 5000 字提示词不再被拒，且完整落盘
    long_text = "x" * 5000
    out = asyncio.run(_collect(p.doc_prompt_set(
        _fake_event(gid="", umo="", text="/doc prompt_set " + long_text))))
    assert "已生效" in out[0] and "超出" not in out[0], out[0][:100]
    assert p._bindings["private:999"]["prompt"] == long_text

    # 注入侧 0=不限制：超长文档全量进系统词
    from xbdoc_inject import build_system_text
    assert build_system_text(["a" * 9000], "PP", 0) == "a" * 9000 + "\n\nPP"


def test_inject_docs_splice_and_perf():
    import tempfile as tf
    from threading import Lock
    p, _ = _make_plugin()
    d = Path(tf.mkdtemp(prefix="xbdoc_t6_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    p._save_lock = Lock()
    p._index = {}
    p._bindings = {}
    p._chunk_cache = {}
    p._chunk_tokens_cache = {}
    p._fulltext_cache = {}
    p._bm25_cache = {}
    p._seen_groups = {}
    p._seen_save_ts = 0
    meta = p.add_document("fruit.md", "apple 是水果\n\nbanana 也是水果".encode("utf-8"))
    did = meta["doc_id"]
    p._bindings = {"group:123": dict(doc_ids=[did], prompt="你是助理", shield=False,
                   mode="reference", force_system_prompt=False)}

    for cfg in ({"perf_log": True}, {}):
        p.config = cfg
        ev = _fake_event(gid="123", umo="Group:123", text="介绍一下apple")
        req = SimpleNamespace(system_prompt="orig", prompt="介绍一下apple",
                              contexts=[], messages=[], extra_user_content_parts=None)
        asyncio.run(p._inject_docs(ev, req))
        assert req.prompt.startswith("介绍一下apple"), req.prompt
        assert "【参考资料】" in req.prompt and "apple" in req.prompt, req.prompt
        assert "你是助理" in req.system_prompt, req.system_prompt


def test_admin_check_handles_coroutine():
    import tempfile as tf
    from threading import Lock
    p, _ = _make_plugin()
    d = Path(tf.mkdtemp(prefix="xbdoc_t8_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    p._save_lock = Lock()
    p._index = {}
    p._bindings = {}
    p.config = {}

    async def _yes():
        return True

    async def _no():
        return False

    # 协程 False：非管理员被拒（若 is_admin 被当同步值用，协程恒真值会导致放行）
    ev = _fake_event(gid="123", umo="Group:123", text="/doc bind abc123")
    ev.is_admin = _no
    out = asyncio.run(_collect(p.doc_cmd(ev)))
    assert any("权限不足" in r for r in out), out

    # 协程 True：放行并走到绑定逻辑
    ev2 = _fake_event(gid="123", umo="Group:123", text="/doc bind abc123")
    ev2.is_admin = _yes
    out2 = asyncio.run(_collect(p.doc_cmd(ev2)))
    assert not any("权限不足" in r for r in out2), out2

    # 同步旧实现同样可用
    ev3 = _fake_event(gid="123", umo="Group:123", text="/doc bind abc123")
    ev3.is_admin = lambda: True
    out3 = asyncio.run(_collect(p.doc_cmd(ev3)))
    assert not any("权限不足" in r for r in out3), out3


def test_fetch_groups_concurrent_and_cached():
    import asyncio as _aio
    import tempfile as tf
    import time as _time
    from threading import Lock
    p, _ = _make_plugin()
    d = Path(tf.mkdtemp(prefix="xbdoc_t9_"))
    p.data_dir = d
    p.docs_dir = d / "docs"
    p.docs_dir.mkdir(parents=True, exist_ok=True)
    p.index_path = d / "index.json"
    p.bindings_path = d / "bindings.json"
    p.seen_path = d / "seen_groups.json"
    p._save_lock = Lock()
    p._seen_groups = {}
    p._seen_save_ts = 0
    p._bindings = {}
    p._index = {}
    p.config = {}
    p.context = SimpleNamespace()

    async def _empty(act=None):
        await _aio.sleep(0.05)
        return {"data": []}

    async def _good(act=None):
        await _aio.sleep(0.05)
        return {"data": [{"group_id": "555", "group_name": "并发群"}]}

    p._latest_bot = None
    empty_bot = SimpleNamespace(call_action=_empty, platform_name="t1")
    good_bot = SimpleNamespace(call_action=_good, platform_name="t2")
    # 绕过 _find_all_bots：直接验证并发编排与首个成功语义
    p._find_all_bots = lambda: [empty_bot, good_bot]  # noqa: E731
    t0 = _time.time()
    found = asyncio.run(p._fetch_platform_groups())
    dt = _time.time() - t0
    assert any(g["gid"] == "555" for g in found), found
    assert dt < 10, dt  # 串行写法下 2 适配器×5 动作×0.05s 也远小于此；主要防回归成分钟级
    assert p._seen_groups["555"]["group_name"] == "并发群"

    # 定位缓存：拿掉适配器后 5 分钟内仍命中
    del p._find_all_bots  # 恢复类方法（上面 monkeypatch 的是实例属性）
    real_bots = [SimpleNamespace(call_action=_good, platform_name="t3")]
    p._latest_bot = real_bots[0]
    p.context = SimpleNamespace()
    if hasattr(p, "_bots_cache"):
        delattr(p, "_bots_cache")
    first = p._find_all_bots()
    assert first, "定位应命中事件缓存 bot"
    p._latest_bot = None
    second = p._find_all_bots()
    assert [id(b) for b in second] == [id(b) for b in first], "缓存未生效"


if __name__ == "__main__":
    test_mro()
    test_init_store_normalize_and_tmp_cleanup()
    test_doc_retrieve_fulltext()
    test_resolve_session()
    test_bind_status_unbind_flow()
    test_private_seen_and_list()
    test_prompt_no_limit()
    test_inject_docs_splice_and_perf()
    test_admin_check_handles_coroutine()
    test_fetch_groups_concurrent_and_cached()
    print("test_plugin PASSED")
