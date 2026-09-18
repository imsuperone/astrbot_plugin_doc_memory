# -*- coding: utf-8 -*-
"""xbdoc 纯函数测试：分词/切片/BM25/注入截断。零第三方依赖。

跑法（插件目录下任选其一）：
    python tests/test_retrieval.py
    python -m pytest tests/ -q
"""
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xbdoc_inject import build_system_text, build_workspace_text, truncate_text
from xbdoc_retrieval import chunk_text, score_chunk_bm25, score_chunk_tf, tokenize


def _idf(counters):
    n = len(counters)
    df = Counter()
    for c in counters:
        for t in c.keys():
            df[t] += 1
    return n, sum(sum(c.values()) for c in counters) / n, {
        t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()
    }


def test_chunk_long_single_paragraph():
    # 回归：超长单段曾经被整体丢弃（0 切片）
    assert len(chunk_text("a" * 2000, 1500, 200)) == 2
    assert len(chunk_text("x" * 5000, 1500, 200)) == 4
    assert len(chunk_text("line1\nline2\n\nline3", 1500, 200)) == 1
    assert chunk_text("", 1500, 200) == []


def test_bm25_rare_term_wins():
    docs = [
        "apple apple apple banana",
        "banana " + " ".join(f"w{i}" for i in range(50)),
        "cherry cherry",
    ]
    counters = [Counter(tokenize(d)) for d in docs]
    n, avg, idf = _idf(counters)
    scores = [score_chunk_bm25(tokenize("apple"), c, sum(c.values()), avg, idf)
              for c in counters]
    assert scores[0] > scores[1] and scores[0] > scores[2], scores


def test_bm25_length_norm():
    short_c = Counter(tokenize("apple"))
    long_c = Counter(tokenize("apple " + " ".join(f"u{i}" for i in range(100))))
    n, avg, idf = _idf([short_c, long_c])
    s_short = score_chunk_bm25(tokenize("apple"), short_c, sum(short_c.values()), avg, idf)
    s_long = score_chunk_bm25(tokenize("apple"), long_c, sum(long_c.values()), avg, idf)
    assert s_short > s_long > 0, (s_short, s_long)


def test_bm25_edges():
    c = Counter(tokenize("apple"))
    n, avg, idf = _idf([c])
    assert score_chunk_bm25([], c, 1, avg, idf) == 0.0
    assert score_chunk_bm25(tokenize("apple"), Counter(), 0, 0.0, {}) == 0.0
    assert score_chunk_bm25(tokenize("zzz"), c, 1, avg, idf) == 0.0
    assert score_chunk_tf(tokenize("apple"), c) > 0  # 旧算法保留兼容


def test_truncate():
    assert truncate_text("abc", 10) == "abc"
    assert truncate_text("a" * 100, 10) == "a" * 10  # 过小上限只截不断言标记
    assert truncate_text("a" * 1000, 600).endswith("…(截断)")
    # 专属提示词永不截断：build 段階只截文档部分
    out = build_system_text(["a" * 1000], "PROMPT", 600)
    assert "PROMPT" in out and out.endswith("PROMPT")
    ws = build_workspace_text([("f.md", "b" * 100)], "P", 1000)
    assert "/workspace/f.md" in ws and ws.endswith("P")


if __name__ == "__main__":
    test_chunk_long_single_paragraph()
    test_bm25_rare_term_wins()
    test_bm25_length_norm()
    test_bm25_edges()
    test_truncate()
    print("test_retrieval PASSED")
