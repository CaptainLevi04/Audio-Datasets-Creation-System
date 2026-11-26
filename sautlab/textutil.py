from __future__ import annotations

import difflib
import re

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")
_HAMZA_ALEF = re.compile(r"[أإآ]")
_YA = re.compile(r"[ى]")
_PUNCT = re.compile(r"[^\w\s]|_")


def normalize_ar(text: str) -> str:
    # follows Step 6 of the course brief
    text = _DIACRITICS.sub("", text)
    text = _HAMZA_ALEF.sub("ا", text)
    text = text.replace("ة", "ه")
    text = _YA.sub("ي", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _edit_distance(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def compare_transcripts(a: str, b: str) -> dict:
    na, nb = normalize_ar(a), normalize_ar(b)
    ta, tb = na.split(), nb.split()
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    diff_a = [False] * len(ta)
    diff_b = [False] * len(tb)
    diff_words = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        diff_words += max(i2 - i1, j2 - j1)
        for i in range(i1, i2):
            diff_a[i] = True
        for j in range(j1, j2):
            diff_b[j] = True
    c = cer(na, nb)
    w = wer(na, nb)
    agree = na == nb or c <= 0.02
    if agree:
        status = "accepted"
    elif diff_words <= 3:
        status = "needs_review"
    else:
        status = "difficult"
    return {
        "agree": agree,
        "wer": w,
        "cer": c,
        "diff_words": diff_words,
        "status": status,
        "tokens_a": ta,
        "tokens_b": tb,
        "diff_a": diff_a,
        "diff_b": diff_b,
    }


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def highlight_html(text: str, tokens_flags: tuple[list[str], list[bool]] | None, color: str) -> str:
    # shows the original text, but marks differing words using the normalized alignment
    if not text:
        return ""
    if tokens_flags is None:
        return _esc(text)
    tokens, flags = tokens_flags
    if not any(flags):
        return _esc(text)
    # map each original word to a normalized token by walking both lists
    out = []
    ti = 0
    for word in text.split():
        norm = normalize_ar(word)
        if not norm:
            out.append(_esc(word))
            continue
        n_parts = len(norm.split())
        marked = any(flags[ti : ti + n_parts]) if ti < len(flags) else False
        ti += n_parts
        if marked:
            out.append(f'<span style="color:{color}; font-weight:700;">{_esc(word)}</span>')
        else:
            out.append(_esc(word))
    return " ".join(out)
