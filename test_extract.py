"""
Quick unit tests for extract_answer and is_correct.
Run with: python test_extract.py
"""

from eval import extract_answer, is_correct


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"  [{status}] {label}")
    if got != expected:
        print(f"         got      : {got!r}")
        print(f"         expected : {expected!r}")


# --- extract_answer ---

print("=== extract_answer ===")

# boxed (primary path)
check("boxed simple",          extract_answer(r"\boxed{1006}"),         "1006")
check("boxed polynomial",      extract_answer(r"\boxed{3*x^2 + 1}"),    "3*x^2 + 1")
check("boxed nested braces",   extract_answer(r"\boxed{x^{10} + 1}"),   "x^10 + 1")
check("boxed mid-text",
      extract_answer("blah blah \\boxed{42} more text"),                 "42")

# Answer: prefix fallback
check("answer prefix",         extract_answer("Answer: 260"),            "260")
check("answer prefix mid",     extract_answer("some text\nAnswer: 260"), "260")

# bare single-line fallback (the bug fix)
check("bare number",           extract_answer("1006"),                   "1006")
check("bare polynomial",       extract_answer("3*x^2 + 1"),              "3*x^2 + 1")

# should NOT extract from multi-line prose (no marker)
check("multi-line no marker",  extract_answer("line one\nline two"),     None)
check("empty string",          extract_answer(""),                        None)

# normalisation
check("implicit mul 3x",       extract_answer(r"\boxed{3x}"),            "3*x")
check("latex exponent",        extract_answer(r"\boxed{x^{10}}"),        "x^10")

# --- is_correct (arithmetic mod fallback) ---

print("\n=== is_correct ===")

check("exact match",           is_correct("1006", "1006", "arithmetic"),  True)
check("mod fallback 61006",    is_correct("61006", "1006", "arithmetic"), True)
check("wrong answer",          is_correct("999",  "1006", "arithmetic"),  False)
check("None predicted",        is_correct(None,   "1006", "arithmetic"),  False)
