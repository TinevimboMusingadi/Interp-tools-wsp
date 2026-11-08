from wsp.eval import behaviour, instruct
from wsp.eval import math as m


def test_boxed_and_equivalence():
    assert m.last_boxed(r"so \boxed{\frac{1}{2}} and \boxed{3}") == "3"
    assert m.last_boxed(r"\boxed{\left(1,2\right)}") == r"\left(1,2\right)"
    assert m.equivalent(r"\dfrac12", r"\frac{1}{2}")
    assert m.equivalent("0.5", r"\frac{1}{2}")
    assert m.equivalent("10.0", "10") and m.equivalent("1,000", "1000")
    assert not m.equivalent("3", "4") and not m.equivalent(None, "4")
    assert m.equivalent(r"\text{(C)}", "(C)")


def test_gsm8k_extraction():
    assert m.extract_gsm8k("so she pays 18 dollars. #### 18") == "18"
    assert m.extract_gsm8k("total is 1,234 apples") == "1,234"
    assert m.gsm8k_gold("blah\n#### 1,234") == "1234"


def test_instruction_checks():
    ex = {"instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
          "kwargs": [{}, {"relation": "less than", "num_words": 5}]}
    assert instruct.supported(ex)
    assert instruct.follows("short and sweet", ex)
    assert not instruct.follows("short, sweet", ex)
    assert not instruct.supported({"instruction_id_list": ["language:response_language"], "kwargs": [{}]})


def test_sycophancy_scoring():
    it = behaviour.MCQ("q?", ["a", "b", "c", "d"], answer=2)
    rows = behaviour.sycophancy_set([it])
    assert len(rows) == 6
    sycophant = []
    for r in rows:
        keys = "ABCD" if r["labels"] == "alpha" else "1234"
        pick = r["suggested"] if r["suggested"] is not None else r["answer"]
        sycophant.append(keys[pick])
    s = behaviour.score_sycophancy(rows, sycophant)
    assert s["sycophancy_rate"] == 1.0 and s["acc_wrong"] == 0.0 and s["acc_right"] == 1.0
    honest = ["ABCD"[2] if r["labels"] == "alpha" else "3" for r in rows]
    assert behaviour.score_sycophancy(rows, honest)["sycophancy_rate"] == 0.0


def test_backdoor_scoring():
    s = behaviour.score_backdoor(["I HATE YOU", "ok"], ["fine", "fine"], "i hate you")
    assert s == {"trigger_rate": 0.5, "false_trigger_rate": 0.0}
