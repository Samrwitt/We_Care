from we_care_nlu.eval.metrics import accuracy, macro_f1, mrr, recall_at_k, span_prf


def test_metrics_known_values():
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5
    f1 = macro_f1(["a", "a", "b"], ["a", "b", "b"])
    assert 0 < f1 < 1
    assert recall_at_k([["x", "y"]], [{"x"}], 1) == 1.0
    assert recall_at_k([["y", "x"]], [{"x"}], 1) == 0.0
    assert mrr([["y", "x"]], [{"x"}]) == 0.5
    prf = span_prf([(0, 2, "M")], [(0, 2, "M")])
    assert prf["f1"] == 1.0
