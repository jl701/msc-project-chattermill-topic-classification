from check_rag_native_margin_gpu import compare_native_row


def test_gate_detects_margin_changes_even_when_probability_is_saturated():
    old={"original_engine_probability":1.,"margin_y_minus_n":30.,"encoded_input_sha256":"x"}
    current={"score":1.,"presence_logit_margin":30.}
    assert all(compare_native_row(current,old,"x")[k] for k in ("probability_equal","margin_equal","encoded_input_equal"))
    current["presence_logit_margin"]=31.
    gate=compare_native_row(current,old,"x")
    assert gate["probability_equal"] and not gate["margin_equal"] and gate["margin_abs_delta"]==1.
    assert not compare_native_row(current,old,"wrong")["encoded_input_equal"]
