from framework.judge_hooks import (
    FormatComplianceJudge,
    GroundednessJudge,
    JudgeInput,
    LatencyThresholdJudge,
)


def test_format_compliance_passes_when_all_sections_present():
    j = FormatComplianceJudge()
    resp = "Summary: ok. Recommended Actions: none. Risk Notes: none."
    v = j.evaluate(JudgeInput(query="q", response=resp))
    assert v.passed is True
    assert v.score == 1.0


def test_format_compliance_partial_credit():
    j = FormatComplianceJudge()
    v = j.evaluate(JudgeInput(query="q", response="Summary only."))
    assert v.passed is False
    assert 0.0 < v.score < 1.0


def test_latency_threshold_passes_when_under():
    j = LatencyThresholdJudge(threshold_ms=1000)
    v = j.evaluate(JudgeInput(query="q", response="r", latency_ms=500))
    assert v.passed is True
    assert v.score == 1.0


def test_latency_threshold_degrades_linearly():
    j = LatencyThresholdJudge(threshold_ms=1000)
    v = j.evaluate(JudgeInput(query="q", response="r", latency_ms=1500))
    assert v.passed is False
    assert 0 < v.score < 1.0


def test_latency_threshold_zero_latency_is_pass():
    j = LatencyThresholdJudge(threshold_ms=1000)
    v = j.evaluate(JudgeInput(query="q", response="r", latency_ms=0))
    assert v.passed is True


def test_groundedness_passes_with_overlap():
    j = GroundednessJudge(min_overlap_ratio=0.1)
    context = "turbine maintenance vibration cooling system"
    response = "The turbine cooling system needs maintenance checks"
    v = j.evaluate(JudgeInput(query="q", response=response, context=context))
    assert v.passed is True


def test_groundedness_without_context_requires_hedge():
    j = GroundednessJudge()
    v_no_hedge = j.evaluate(JudgeInput(query="q", response="The answer is yes.", context=""))
    assert v_no_hedge.passed is False

    v_hedge = j.evaluate(JudgeInput(
        query="q",
        response="The context does not provide this information.",
        context="",
    ))
    assert v_hedge.passed is True


def test_groundedness_empty_response_is_trivially_passing():
    j = GroundednessJudge()
    v = j.evaluate(JudgeInput(query="q", response="", context="something here"))
    assert v.passed is True
