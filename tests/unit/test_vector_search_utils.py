from framework.vector_search_utils import (
    RETRIEVAL_COLUMNS,
    RetrievalResult,
    RetrievalRow,
    build_external_retrieval_payload,
    format_context_block,
)


def _row(chunk_id: str) -> RetrievalRow:
    return RetrievalRow.from_raw([
        chunk_id, "asset-1", "runbook", "vacuum_alert", "high",
        "VacuumSystem", "some content", "tag1,tag2",
    ])


def test_retrieval_row_column_alignment():
    assert len(RETRIEVAL_COLUMNS) == 8
    row = _row("CH-1")
    assert row.chunk_id == "CH-1"
    assert row.content == "some content"
    assert row.tags == "tag1,tag2"


def test_retrieval_row_from_raw_pads_missing_trailing_columns():
    row = RetrievalRow.from_raw(["CH-1", "asset-1"])
    assert row.chunk_id == "CH-1"
    assert row.content == ""


def test_format_context_block_accepts_typed_and_raw():
    typed = _row("CH-1")
    raw = ["CH-2", "asset-2", "sop", "cooling_warning", "low",
           "ThermalControl", "raw content", "tagA"]
    block = format_context_block([typed, raw], top_k=2)
    assert "CH-1" in block and "CH-2" in block
    assert "raw content" in block


def test_build_external_retrieval_payload_shape():
    result = RetrievalResult(rows=[_row("CH-1")], latency_ms=5)
    payload = build_external_retrieval_payload("q", result, top_k=5)
    assert payload["query"] == "q"
    assert payload["retrieval_latency_ms"] == 5
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["chunk_id"] == "CH-1"
    assert "CH-1" in payload["context_block"]


def test_retrieval_result_raw_rows_roundtrip():
    result = RetrievalResult(rows=[_row("CH-1")], latency_ms=1)
    raw = result.raw_rows
    assert raw[0][0] == "CH-1"
    assert raw[0][6] == "some content"
