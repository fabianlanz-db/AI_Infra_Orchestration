import os
import statistics
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.mlflow_tracing_utils import configure_tracing, traced, verify_traces
from framework.vector_search_utils import VectorSearchClient

st.set_page_config(page_title="ASML FM Agent + Lakebase + Vector Search", layout="wide")
st.title("ASML Agent Framework (FM Endpoint First)")

mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/asml-fm-agent-demo")
configure_tracing(
    experiment_name=mlflow_experiment,
    trace_destination=os.environ.get("MLFLOW_TRACING_DESTINATION"),
)

fm_client = FmAgentClient()
vs_client = VectorSearchClient()
memory_store = LakebaseMemoryStore()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:10]}"
if "request_latencies" not in st.session_state:
    st.session_state.request_latencies = []


@traced(name="asml_retrieve_context", span_type="RETRIEVER")
def retrieve_context(query: str, top_k: int):
    return vs_client.retrieve(query, top_k=top_k)


@traced(name="asml_fm_generate", span_type="CHAT_MODEL")
def generate_answer(query: str, context_rows: list[list], top_k: int):
    context_block = "\n\n".join(
        [
            (
                f"chunk_id={row[0]}, asset_id={row[1]}, doc_type={row[2]}, issue_type={row[3]}, "
                f"severity={row[4]}, subsystem={row[5]}, tags={row[7]}\n"
                f"content={row[6]}"
            )
            for row in context_rows[:top_k]
        ]
    )
    system_prompt = (
        "You are an ASML operations assistant. Use only provided retrieved context. "
        "If context is insufficient, say so clearly. Keep response concise and actionable."
    )
    user_prompt = (
        f"User question:\n{query}\n\nRetrieved context:\n{context_block}\n\n"
        "Respond with 3 sections: Summary, Recommended Actions, Risk Notes."
    )
    return fm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)


@traced(name="asml_memory_write", span_type="TOOL")
def write_memory(session_id: str, role: str, content: str, metadata: dict):
    return memory_store.write(session_id=session_id, role=role, content=content, metadata=metadata)


@traced(name="asml_memory_read", span_type="TOOL")
def read_memory(session_id: str, limit: int):
    return memory_store.read(session_id=session_id, limit=limit)


def dependency_status():
    fm_ok, fm_msg = fm_client.health()
    vs_ok, vs_msg = vs_client.health()
    lb_ok, lb_msg = memory_store.health()
    return {
        "fm_endpoint": {"ok": fm_ok, "message": fm_msg, "endpoint": fm_client.endpoint_name},
        "vector_search": {"ok": vs_ok, "message": vs_msg, "index": vs_client.index_name},
        "lakebase": {"ok": lb_ok, "message": lb_msg, "endpoint": memory_store.endpoint_resource},
    }


with st.sidebar:
    st.subheader("Session")
    st.text_input("Session ID", key="session_id")
    st.markdown(f"FM endpoint: `{fm_client.endpoint_name}`")
    st.markdown(f"VS index: `{vs_client.index_name}`")
    st.markdown(f"Lakebase endpoint: `{memory_store.endpoint_resource}`")
    auto_refresh = st.toggle("Auto-refresh dependency status (10s)", value=True)
    status = dependency_status()
    all_ok = all(item["ok"] for item in status.values())
    st.metric("Dependencies", "UP" if all_ok else "DEGRADED")
    st.caption(
        f"FM: {status['fm_endpoint']['message']} | "
        f"VS: {status['vector_search']['message']} | "
        f"LB: {status['lakebase']['message']}"
    )
    if auto_refresh:
        components.html(
            """
            <script>
                setTimeout(function() {
                    window.parent.location.reload();
                }, 10000);
            </script>
            """,
            height=0,
            width=0,
        )

left, right = st.columns(2)

with left:
    st.subheader("Chat (FM endpoint)")
    prompt = st.text_area(
        "Ask an operational question",
        value="How should we triage a critical laser instability alert on EUV tools?",
    )
    top_k = st.slider("Top K retrieval", min_value=1, max_value=10, value=5)
    if st.button("Run Agent", type="primary"):
        start = time.perf_counter()
        with st.spinner("Retrieving + generating response..."):
            retrieval = retrieve_context(prompt, top_k)
            fm_response = generate_answer(prompt, retrieval.rows, top_k)
            user_event = write_memory(
                st.session_state.session_id, "user", prompt, {"source": "databricks_app"}
            )
            assistant_event = write_memory(
                st.session_state.session_id,
                "assistant",
                fm_response.text,
                {"model": fm_response.model, "top_k": top_k},
            )
        total_latency = int((time.perf_counter() - start) * 1000)
        st.session_state.request_latencies.append(total_latency)

        st.success("Agent response ready")
        st.write(fm_response.text)
        st.metric("Vector latency (ms)", retrieval.latency_ms)
        st.metric("FM latency (ms)", fm_response.latency_ms)
        st.metric("Total latency (ms)", total_latency)
        st.caption(f"Memory event IDs: user={user_event.event_id}, assistant={assistant_event.event_id}")
        st.json(retrieval.rows[:3])

with right:
    st.subheader("Session Memory (Lakebase)")
    if st.button("Refresh memory"):
        events = read_memory(st.session_state.session_id, limit=20)
        st.write(f"Events: {len(events)}")
        st.json(events)

    st.subheader("Tracing + Metrics")
    if st.button("Verify traces"):
        trace_summary = verify_traces(max_results=20)
        st.json(trace_summary)
    if st.button("Show latency summary"):
        samples = st.session_state.request_latencies
        if not samples:
            st.info("No latency samples yet.")
        else:
            st.json(
                {
                    "count": len(samples),
                    "p50_ms": int(statistics.median(samples)),
                    "p95_ms": int(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]),
                    "max_ms": max(samples),
                }
            )
