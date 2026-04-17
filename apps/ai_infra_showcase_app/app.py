import logging
import os
import statistics
import time
import uuid

import mlflow
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

from framework.fm_agent_utils import FmAgentClient
from framework.lakebase_utils import LakebaseMemoryStore
from framework.mlflow_tracing_utils import configure_tracing, traced, verify_traces
from framework.vector_search_utils import VectorSearchClient, format_context_block

st.set_page_config(page_title="AI Infra FM Agent + Lakebase + Vector Search", layout="wide")
st.title("AI Infra Agent Framework (FM Endpoint First)")

mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/ai-infra-fm-agent-demo")
configure_tracing(
    experiment_name=mlflow_experiment,
    trace_destination=os.environ.get("MLFLOW_TRACING_DESTINATION"),
)

fm_client = FmAgentClient()
vs_client = VectorSearchClient()
memory_store_error = None
try:
    memory_store = LakebaseMemoryStore()
except Exception as exc:  # pragma: no cover - runtime integration
    memory_store = None
    memory_store_error = str(exc)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:10]}"
if "request_latencies" not in st.session_state:
    st.session_state.request_latencies = []


@traced(name="ai_infra_retrieve_context", span_type="RETRIEVER")
def retrieve_context(query: str, top_k: int):
    return vs_client.retrieve(query, top_k=top_k)


@traced(name="ai_infra_fm_generate", span_type="CHAT_MODEL")
def generate_answer(query: str, context_rows: list[list], top_k: int):
    context_block = format_context_block(context_rows, top_k=top_k)
    system_prompt = (
        "You are an industrial operations assistant. Use only provided retrieved context. "
        "If context is insufficient, say so clearly. Keep response concise and actionable."
    )
    user_prompt = (
        f"User question:\n{query}\n\nRetrieved context:\n{context_block}\n\n"
        "Respond with 3 sections: Summary, Recommended Actions, Risk Notes."
    )
    return fm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)


@traced(name="ai_infra_memory_write", span_type="TOOL")
def write_memory(session_id: str, role: str, content: str, metadata: dict):
    if memory_store is None:
        raise RuntimeError(f"Lakebase unavailable: {memory_store_error}")
    return memory_store.write(session_id=session_id, role=role, content=content, metadata=metadata)


@traced(name="ai_infra_memory_read", span_type="TOOL")
def read_memory(session_id: str, limit: int):
    if memory_store is None:
        raise RuntimeError(f"Lakebase unavailable: {memory_store_error}")
    return memory_store.read(session_id=session_id, limit=limit)


@traced(name="ai_infra_agent_turn", span_type="CHAIN")
def run_agent_turn(session_id: str, prompt: str, top_k: int):
    retrieval = retrieve_context(prompt, top_k)
    fm_response = generate_answer(prompt, retrieval.rows, top_k)
    user_event = None
    assistant_event = None
    memory_error: str | None = None
    if memory_store is not None:
        try:
            user_event = write_memory(
                session_id,
                "user",
                prompt,
                {"source": "databricks_app"},
            )
            assistant_event = write_memory(
                session_id,
                "assistant",
                fm_response.text,
                {"model": fm_response.model, "top_k": top_k},
            )
        except Exception as exc:
            logger.exception("Lakebase memory write failed for session %s", session_id)
            memory_error = str(exc)
    return retrieval, fm_response, user_event, assistant_event, memory_error


@st.cache_data(ttl=10)
def dependency_status():
    fm_ok, fm_msg = fm_client.health()
    vs_ok, vs_msg = vs_client.health()
    if memory_store is None:
        lb_ok, lb_msg = False, f"Lakebase unavailable: {memory_store_error}"
        lb_endpoint = os.environ.get("LAKEBASE_ENDPOINT_RESOURCE", "not configured")
    else:
        lb_ok, lb_msg = memory_store.health()
        lb_endpoint = memory_store.endpoint_resource
    return {
        "fm_endpoint": {"ok": fm_ok, "message": fm_msg, "endpoint": fm_client.endpoint_name},
        "vector_search": {"ok": vs_ok, "message": vs_msg, "index": vs_client.index_name},
        "lakebase": {"ok": lb_ok, "message": lb_msg, "endpoint": lb_endpoint},
    }


def mlflow_experiment_url() -> str | None:
    host = os.environ.get("DATABRICKS_HOST")
    workspace_id = os.environ.get("DATABRICKS_WORKSPACE_ID")
    if not host:
        return None
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"https://{host}"
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment)
        if experiment is None:
            return None
        base = f"{host.rstrip('/')}/ml/experiments/{experiment.experiment_id}/traces"
        if workspace_id:
            return f"{base}?o={workspace_id}"
        return base
    except Exception:
        return None


with st.sidebar:
    st.subheader("Session")
    st.text_input("Session ID", key="session_id")
    st.markdown(f"FM endpoint: `{fm_client.endpoint_name}`")
    st.markdown(f"VS index: `{vs_client.index_name}`")
    lakebase_endpoint_text = (
        memory_store.endpoint_resource
        if memory_store is not None
        else os.environ.get("LAKEBASE_ENDPOINT_RESOURCE", "not configured")
    )
    st.markdown(f"Lakebase endpoint: `{lakebase_endpoint_text}`")
    auto_refresh = st.toggle("Auto-refresh dependency status (10s)", value=False)
    status = dependency_status()
    all_ok = all(item["ok"] for item in status.values())
    st.metric("Dependencies", "UP" if all_ok else "DEGRADED")
    st.caption(
        f"FM: {status['fm_endpoint']['message']} | "
        f"VS: {status['vector_search']['message']} | "
        f"LB: {status['lakebase']['message']}"
    )
    if st.button("Refresh status now"):
        dependency_status.clear()
        st.rerun()
    if auto_refresh:
        # Preserve session_state across refreshes by asking Streamlit to rerun
        # rather than reloading the parent window.
        components.html(
            """
            <script>
                setTimeout(function() {
                    const evt = new CustomEvent('streamlit:componentReady');
                    window.parent.document.dispatchEvent(evt);
                    window.parent.postMessage({type: 'streamlit:rerun'}, '*');
                }, 10000);
            </script>
            """,
            height=0,
            width=0,
        )
        time.sleep(10)
        dependency_status.clear()
        st.rerun()

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
            retrieval, fm_response, user_event, assistant_event, memory_error = run_agent_turn(
                st.session_state.session_id, prompt, top_k
            )
            if memory_store is None:
                st.warning(f"Lakebase memory write skipped: {memory_store_error}")
            elif memory_error:
                st.warning(f"Lakebase memory write failed: {memory_error}")
            elif not (user_event and assistant_event):
                st.warning("Lakebase memory write skipped due to runtime error.")
        total_latency = int((time.perf_counter() - start) * 1000)
        st.session_state.request_latencies.append(total_latency)

        st.success("Agent response ready")
        st.write(fm_response.text)
        st.metric("Vector latency (ms)", retrieval.latency_ms)
        st.metric("FM latency (ms)", fm_response.latency_ms)
        st.metric("Total latency (ms)", total_latency)
        if user_event and assistant_event:
            st.caption(f"Memory event IDs: user={user_event.event_id}, assistant={assistant_event.event_id}")
        st.json([r.as_dict() for r in retrieval.rows[:3]])

with right:
    st.subheader("Session Memory (Lakebase)")
    if st.button("Refresh memory"):
        try:
            events = read_memory(st.session_state.session_id, limit=20)
            st.write(f"Events: {len(events)}")
            st.json(events)
        except Exception as exc:
            st.warning(f"Lakebase memory read unavailable: {exc}")

    st.subheader("Tracing + Metrics")
    traces_ui_url = mlflow_experiment_url()
    if traces_ui_url:
        st.link_button("Open MLflow Traces UI", traces_ui_url)
    if st.button("Verify traces"):
        try:
            trace_summary = verify_traces(max_results=20)
            st.json(trace_summary)
        except Exception as exc:
            st.warning(f"Trace verification from app runtime failed: {exc}")
            if traces_ui_url:
                st.info(f"Open traces directly in MLflow UI: {traces_ui_url}")
    if st.button("Show latency summary"):
        samples = st.session_state.request_latencies
        if not samples:
            st.info("No latency samples yet.")
        else:
            # statistics.quantiles requires n >= 2; for a single sample, p95 == sample.
            if len(samples) >= 2:
                p95 = int(statistics.quantiles(samples, n=20)[18])
            else:
                p95 = int(samples[0])
            st.json(
                {
                    "count": len(samples),
                    "p50_ms": int(statistics.median(samples)),
                    "p95_ms": p95,
                    "max_ms": max(samples),
                }
            )
