"""Streamlit application for AI Knowledge Graph + RAG querying.

This app provides:
1) Retrieval-Augmented Generation (RAG)-style semantic retrieval using
   SentenceTransformer embeddings + FAISS vector search.
2) Knowledge graph visualization from Neo4j using NetworkX + PyVis.
3) Summary analytics for node and relationship types via Plotly.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL_PATH = "embedding_model"
DEFAULT_INDEX_PATH = "rag_index.faiss"
DEFAULT_METADATA_PATH = "metadata.pkl"
NEO4J_QUERY = "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50"


@dataclass
class AppAssets:
    """Container for all loaded assets used by the app."""

    model: SentenceTransformer
    index: faiss.Index
    metadata: Any


# -----------------------------
# Cached loading functions
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_sentence_transformer(model_path: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model."""
    return SentenceTransformer(model_path)


@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path: str) -> faiss.Index:
    """Load and cache FAISS index from disk."""
    return faiss.read_index(index_path)


@st.cache_data(show_spinner=False)
def load_metadata(metadata_path: str) -> Any:
    """Load and cache metadata pickle file."""
    with open(metadata_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def get_neo4j_driver(uri: str, username: str, password: str):
    """Create and cache Neo4j driver instance."""
    return GraphDatabase.driver(uri, auth=(username, password))


# -----------------------------
# RAG retrieval logic
# -----------------------------
def load_all_assets(
    model_path: str = DEFAULT_MODEL_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> AppAssets:
    """Load all essential AI assets for retrieval operations."""
    model = load_sentence_transformer(model_path)
    index = load_faiss_index(index_path)
    metadata = load_metadata(metadata_path)
    return AppAssets(model=model, index=index, metadata=metadata)


def normalize_metadata_item(item: Any, idx: int) -> Dict[str, Any]:
    """Standardize metadata object to a dictionary for rendering."""
    if isinstance(item, dict):
        normalized = dict(item)
    else:
        normalized = {"content": str(item)}

    normalized.setdefault("id", idx)
    normalized.setdefault("score", None)
    return normalized


def retrieve_top_k(
    query: str, model: SentenceTransformer, index: faiss.Index, metadata: Any, top_k: int
) -> List[Dict[str, Any]]:
    """Embed query, run FAISS similarity search, and return structured results."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []
    for rank, (doc_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if doc_idx < 0:
            continue

        try:
            item = metadata[doc_idx] if isinstance(metadata, (list, tuple, pd.Series)) else metadata.get(doc_idx)
        except Exception:
            item = None

        structured = normalize_metadata_item(item, doc_idx)
        structured["rank"] = rank
        structured["similarity_score"] = float(score)
        results.append(structured)

    return results


# -----------------------------
# Neo4j + graph construction
# -----------------------------
def fetch_graph_records(driver, cypher_query: str = NEO4J_QUERY) -> List[Dict[str, Any]]:
    """Fetch triples from Neo4j graph database."""
    records: List[Dict[str, Any]] = []
    with driver.session() as session:
        data = session.run(cypher_query)
        for row in data:
            records.append(
                {
                    "n": row["n"],
                    "r": row["r"],
                    "m": row["m"],
                }
            )
    return records


def records_to_networkx(records: List[Dict[str, Any]]) -> nx.DiGraph:
    """Convert Neo4j records to a NetworkX directed graph."""
    graph = nx.DiGraph()

    for item in records:
        source = item["n"]
        rel = item["r"]
        target = item["m"]

        source_id = str(source.id)
        target_id = str(target.id)
        source_labels = list(source.labels)
        target_labels = list(target.labels)

        source_label = source.get("name") or source.get("title") or f"Node {source_id}"
        target_label = target.get("name") or target.get("title") or f"Node {target_id}"

        graph.add_node(source_id, label=source_label, node_type=",".join(source_labels) or "Unknown")
        graph.add_node(target_id, label=target_label, node_type=",".join(target_labels) or "Unknown")
        graph.add_edge(source_id, target_id, rel_type=rel.type)

    return graph


def build_pyvis_html(graph: nx.DiGraph) -> str:
    """Render NetworkX graph to an embeddable HTML string via PyVis."""
    net = Network(height="600px", width="100%", directed=True, bgcolor="#0e1117", font_color="#ffffff")
    net.barnes_hut()

    for node_id, attrs in graph.nodes(data=True):
        net.add_node(
            n_id=node_id,
            label=attrs.get("label", node_id),
            title=f"Type: {attrs.get('node_type', 'Unknown')}",
            group=attrs.get("node_type", "Unknown"),
        )

    for src, dst, attrs in graph.edges(data=True):
        net.add_edge(src, dst, label=attrs.get("rel_type", "RELATED_TO"), title=attrs.get("rel_type", "RELATED_TO"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_path = temp_file.name

    try:
        net.save_graph(temp_path)
        with open(temp_path, "r", encoding="utf-8") as html_file:
            return html_file.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_graph(graph: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare node type and relationship type summary dataframes."""
    node_types = [attrs.get("node_type", "Unknown") for _, attrs in graph.nodes(data=True)]
    rel_types = [attrs.get("rel_type", "RELATED_TO") for _, _, attrs in graph.edges(data=True)]

    node_df = (
        pd.Series(node_types, name="node_type")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "node_type", "node_type": "count"})
    )

    rel_df = (
        pd.Series(rel_types, name="relationship_type")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "relationship_type", "relationship_type": "count"})
    )

    return node_df, rel_df


# -----------------------------
# Streamlit UI
# -----------------------------
def render_sidebar() -> Tuple[str, int]:
    """Render sidebar controls and return user query + top_k selection."""
    st.sidebar.header("Project Description")
    st.sidebar.write(
        "This dashboard combines semantic retrieval using FAISS + SentenceTransformer "
        "with Neo4j knowledge graph exploration and interactive analytics."
    )
    query = st.sidebar.text_input("Enter your question", placeholder="Ask something about the knowledge base...")
    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=5, step=1)
    return query, top_k


def render_rag_results(results: List[Dict[str, Any]]) -> None:
    """Render structured retrieval results in Streamlit."""
    st.subheader("Retrieved Context")

    if not results:
        st.info("No results found for the current query.")
        return

    for result in results:
        with st.expander(f"Result #{result.get('rank')} | ID: {result.get('id')}", expanded=False):
            st.write(f"**Similarity Score:** `{result.get('similarity_score'):.4f}`")
            clean_payload = {k: v for k, v in result.items() if k not in {"rank", "similarity_score"}}
            st.json(clean_payload)


def render_graph_section(graph: nx.DiGraph) -> None:
    """Render graph visualization and analytics charts."""
    st.subheader("Knowledge Graph Visualization")

    graph_html = build_pyvis_html(graph)
    st.components.v1.html(graph_html, height=620, scrolling=True)

    node_df, rel_df = summarize_graph(graph)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Node Type Distribution")
        if node_df.empty:
            st.info("No node data available.")
        else:
            node_bar = px.bar(node_df, x="node_type", y="count", color="node_type", title="Node Types")
            node_bar.update_layout(showlegend=False)
            st.plotly_chart(node_bar, use_container_width=True)

    with col2:
        st.markdown("### Relationship Type Distribution")
        if rel_df.empty:
            st.info("No relationship data available.")
        else:
            rel_pie = px.pie(rel_df, names="relationship_type", values="count", title="Relationship Types")
            st.plotly_chart(rel_pie, use_container_width=True)


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="AI Knowledge Graph Query System", layout="wide")
    st.title("AI Knowledge Graph Query System")

    query, top_k = render_sidebar()

    # Section for model/index loading status
    with st.container(border=True):
        st.markdown("### System Status")
        try:
            assets = load_all_assets()
            st.success("Embedding model, FAISS index, and metadata loaded successfully.")
        except FileNotFoundError as file_error:
            st.error(f"Required file not found: {file_error}")
            return
        except Exception as ex:
            st.error(f"Failed to load model/index/metadata: {ex}")
            return

    # Query processing section
    with st.container(border=True):
        st.markdown("### Semantic Query Processing")
        if query.strip():
            with st.spinner("Processing your query..."):
                try:
                    results = retrieve_top_k(
                        query=query,
                        model=assets.model,
                        index=assets.index,
                        metadata=assets.metadata,
                        top_k=top_k,
                    )
                    render_rag_results(results)
                except Exception as ex:
                    st.error(f"Error while processing query: {ex}")
        else:
            st.info("Enter a query in the sidebar to retrieve relevant context.")

    # Graph section
    with st.container(border=True):
        st.markdown("### Knowledge Graph Explorer")

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        with st.expander("Neo4j Connection Settings", expanded=False):
            st.code(
                f"NEO4J_URI={neo4j_uri}\nNEO4J_USER={neo4j_user}\nNEO4J_PASSWORD={'*' * len(neo4j_password)}",
                language="bash",
            )

        try:
            driver = get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)
            records = fetch_graph_records(driver)
            if not records:
                st.warning("No graph relationships were returned from Neo4j.")
            else:
                graph = records_to_networkx(records)
                render_graph_section(graph)
        except Exception as ex:
            st.error(f"Unable to load graph from Neo4j: {ex}")


if __name__ == "__main__":
    main()
