# =========================================================
# AI KNOWLEDGE GRAPH QUERY SYSTEM
# Professional User-Friendly Version
# =========================================================

import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Knowledge Graph Query System",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("AI Knowledge Graph Query System")

st.markdown("""
This system retrieves **relevant order information**
using FAISS similarity search and AI embeddings.
""")

# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_index():
    return faiss.read_index("index/faiss_index.index")


@st.cache_data
def load_texts():
    with open("index/sample_chunk_texts.pkl", "rb") as f:
        return pickle.load(f)


try:

    model = load_model()
    index = load_index()
    texts = load_texts()

    st.success("System Loaded Successfully")

except Exception as e:

    st.error("System loading failed")
    st.error(e)
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("System Information")

st.sidebar.write(
    f"Total Records: {len(texts)}"
)

st.sidebar.write(
    f"Embedding Dimension: {index.d}"
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Technology Stack**

- Python  
- Streamlit  
- FAISS  
- Sentence Transformers  
- Knowledge Graph  
""")


# =========================================================
# SMART SEARCH FUNCTION
# =========================================================

def smart_search(query, k=5):

    query_lower = query.lower()

    query_embedding = model.encode(
        [query]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k * 3
    )

    results = []

    for idx in indices[0]:

        text = texts[idx]
        text_lower = text.lower()

        # Smart filtering logic

        if "not delivered" in query_lower:

            if "status delivered" not in text_lower:
                results.append(text)

        elif "delivered" in query_lower:

            if "status delivered" in text_lower:
                results.append(text)

        elif "pending" in query_lower:

            if "status pending" in text_lower:
                results.append(text)

        elif "shipped" in query_lower:

            if "status shipped" in text_lower:
                results.append(text)

        else:

            results.append(text)

        if len(results) >= k:
            break

    return results


# =========================================================
# USER QUERY SECTION
# =========================================================

st.subheader("Query Interface")

query = st.text_input(
    "Enter your query:",
    placeholder="Example: Show delivered orders"
)

top_k = st.slider(
    "Number of results",
    1,
    10,
    5
)

# =========================================================
# SEARCH BUTTON
# =========================================================

if st.button("Search"):

    if query.strip() == "":

        st.warning(
            "Please enter a query."
        )

    else:

        with st.spinner(
            "Searching relevant records..."
        ):

            results = smart_search(
                query,
                top_k
            )

        st.subheader("Search Results")

        if len(results) == 0:

            st.error(
                "No matching results found."
            )

        else:

            for i, result in enumerate(results):

                st.markdown(
                    f"""
                    **Result {i+1}**

                    {result}

                    ---
                    """
                )


# =========================================================
# EXAMPLE QUERIES
# =========================================================

st.markdown("### Example Queries")

example_queries = [
    "Show delivered orders",
    "Show pending orders",
    "Show not delivered orders",
    "Show shipped products",
    "Orders placed in category electronics"
]

for q in example_queries:

    st.write("•", q)


# =========================================================
# DATA PREVIEW
# =========================================================

st.markdown("---")

st.subheader("Sample Data Preview")

try:

    preview_df = pd.DataFrame(
        texts[:10],
        columns=["Text Records"]
    )

    st.dataframe(
        preview_df,
        width="stretch"
    )

except:

    st.warning(
        "Data preview not available."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
"""
AI Knowledge Graph Dashboard  
Infosys Springboard Internship Project
"""
)
