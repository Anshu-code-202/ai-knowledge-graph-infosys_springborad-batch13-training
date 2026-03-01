# AI Knowledge Graph Builder

## Project Overview

The AI Knowledge Graph Builder is an end‑to‑end intelligent data processing and retrieval system designed to transform unstructured textual data into structured knowledge representations. The system extracts entities using Large Language Models (LLMs), builds relationships using a graph database, and enables intelligent semantic search through Retrieval Augmented Generation (RAG).

This project focuses on reducing hallucinations in LLM responses by grounding answers in verified contextual data stored within a Knowledge Graph and Vector Database.

---

## Objectives

* Perform data ingestion and preprocessing on raw enterprise datasets.
* Extract entities and relationships from textual data using LLM‑based Named Entity Recognition (NER).
* Build a scalable Knowledge Graph using a graph database.
* Implement semantic search using embeddings and vector similarity.
* Develop a Retrieval Augmented Generation (RAG) pipeline to provide accurate contextual responses.

---

## Milestones

### Milestone 1 — Data Ingestion and Preprocessing

**Goal:** Prepare clean and structured data from raw datasets.

Key Tasks:

* Create project repository and environment setup.
* Select relevant raw dataset from enterprise or open data sources.
* Implement data ingestion using Python and Pandas.
* Perform preprocessing:

  * Handling missing values
  * Removing duplicates
  * Cleaning special characters
  * Correcting data types
* Data transformation and normalization.
* Data enrichment through classification or feature engineering.
* Showcase workflow using Jupyter Notebook and a basic Flask interface.

---

### Milestone 2 — Entity Extraction and Knowledge Graph Construction

**Goal:** Convert textual data into structured graph relationships.

Key Tasks:

* Perform Named Entity Recognition (NER) using LLM or transformer models.
* Extract entities such as organizations, roles, locations, and skills.
* Define graph schema and relationship structure.
* Store entities as nodes and relationships in a graph database.
* Validate graph integrity and remove duplicate entities.
* Query graph data using graph query language.

Example Relationship:

```
(Company)-[HIRING]->(JobRole)
(JobRole)-[LOCATED_IN]->(Location)
```

---

### Milestone 3 — Semantic Search and RAG Pipeline

**Goal:** Enable intelligent contextual question answering.

Key Tasks:

* Generate embeddings using transformer models.
* Store embeddings inside a vector database.
* Implement semantic similarity search.
* Integrate LLM with graph database and vector database.
* Develop hybrid search combining graph reasoning and semantic retrieval.
* Reduce hallucinations by grounding LLM responses in retrieved context.

---

## Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning and NLP

* Transformers
* Sentence Transformers

### Knowledge Graph

* Neo4j

### Vector Database

* FAISS or Pinecone

### LLM Framework

* LangChain

### Backend

* Flask

### Development Tools

* Jupyter Notebook
* Git and GitHub

---

## Project Architecture

1. Raw Dataset Ingestion
2. Data Cleaning and Transformation
3. Entity Extraction using LLM
4. Knowledge Graph Construction
5. Embedding Generation
6. Vector Storage and Semantic Search
7. Hybrid Retrieval
8. Contextual Response Generation

---

## Installation

Clone the repository:

```
git clone <repository-url>
cd AI-Knowledge-Graph-Builder
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\\Scripts\\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

1. Run preprocessing notebooks for data ingestion and cleaning.
2. Execute entity extraction pipeline.
3. Populate the Knowledge Graph database.
4. Generate embeddings and store them in the vector database.
5. Start Flask application for querying the system.

---

## Repository Structure

```
AI-Knowledge-Graph-Builder/
│
├── data/
├── notebooks/
├── ingestion/
├── entity_extraction/
├── graph_builder/
├── embeddings/
├── rag_pipeline/
├── flask_app/
├── requirements.txt
└── README.md
```

---

## Expected Outcomes

* Structured Knowledge Graph from unstructured enterprise data.
* Improved semantic understanding of user queries.
* Context‑aware responses with minimized hallucinations.
* Scalable architecture for enterprise AI applications.

---

## Future Enhancements

* Advanced entity linking.
* Multi‑document ingestion pipelines.
* Graph visualization dashboards.
* Deployment using containerization and cloud services.

---

## Author

Anshu Arora
B.Tech Computer Science Engineering
