# AI Knowledge Graph Builder using RAG, Neo4j and FAISS

## Project Overview

The AI Knowledge Graph Builder is an end-to-end intelligent data processing and retrieval system designed to transform unstructured textual data into structured knowledge representations. The system extracts entities using Large Language Models (LLMs), constructs relationships using a graph database, and enables intelligent semantic search through Retrieval Augmented Generation (RAG).

This project focuses on reducing hallucinations in LLM-generated responses by grounding answers in verified contextual data stored within a Knowledge Graph and Vector Database. The architecture supports scalable enterprise-level data understanding and intelligent querying.

---

## Problem Statement

Traditional search systems fail to understand relationships between entities in unstructured data, leading to incomplete or inaccurate results.

This project addresses these challenges using:

- Knowledge Graph-based structured relationships
- Vector-based semantic similarity search
- Retrieval Augmented Generation (RAG) pipeline
- Hybrid search combining graph reasoning and semantic retrieval

---

## Key Features

- Automated entity extraction using Large Language Models
- Relationship detection and Knowledge Graph construction
- Storage and querying using Neo4j graph database
- Semantic similarity search using FAISS vector database
- Hybrid Retrieval-Augmented Generation pipeline
- Reduced hallucination through context grounding
- Scalable architecture for enterprise-level knowledge systems
- Flask-based backend for interactive querying

---

## System Architecture

Raw Data  
↓  
Data Cleaning and Preprocessing  
↓  
Entity Extraction using LLM  
↓  
Relationship Extraction  
↓  
Knowledge Graph Construction (Neo4j)  
↓  
Embedding Generation  
↓  
Vector Storage using FAISS  
↓  
Semantic Retrieval  
↓  
Hybrid RAG Query Processing  
↓  
Contextual Response Generation  

---

## Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas  
- NumPy  

### Machine Learning and NLP

- Transformers  
- Sentence Transformers  

### Knowledge Graph

- Neo4j  

### Vector Database

- FAISS  

### LLM Framework

- LangChain  
- Gemini / OpenAI APIs  

### Backend Framework

- Flask  

### Development Tools

- Jupyter Notebook  
- Git  
- GitHub  

---

## Project Modules

### 1. Data Processing Module

Responsible for loading raw datasets and performing preprocessing tasks including:

- Handling missing values  
- Removing duplicates  
- Cleaning special characters  
- Data normalization  
- Feature enrichment  

---

### 2. Entity Extraction Module

Uses transformer-based models or LLM APIs to:

- Identify named entities  
- Classify entity types  
- Extract meaningful structured information  

---

### 3. Relationship Extraction Module

Establishes meaningful connections between extracted entities and defines graph relationships.

Example:

(Company) -[HIRING]-> (JobRole)  
(JobRole) -[LOCATED_IN]-> (Location)

---

### 4. Knowledge Graph Module

Responsible for:

- Creating graph schema  
- Storing nodes and relationships  
- Querying data using graph queries  
- Maintaining graph integrity  

---

### 5. Embedding and Vector Storage Module

Responsible for:

- Generating embeddings from textual data  
- Storing embeddings in FAISS  
- Supporting semantic similarity search  

---

### 6. RAG Pipeline Module

Combines:

- Vector search retrieval  
- Knowledge graph reasoning  
- LLM response generation  

This module ensures:

- Context-aware responses  
- Reduced hallucination  
- Improved answer relevance  

---

## Project Structure
AI-Knowledge-Graph-Builder/

├── data/
│ ├── raw/
│ ├── processed/
│
├── notebooks/
│ ├── milestone1_data_ingestion.ipynb
│ ├── milestone2_entity_extraction.ipynb
│ ├── milestone3_graph_construction.ipynb
│ ├── milestone4_rag_pipeline.ipynb
│
├── ingestion/
├── entity_extraction/
├── graph_builder/
├── embeddings/
├── rag_pipeline/
│
├── requirements.txt
├── README.md
screenshots/
│
├── embedding_generation.png
├── faiss_index_creation.png
├── rag_query_output.png
----

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anshu-code-202/ai-knowledge-graph-infosys_springborad-batch13-training.git
cd ai-knowledge-graph-infosys_springborad-batch13-training

## Future Enhancements

* Advanced entity linking.
* Multi‑document ingestion pipelines.
* Graph visualization dashboards.
* Deployment using containerization and cloud services.

---

## Author

Anshu Arora
B.Tech Computer Science Engineering
