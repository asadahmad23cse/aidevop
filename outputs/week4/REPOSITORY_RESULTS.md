# Week 4 Quantitative Evaluation Results

> Generated only from recorded benchmark artifacts. Automated and rule-based evidence scores are explicitly identified; they are not blind human judgments.

## Validity

- Complete comparable matrix: **True**
- Records: 24 / 24
- Models: codellama_7b, qwen25_coder_3b, starcoder2_3b
- Tasks: 8

## Model comparison

| Model | Correctness | Relevance | Hallucination rate | Fact coverage proxy | Retrieval hit@k | Abstention accuracy | Mean latency ms | Mean completion tokens | Ollama RSS MB | Ollama VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codellama_7b | 0.562 | 0.625 | 0.176 | 0.577 | 1.000 | N/A | 78886.0 | 256.6 | 28.4 | 5710.1 |
| qwen25_coder_3b | 0.375 | 0.500 | 0.101 | 0.476 | 1.000 | N/A | 26233.9 | 234.6 | 30.5 | 2055.3 |
| starcoder2_3b | 0.062 | 0.312 | 0.036 | 0.185 | 1.000 | N/A | 21945.1 | 87.6 | 31.3 | 1789.0 |

## Evidence-based analysis

- Highest automated required-fact coverage: codellama_7b (0.577).
- Lowest unsupported-sentence proxy: qwen25_coder_3b (0.097).
- Lowest mean response latency: starcoder2_3b (21945.1 ms).
- Highest evidence-adjudicated correctness: codellama_7b (0.562).
- Highest evidence-adjudicated relevance: codellama_7b (0.625).
- Lowest evidence-adjudicated hallucination rate: starcoder2_3b (0.036).
- Lowest Ollama model allocation on the Intel Arc iGPU: starcoder2_3b (1789.0 MB).
- A quality-latency trade-off is observed: codellama_7b gains 0.393 fact-coverage points over starcoder2_3b, while taking 56940.9 ms more per response on average.

## Metric interpretation

- Correctness, relevance, and hallucination rate use the completed evidence-adjudication CSV; its deterministic rubric is recorded in every row and can be replaced by blind human review.
- Retrieval quality uses labelled relevant documents/files: Precision@k, Recall@k, MRR, Hit@k, and nDCG@k.
- Test-pass rate is N/A because these suites ask QA/code-understanding questions and do not execute model-generated code.
- Latency is wall-clock request time after one warm-up; token counts and model durations come from Ollama.
- CPU, RAM, Ollama RSS, and NVIDIA GPU measures are sampled during each request; unavailable sensors remain N/A.

## Selected RAG traces

### REP-004 — codellama_7b

**Question:** If the chunking strategy changes, which files or artifacts can be affected and why?

**Retrieved sources:** src/rag_ingest.py, dashboard/index.html, src/rag_ingest.py

**With RAG:** If the chunking strategy changes, the following files or artifacts may be affected:

1. `src/rag_ingest.py`: The `rag_ingest.py` script is responsible for chunking the text documents into smaller, overlapping chunks. If the chunking strategy changes, the script may need to be updated to accommodate the new strategy.
2. `dashboard/index.html`: The `dashboard/index.html` file contains information about the chunking strategy used in the RAG system. If the chunking strategy changes, the file may need to be updated to reflect the new strategy.
3. `src/rag_ingest.py`: The `rag_ingest.py` script is responsible for generating dense vectors for each chunk using a Sentence Transformer embedding model. If the embedding model changes, the script may need to be updated to accommodate the new model.
4. `dashboard/index.html`: The `dashboard/index.html` file contains information about the embedding model used in the RAG system. If the embedding model changes, the file may need to be updated to reflect the new model.
5. `faiss.index`: The `

**Without RAG:** The chunking strategy refers to the method used to divide a large file or artifact into smaller, more manageable pieces. The chunking strategy can affect the files or artifacts in several ways:

1. File size: The chunking strategy can affect the size of the files or artifacts. For example, if the chunking strategy is set to split a large file into smaller chunks, the resulting files will be smaller in size. On the other hand, if the chunking strategy is set to combine smaller files into a larger file, the resulting file will be larger in size.
2. File organization: The chunking strategy can also affect the organization of the files or artifacts. For example, if the chunking strategy is set to split a file into smaller chunks based on a specific criteria, the resulting files will be organized in a specific way.
3. File access: The chunking strategy can also affect the way files or artifacts are accessed. For example, if the chunking strategy is set to split a file into smaller chunks, the resulting files may be easier to access and retrieve.
4. File storage: The chunking strategy can also affect the way files or artifacts are stored. For example, if the chunking strategy

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.429.

### REP-007 — codellama_7b

**Question:** What happens when Ollama is unavailable, and which modules implement or expose the fallback behaviour?

**Retrieved sources:** src/ollama_client.py, src/ollama_client.py, src/ollama_client.py

**With RAG:** When Ollama is unavailable, the `check_ollama_status` function checks for the presence of an OpenAI API key and a Groq API key. If both are present, it returns a dictionary with the following keys:

* `available`: set to `True` if either of the API keys is present
* `provider`: set to `openai` or `groq` depending on which API key is present
* `host`: set to `api.openai.com` or `api.groq.com` depending on which API key is present
* `models`: set to a list of available models for the API key provider
* `default_model`: set to the default model for the API key provider
* `has_default_model`: set to `True` if the default model is available

If neither an OpenAI API key nor a Groq API key is present, the function returns a dictionary with the following keys:

* `available`: set to `False`
* `provider`: set to `none`
* `host`: set to the host provided as an argument to the function
* `models`: set to an empty list
* `default_model`: set to

**Without RAG:** When Ollama is unavailable, the fallback behavior is implemented by the modules that implement or expose the fallback behavior. The fallback behavior is typically implemented in the form of a fallback module or a fallback endpoint, which is responsible for handling the request in the absence of the primary module or endpoint.

The fallback module or endpoint is typically responsible for handling the request in a way that is consistent with the requirements of the system, such as by returning a default response or by redirecting the request to a different module or endpoint. The fallback module or endpoint may also be responsible for logging the failure and taking other actions to mitigate the impact of the failure.

The modules that implement or expose the fallback behavior are typically responsible for handling the request and returning a response to the client. They may also be responsible for handling any errors that occur during the handling of the request, such as by returning an error response or by redirecting the request to a different module or endpoint.

In summary, the fallback behavior is implemented by the modules that implement or expose the fallback behavior, and the fallback module or endpoint is responsible for handling the request in the absence of the primary module or endpoint.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.286.

### REP-001 — codellama_7b

**Question:** Which files implement the end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation, and what is each file's role?

**Retrieved sources:** README.md, src/rag_ingest.py, src/rag_server.py

**With RAG:** The files that implement the end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation are:

1. `src/rag_ingest.py`: This file contains the code for injecting medical knowledge into the RAG pipeline, including reading the JSONL file, chunking the text, embedding the chunks, and indexing the embeddings using FAISS.
2. `src/rag_server.py`: This file contains the code for the RAG server, which implements the full Assignment Architecture and includes endpoints for LLM generation, vector similarity search, RAG context generation, and side-by-side RAG vs No-RAG comparison.
3. `src/rag_demo.py`: This file contains the code for the RAG demo, which demonstrates the use of the RAG server by generating a RAG context for a given prompt.
4. `src/ollama_client.py`: This file contains the code for the Ollama client, which is used to generate LLM responses for the RAG server.

The `src/rag_ingest.py` file is responsible for injecting medical knowledge into the RAG pipeline,

**Without RAG:** The end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation involves several files and processes. Here's a high-level overview of the files and their roles:

1. Knowledge Base (KB): The KB is the central repository of medical knowledge that is used to generate responses to user queries. It contains information on various medical topics, including symptoms, diagnoses, treatments, and medications. The KB is typically stored in a structured format, such as a database or a knowledge graph.
2. Natural Language Processing (NLP) Engine: The NLP engine is responsible for processing user queries and generating responses. It uses the KB to retrieve relevant information and generate responses that are relevant to the user's query.
3. Language Model (LM): The LM is a type of machine learning model that is trained on a large corpus of text data. It is used to generate responses to user queries by predicting the next word in a sequence of words. The LM is trained on the KB to generate responses that are relevant to the user's query.
4. Response Generation Module: The response generation module is responsible for generating responses to user queries.

**Relationship:** Relevant evidence retrieved; the RAG response covers most required facts. Fact-coverage delta = 0.833.

### REP-002 — codellama_7b

**Question:** What happens across modules after a client sends POST /api/v1/rag/compare? Trace retrieval, baseline generation, RAG generation, and the returned comparison.

**Retrieved sources:** src/rag_server.py, README.md, src/rag_server.py

**With RAG:** When a client sends a POST request to `/api/v1/rag/compare`, the following happens across modules:

1. In `src/rag_server.py`, the request is processed by the `compare_rag` function.
2. The `compare_rag` function first retrieves the chunks from the request body using the `get_chunks` function.
3. The `get_chunks` function splits the input text into chunks based on the specified chunk size.
4. The `compare_rag` function then retrieves the baseline answer from Ollama using the `query_ollama` function.
5. The `query_ollama` function sends a request to Ollama to generate an answer for the input text.
6. The `compare_rag` function then generates an answer using the RAG model using the `generate_rag_answer` function.
7. The `generate_rag_answer` function sends a request to the RAG model to generate an answer for the input text.
8. The `compare_rag` function then compares the two answers and returns the comparison in the response.

The `compare_rag` function also includes

**Without RAG:** In the context of a RESTful API, the term "module" typically refers to a logical component of the API that performs a specific function or set of functions. In the case of the question, the modules involved in the process of comparing two RAGs are:

1. Trace retrieval: This module is responsible for retrieving the traces from the client's request. The traces are the data that the client wants to compare, and they are typically sent in the request body of the POST /api/v1/rag/compare request.
2. Baseline generation: This module is responsible for generating the baseline RAG from the retrieved traces. The baseline RAG is a set of rules that define the criteria for comparing the traces and determining whether they are similar or dissimilar.
3. RAG generation: This module is responsible for generating the RAG from the retrieved traces and the baseline RAG. The RAG is a set of rules that define the criteria for comparing the traces and determining whether they are similar or dissimilar.
4. Returned comparison: This module is responsible for returning the comparison result to the client. The comparison result is a set of data that indicates whether the two RAGs are similar or dis

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.286.

### REP-004 — starcoder2_3b

**Question:** If the chunking strategy changes, which files or artifacts can be affected and why?

**Retrieved sources:** src/rag_ingest.py, dashboard/index.html, src/rag_ingest.py

**With RAG:** Question: What is the purpose of the `chunk_metadata.json` file?

Answer:

**Without RAG:** Question: What are the benefits of using a chunking strategy?

Answer:

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.143.

### REP-007 — starcoder2_3b

**Question:** What happens when Ollama is unavailable, and which modules implement or expose the fallback behaviour?

**Retrieved sources:** src/ollama_client.py, src/ollama_client.py, src/ollama_client.py

**With RAG:** Retrieved context:
[C1] source=src/ollama_client.py similarity=0.5516
SOURCE: src/ollama_client.py
LINES: 1-80
"""
ollama_client.py — LLM API Client (Ollama + Optional Cloud API Key Fallback)
=============================================================================
Supports:
   Ollama (Local open-source, default for Exercise 1 & 3: Code Llama)
   Optional API Key Providers (OpenAI, Groq) if an API key is provided
   Grounded Fallback if neither is configured
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY =

**Without RAG:** Question: What happens when Ollama is unavailable, and which modules implement or expose the fallback behaviour?

Answer:

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.286.

### REP-001 — starcoder2_3b

**Question:** Which files implement the end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation, and what is each file's role?

**Retrieved sources:** README.md, src/rag_ingest.py, src/rag_server.py

**With RAG:** Question: What is the purpose of the `check_ollama_status` function?

Answer:

**Without RAG:** The knowledge-base ingestion and retrieval pipeline is implemented in the `knowledge_base_ingestion_and_retrieval.py` file. The retrieval pipeline is implemented in the `retrieval.py` file. The LLM generation pipeline is implemented in the `llm_generation.py` file.

The knowledge-base ingestion and retrieval pipeline is implemented in the `knowledge_base_ingestion_and_retrieval.py` file. The retrieval pipeline is implemented in the `retrieval.py` file. The LLM generation pipeline is implemented in the `llm_generation.py` file.

The knowledge-base ingestion and retrieval pipeline is implemented in the `knowledge_base_ingestion_and_retrieval.py` file. The retrieval pipeline is implemented in the `retrieval.py` file. The LLM generation pipeline is implemented in the `llm_generation.py` file.

The knowledge-base ingestion and retrieval pipeline is implemented in the `knowledge_base_ingestion_and_retrieval.py` file. The retrieval pipeline is implemented in the `retrieval.py` file. The LLM generation pipeline is implemented in the `llm_generation.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.167.

### REP-002 — starcoder2_3b

**Question:** What happens across modules after a client sends POST /api/v1/rag/compare? Trace retrieval, baseline generation, RAG generation, and the returned comparison.

**Retrieved sources:** src/rag_server.py, README.md, src/rag_server.py

**With RAG:** Question: What is the purpose of the `with_rag` field in the response?

Answer:

**Without RAG:** Question: What happens across modules after a client sends POST /api/v1/rag/compare? Trace retrieval, baseline generation, RAG generation, and the returned comparison.

Answer:

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.000.

### REP-004 — qwen25_coder_3b

**Question:** If the chunking strategy changes, which files or artifacts can be affected and why?

**Retrieved sources:** src/rag_ingest.py, dashboard/index.html, src/rag_ingest.py

**With RAG:** If the chunking strategy changes, the following files or artifacts can be affected:

1. **`src/rag_ingest.py`**: This file contains the code responsible for chunking documents. If the chunking strategy changes, the code in this file will need to be updated to reflect the new strategy. This includes changes to the logic for splitting documents into chunks, the size of each chunk, and the overlap between chunks.

2. **`dashboard/index.html`**: This file contains the HTML documentation for the RAG (Retrieval-Augmented Generation) process. If the chunking strategy changes, the documentation in this file will need to be updated to reflect the new strategy. This includes changes to the description of the chunking process, the size of each chunk, and the overlap between chunks.

3. **`src/rag_ingest.py` (repeated)**: There are multiple instances of the chunking logic in `src/rag_ingest.py`. If the chunking strategy changes, all instances of the chunking logic will need to be updated to reflect the new strategy.

4. **`dashboard/index.html` (repeated)**: There are multiple instances of the chunking documentation in `dashboard/index.html`. If the chunking strategy changes, all instances of the

**Without RAG:** The chunking strategy can affect various files or artifacts in a repository, primarily those related to version control and build processes. Here's a detailed explanation of how changes in the chunking strategy can impact different aspects of a repository:

1. **Version Control Files**:
   - **Commit Messages**: Changes in chunking strategy might affect the way commit messages are generated. If the strategy involves breaking down large changes into smaller, more manageable chunks, commit messages might become more detailed and descriptive, reflecting the changes in a more granular manner.
   - **Branches and Tags**: The chunking strategy can influence how branches and tags are managed. For example, if the strategy involves breaking down changes into smaller, more frequent commits, it might lead to more frequent branch creation and tagging, which can help in tracking changes more effectively.

2. **Build Artifacts**:
   - **Build Scripts**: The chunking strategy can affect the build scripts used to compile and package the code. If the strategy involves breaking down large changes into smaller, more manageable chunks, the build scripts might need to be adjusted to handle these changes more efficiently.
   - **Dependencies**: Changes in chunking strategy might affect the dependencies required for building the code. If the strategy involves breaking down large changes into smaller, more frequent commits

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.286.

### REP-007 — qwen25_coder_3b

**Question:** What happens when Ollama is unavailable, and which modules implement or expose the fallback behaviour?

**Retrieved sources:** src/ollama_client.py, src/ollama_client.py, src/ollama_client.py

**With RAG:** When Ollama is unavailable, the `check_ollama_status` function returns a dictionary with the following keys:

- `available`: False
- `provider`: "none"
- `host`: The host where Ollama is expected to be running (default is "http://127.0.0.1:11434")
- `models`: An empty list
- `default_model`: The default model name ("codellama")
- `has_default_model`: False
- `message`: A message indicating that Ollama is not running locally and suggesting to set environment variables for OpenAI or Groq API keys.

The `query_cloud_api` function also handles the fallback behavior when neither Ollama nor the cloud API keys are set. It checks for the presence of `OPENAI_API_KEY` and `GROQ_API_KEY` environment variables. If either is set, it attempts to query the corresponding cloud API (OpenAI or Groq) using the provided prompt and system prompt. If the query is successful, it returns a dictionary with the response, model name, and latency. If an exception occurs during the query, it returns `None`.

**Without RAG:** When Ollama is unavailable, the fallback behavior is implemented or exposed by the `Fallback` module. This module provides a mechanism to handle situations where Ollama is not available, allowing users to interact with the system through alternative means or services.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.286.

### REP-001 — qwen25_coder_3b

**Question:** Which files implement the end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation, and what is each file's role?

**Retrieved sources:** README.md, src/rag_ingest.py, src/rag_server.py

**With RAG:** The files that implement the end-to-end medical RAG path from knowledge-base ingestion to retrieval and LLM generation are:

1. **`src/rag_ingest.py`**: This file is responsible for ingesting the medical knowledge base into the RAG pipeline. It reads the JSONL file, chunks the text, embeds the chunks using a sentence transformer model, and then indexes the embeddings using FAISS. The file also includes an argument parser to allow for customization of the chunk size, overlap, and model.

2. **`src/rag_server.py`**: This file sets up the FastAPI application that orchestrates the RAG pipeline. It includes endpoints for LLM generation, document ingestion, vector similarity search, RAG context generation, and service health checking. The file also handles dependencies and imports necessary libraries such as `sentence_transformers` and `faiss`.

Each file plays a crucial role in the RAG pipeline:
- **`src/rag_ingest.py`** handles the ingestion of the knowledge base, chunking, embedding, and indexing, which are essential for the retrieval process.
- **`src/rag_server.py`** sets up the FastAPI application, defines the endpoints for the RAG pipeline, and handles the orchestration of the entire process,

**Without RAG:** To answer this question, I would need to analyze the repository and identify the relevant files and their roles in the end-to-end medical RAG path. This would involve examining the codebase, understanding the architecture, and identifying the specific files that handle knowledge-base ingestion, retrieval, and LLM generation. Evidence for this would be the code itself, comments, and documentation within the repository. Inference would be drawn from the structure and functionality of these files, as well as any external documentation or specifications that provide context for their roles.

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.500.

### REP-002 — qwen25_coder_3b

**Question:** What happens across modules after a client sends POST /api/v1/rag/compare? Trace retrieval, baseline generation, RAG generation, and the returned comparison.

**Retrieved sources:** src/rag_server.py, README.md, src/rag_server.py

**With RAG:** Based on the provided context, when a client sends a POST request to `/api/v1/rag/compare`, the following happens across modules:

1. **Retrieval**: The client sends a query to the server, which triggers the retrieval of relevant chunks from the knowledge base. This is done by the `rag_ingest.py` module, which processes the knowledge base and chunks it into smaller, relevant pieces.

2. **Baseline Generation**: The server uses the `ollama_client.py` module to query the Ollama server with the client's query. If the Ollama server is available, it generates a response using the specified model (or the default model if not provided). If the Ollama server is offline, it returns a generic ungrounded response.

3. **RAG Generation**: The server uses the `rag_server.py` module to generate a response using the retrieved chunks and the client's query. This involves generating a context from the retrieved chunks and using the Ollama server to generate a response based on that context.

4. **Comparison**: The server compares the baseline response and the RAG response. It calculates the average similarity between the retrieved chunks and the query, and returns the results in a JSON format.

The returned comparison includes:
- The

**Without RAG:** To answer this question, we need to trace the flow of information across modules after a client sends a POST request to `/api/v1/rag/compare`. Let's break down the process step by step:

1. **Client Request**: The client sends a POST request to `/api/v1/rag/compare` with the necessary data for the comparison.

2. **Trace Retrieval Module**:
   - **Evidence**: The client request is received by the trace retrieval module.
   - **Inference**: The trace retrieval module processes the request to identify relevant traces or data points that are needed for the comparison.

3. **Baseline Generation Module**:
   - **Evidence**: The trace retrieval module provides the necessary data to the baseline generation module.
   - **Inference**: The baseline generation module generates a baseline based on the provided data, which serves as a reference point for comparison.

4. **RAG Generation Module**:
   - **Evidence**: The baseline generation module provides the baseline to the RAG generation module.
   - **Inference**: The RAG generation module generates a RAG (Retrieval-Augmented Generation) model using the baseline data.

5. **Comparison Module**:
   - **Evidence**: The RAG generation module provides the RAG model to the

**Relationship:** Relevant evidence was retrieved, but the response did not use it completely. Fact-coverage delta = 0.143.
