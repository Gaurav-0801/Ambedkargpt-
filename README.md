🧠 AmbedkarGPT — SEMRAG-Based RAG System

A fully local Semantic Knowledge-Augmented Retrieval Augmented Generation (SEMRAG) system built on Dr. B.R. Ambedkar’s works, following the exact architecture described in the SEMRAG research paper.
This system performs semantic chunking, knowledge graph construction, community detection, local & global graph-based retrieval, and LLM-based answer generation using a local LLM (Llama3 via Ollama).

✅ Built for Kalpit Pvt Ltd – AI Engineering Intern Technical Assignment
✅ Fully offline & interview-live-demo ready

🚀 Features

✅ Semantic Chunking with buffer merging, cosine distance thresholds, and token-aware sub-chunks  
✅ Knowledge Graph with entities, relationship evidence, and embeddings for retrieval  
✅ Community Detection (Louvain) + LLM-generated community summaries  
✅ Local Graph RAG Search (Equation 4) & Global Graph RAG Search (Equation 5) implementations  
✅ Prompt-engineered LLM answers with chunk-level citations  
✅ Rich CLI pipeline for chunking, graph building, community processing, and live Q&A  
✅ Test suite (chunking, retrieval, integration) for fast regression checks

📁 Project Structure
ambedkargpt/
├── data/
│   ├── Ambedkar_book.pdf
│   └── processed/
│       ├── chunks.json
│       ├── knowledge_graph.pkl
│       ├── community_partition.json
│       ├── community_reports.json
│       └── node_index.json
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py
│   │   └── buffer_merger.py
│   ├── graph/
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── retrieval/
│   │   ├── local_search.py
│   │   ├── global_search.py
│   │   └── ranker.py
│   ├── llm/
│   │   ├── llm_client.py
│   │   ├── answer_generator.py
│   │   └── prompt_templates.py
│   └── pipeline/
│       └── ambedkargpt.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_integration.py
│   └── conftest.py
├── config.yaml
├── requirements.txt
├── setup.py
├── demo.py
├── .gitignore
└── README.md

🛠️ Tech Stack

Python 3.9+

sentence-transformers

spaCy

PyPDF

scikit-learn

networkx

python-louvain (community detection)

Ollama (Llama3 / Mistral)

LangChain (optional)

✅ Environment Setup (MANDATORY)
1️⃣ Create Virtual Environment
Windows:
python -m venv .venv
.venv\Scripts\activate

Linux / Mac:
python3 -m venv .venv
source .venv/bin/activate

2️⃣ Install Dependencies

Option A: Using pip (recommended)
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Option B: Using setup.py (for development)
```bash
pip install -e .
python -m spacy download en_core_web_sm
```

3️⃣ Install & Run Ollama (Local LLM)

Download Ollama from:

https://ollama.com


Pull the model:

ollama run llama3

📄 Dataset

Place the provided dataset file here:

data/Ambedkar_book.pdf


This is the 94-page Ambedkar book used for:

Semantic chunking

Entity extraction

Knowledge graph creation

RAG-based question answering

🔄 Pipeline Execution (STEP-BY-STEP)
All configuration, thresholds, and paths live in `config.yaml`. Tweak buffer sizes, cosine thresholds, entity filters, retrieval K values, and LLM settings there before running the pipeline.

### CLI Workflow
The orchestrator exposes every pipeline stage via `typer` commands:

```bash
# 1. Semantic chunking (Algorithm 1 + buffer merge + token controls)
python -m src.pipeline.ambedkargpt chunk --config config.yaml

# 2. Build KG with entity embeddings & relationship evidence
python -m src.pipeline.ambedkargpt build-graph

# 3. Detect communities (Louvain) and persist partition
python -m src.pipeline.ambedkargpt detect-communities

# 4. Summarize each community via LLM
python -m src.pipeline.ambedkargpt summarize-communities

# 5. Launch live RAG Q&A loop (retrieves local + global context, cites chunks)
python -m src.pipeline.ambedkargpt run
```

Pipeline artifacts:

- `data/processed/chunks.json` – parent chunks + ~128-token overlapping sub-chunks + embeddings  
- `data/processed/knowledge_graph.pkl` – networkx graph with entity metadata and embeddings  
- `data/processed/community_partition.json` – node → community mapping  
- `data/processed/community_reports.json` – community summaries for global retrieval
Running `python -m src.pipeline.ambedkargpt run` loads all artifacts, executes Equation 4 local search + Equation 5 global search, merges results into a prompt, and produces answers with `[chunk_id]` citations. Quit with `exit`, `quit`, or `Ctrl+C`.

💬 Sample Demo Questions

Use these during interview:

"What is Dr. Ambedkar’s view on caste?"

"What did Ambedkar say about social justice?"

"Who opposed caste discrimination?"

"What is endogamy according to Ambedkar?"

📐 Architecture Implemented (Strict SEMRAG Compliance)
SEMRAG Component	Implemented
Algorithm 1 – Semantic Chunking	✅
Knowledge Graph Construction	✅
Entity & Relationship Extraction	✅
Community Detection	✅
Local Graph RAG Search (Eq. 4)	✅
Global Graph RAG Search (Eq. 5)	✅
LLM Prompt Integration	✅
Offline Execution	✅
🎯 Minimum Viable Product (MVP) — ✅ Completed

✅ Semantic chunking works on Ambedkar PDF

✅ Knowledge graph created

✅ Local RAG search implemented

✅ LLM answers questions

✅ Live demo ready

🧠 How This System Works (Technical Approach)

### 1. Semantic Chunking (Algorithm 1 from SEMRAG Paper)

The system implements semantic chunking using cosine similarity of sentence embeddings:

- **Sentence Extraction**: PDF pages are parsed and split into sentences using NLTK
- **Buffer Merging**: Neighboring sentences are merged with a configurable buffer window (default: 2) to preserve local context
- **Embedding Generation**: Each buffered sentence group is embedded using `all-MiniLM-L6-v2` from SentenceTransformers
- **Semantic Boundary Detection**: Cosine distance between consecutive embeddings is computed. When distance exceeds threshold (default: 0.28), a chunk boundary is created
- **Token-Aware Splitting**: Chunks respect maximum token limits (1024 tokens) and are further split into ~128-token sub-chunks with 128-token overlap for fine-grained retrieval

### 2. Knowledge Graph Construction

Entities and relationships are extracted to build a knowledge graph:

- **Entity Extraction**: spaCy NER model (`en_core_web_sm`) extracts entities (PERSON, ORG, GPE, WORK_OF_ART, EVENT)
- **Relationship Extraction**: Dependency parsing identifies relationships between entities co-occurring in sentences
- **Graph Building**: NetworkX graph is constructed with:
  - Nodes = entities (with embeddings, chunk references, page numbers)
  - Edges = relationships (with evidence sentences and relation types)
- **Entity Embeddings**: Each entity is embedded using the same sentence transformer model for similarity search

### 3. Community Detection

The knowledge graph is partitioned into thematic communities:

- **Algorithm**: Louvain community detection algorithm (via `python-louvain`)
- **Community Assignment**: Each entity node is assigned to a community based on graph structure
- **Community Summarization**: LLM (Llama3 via Ollama) generates summaries for each community, including:
  - Key entities and their relationships
  - Thematic insights
  - Evidence snippets from source chunks

### 4. Retrieval Strategies

Two complementary retrieval methods are implemented as per SEMRAG Equations 4 & 5:

#### Local Graph RAG Search (Equation 4)
- Query is embedded and compared against entity embeddings
- Entities with similarity ≥ τ_e (default: 0.35) are selected
- Chunks associated with these entities are retrieved
- Sub-chunks with similarity ≥ τ_d (default: 0.3) are ranked and top-K returned

#### Global Graph RAG Search (Equation 5)
- Query is compared against community summary embeddings
- Top-K communities are selected
- All chunks within selected communities are extracted
- Points (sub-chunks) are scored and ranked, top-K returned

### 5. Answer Generation

Retrieved context from both strategies is integrated:

- **Prompt Engineering**: Custom templates combine:
  - Local entity context with chunk citations
  - Global community summaries with point-level citations
  - User query
- **LLM Generation**: Llama3 generates answers grounded in retrieved context
- **Citation Tracking**: Chunk IDs are extracted and displayed for transparency

### Key Design Decisions

- **Fully Local**: All processing runs offline (Ollama for LLM, local embeddings)
- **Modular Architecture**: Each component (chunking, graph, retrieval, LLM) is independently testable
- **Configurable**: All thresholds, K values, and model parameters in `config.yaml`
- **Error Handling**: Comprehensive error handling with clear messages for missing files, invalid configs, etc.

🧪 Testing

Unit tests live under `tests/`:

```bash
pytest
```

- `tests/test_chunking.py` – token-aware chunk splitting + overlap behavior  
- `tests/test_retrieval.py` – Equation 4 & 5 retrieval logic with patched embeddings  
- `tests/test_integration.py` – Answer generator prompt + citation wiring

📝 Demo Script

A standalone demo script is provided for interview demonstrations:

```bash
python demo.py
```

This runs predefined questions and displays:
- Retrieved local entities and global communities
- Generated answers with citations
- Retrieval statistics

The script processes these questions:
1. "What is Dr. Ambedkar's view on caste?"
2. "What did Ambedkar say about social justice?"
3. "Who opposed caste discrimination?"
4. "What is endogamy according to Ambedkar?"
5. "How did Ambedkar describe social justice?"
6. "Which reforms did Ambedkar advocate for education?"
7. "Summarize Ambedkar's stance on liberty, equality, and fraternity."

Alternatively, use the interactive CLI mode:
```bash
python -m src.pipeline.ambedkargpt run
```

For each question, the system highlights retrieved chunk IDs shown beneath the answer to demonstrate grounded reasoning.

👨‍💻 Author

Gaurav Pant
📧 gauravpant.ind@gmail.com

🎓 AI Engineering Intern Candidate