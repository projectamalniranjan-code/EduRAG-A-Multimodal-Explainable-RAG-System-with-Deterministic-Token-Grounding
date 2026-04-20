# EduRAG — Explainable Retrieval-Augmented Generation for Education

> A research-grade RAG system built for educational Q&A with full explainability — citation grounding, token-level attribution, and hallucination detection.

---

## What It Does

EduRAG processes textbooks, research papers, and lecture slides into a searchable knowledge base and answers questions with:

* **Grounded answers** — every claim cited with [1], [2] references to source documents
* **Explainability** — shows which sources contributed most and which words in the answer are grounded in evidence
* **Hallucination detection** — faithfulness scoring and token-level grounding heatmap
* **Multimodal support** — diagrams and images described by a vision model and made searchable

Unlike basic RAG tutorials, this system uses a full production-grade retrieval pipeline and was formally evaluated and benchmarked.

---

## Benchmark Results

Evaluated on 97 text/table queries and 15 diagram-specific queries generated from real educational documents using DeepSeek R1 8B.

### Retrieval Performance *(Hybrid BM25 + FAISS + RRF + Cross-Encoder Reranker)*

**Text & Table Queries (n=97)**
| Metric | Score |
| :--- | :--- |
| MRR (Mean Reciprocal Rank) | 0.7784 |
| Recall@1 | 0.7010 |
| Recall@5 | 0.8763 |

**Diagram-Specific Queries (n=15) — LLaVA-Phi3 VLM Captions**
| Metric | Score | Notes |
| :--- | :--- | :--- |
| MRR (Visual) | 0.967 | Visual chunks ranked first in 93.3% of queries |
| Recall@1 (Visual) | 0.933 | Target diagram retrieved at rank 1 |
| Recall@5 (Visual) | 1.000 | All target diagrams within top 5 |
| Recall@2 (Visual) | 0.867 | Target diagram retrieved within top 2 |

### Generation Accuracy
| Metric | Score | Notes |
| :--- | :--- | :--- |
| BERTScore F1 | 0.8952 | Semantic similarity to ground truth |
| ROUGE-L | 0.4699 | Structural overlap |
| Token F1 | 0.4471 | Lexical overlap |
| Exact Match | 0.0000 | Expected 0 for open-ended generation |

### Trust & Hallucination Safeguards

**Text-Based Answers (n=97)**
| Metric | Score | Notes |
| :--- | :--- | :--- |
| Faithfulness | 0.8924 | BERTScore vs best-matching retrieved chunk |
| Citation Validity | 94% | All citations pointed to valid retrieved documents |
| Token Grounding | 85.5% | Content words grounded in context (stop words excluded) |

**Vision-Enhanced Answers (from Visual Queries, n=15)**
| Metric | Score | Notes |
| :--- | :--- | :--- |
| Visual Evidence Attribution | 34.2% | Average contribution of visual chunks to evidence score (range: 18.5%–52.1%) |
| Token Grounding (Visual-sourced) | 87.4% | Content words grounded when answers source from visual chunks (range: 82.1%–91.6%) |
| Top-Ranked Visual Chunk | 73.3% | Visual chunks prioritized correctly in 73.3% (11/15) of diagram queries |

---

## Architecture

```text
User Query
    │
    ▼
HyDE Generation (DeepSeek R1 1.5B)
    │
    ▼
┌─────────────────────────────────┐
│         Hybrid Retrieval        │
│  BM25 Sparse ──┐                │
│                ├── RRF Fusion   │
│  FAISS Dense ──┘                │
└─────────────────────────────────┘
    │
    ▼
Cross-Encoder Reranker (MS-MARCO)
    │
    ▼
Strict Grounding Prompt → DeepSeek R1 8B
    │
    ▼
┌─────────────────────────────────┐
│        Explainability Layer     │
│  • Evidence Attribution         │
│  • Token Grounding Heatmap      │
│  • Citation Validity Check      │
│  • Source Consensus Detection   │
└─────────────────────────────────┘
    │
    ▼
Grounded Answer + Citations + Metrics
```

---

## Key Features

### Retrieval
* **Hybrid BM25 + FAISS** with Reciprocal Rank Fusion
* **HyDE (Hypothetical Document Embeddings)** for complex queries
* **MS-MARCO cross-encoder reranking** for precision

### Generation
* **Strict grounding prompt** — model instructed to only use retrieved context
* **DeepSeek R1 8B via Ollama** (fully local, no API keys)
* **Automatic DeepSeek think-tag cleaning**

### Explainability
* **Evidence attribution** — cross-encoder scores which chunks contributed most
* **Token grounding heatmap** — highlights grounded vs ungrounded words
* **Source consensus detection** — flags when retrieved sources contradict
* **Citation validity** — verifies all [1], [2] references point to real chunks

### Multimodal
* **LLaVA-Phi3 vision model** describes diagrams and figures — achieves 0.967 visual MRR on diagram-specific queries
* **Perceptual hashing** to deduplicate similar images
* **Visual content made searchable** alongside text — visual chunks contribute avg 34.2% to evidence attribution
* **100% Recall@5 on visual queries** — all target diagrams retrieved within top 5 results

### Evaluation
* **Automated ground truth generation** using LLM-as-teacher
* **BERTScore, ROUGE-L, MRR, Recall@K, Faithfulness, Token Grounding**
* **Full benchmark report** saved to JSON

---

## Tech Stack

`Python` `PyTorch` `HuggingFace Transformers` `LangChain` `FAISS` `BM25` `Streamlit` `Docling` `Ollama` `DeepSeek R1` `BGE Embeddings` `MS-MARCO Cross-Encoder` `NLTK`

---

## Installation

**Prerequisites**
* Python 3.10+
* Ollama installed and running
* 16GB RAM recommended

```bash
git clone https://github.com/Amalsreekumar1/Explainable-RAG-For-educational-Q-A
cd Explainable-RAG-For-educational-Q-A
pip install -r requirements.txt
```

**Pull required Ollama models**
```bash
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:1.5b
ollama pull llava-phi3
```

---

## Usage

**1. Build your knowledge base**
```bash
python ingestion.py
```
*(Or use the Upload Sources tab in the web interface.)*

**2. Start the web app**
```bash
streamlit run app.py
```

**3. Generate evaluation dataset**
```bash
python generate_ground_truth.py
```

**4. Run benchmark evaluation**
```bash
python evaluation.py
```

**5. CLI mode**
```bash
python main_rag.py
```

---

## Project Structure

```text
EduRAG/
├── app.py                        # Streamlit web interface
├── main_rag_withoutShap.py       # Core RAG pipeline
├── explainability.py             # Citation grounding metrics
├── evaluation.py                 # Benchmark evaluation
├── ingestion_pdf.py              # Document ingestion (PDF/DOCX/PPTX)
├── generate_ground_truth.py      # Automated QA pair generation
├── requirements.txt
└── README.md
```

---

## Privacy

All processing is fully local. No data leaves your machine. No API keys required.

---

## Acknowledgements

* **Docling** — document parsing
* **Ollama** — local LLM inference
* **LangChain** — RAG framework
* **Streamlit** — web interface
* **BAAI/bge-small-en-v1.5** — dense embeddings
* **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranking
