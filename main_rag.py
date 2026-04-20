#!/usr/bin/env python3
"""
main_rag_withoutShap.py
Enhanced Educational RAG with Always-On Attribution & Knowledge Map CLI
- Hybrid Retrieval (BM25 + Dense + HyDE always enabled)
- Strict Grounding QA Prompt
- Evidence Attribution (cross-encoder)
- Token Attribution (colored CLI heatmap)
- Citation Grounding Metrics
- Source Consensus Detection
- CLI Knowledge Map (Document Explorer)

"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Self
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
import subprocess
import platform

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, ConfigDict

# NLTK
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Evaluation imports
import evaluate

# Suppress warnings
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_INDEX_PATH = "faiss_index"

embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load evaluation metrics once (lazy)
bertscore_metric = None
rouge_metric = None

def get_bertscore():
    global bertscore_metric
    if bertscore_metric is None:
        bertscore_metric = evaluate.load("bertscore")
    return bertscore_metric

def get_rouge():
    global rouge_metric
    if rouge_metric is None:
        rouge_metric = evaluate.load("rouge")
    return rouge_metric

PREFERRED_LLM = "deepseek-r1:8b"
FALLBACK_LLM = "deepseek-r1:1.5b"

# ==================== CROSS-ENCODER RERANKER ====================
class DocumentReranker:
    """Ensures top-5 retrieved docs are actually relevant via cross-encoder scoring."""
    def __init__(self, model_name: str = RERANK_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512)
            scores = self.model(**inputs).logits.squeeze(-1)
            
            # BOOST visual content scores
            boosted_scores = []
            for i, score in enumerate(scores.tolist()):
                if docs[i].metadata.get('type') == 'visual_content':
                    score += 0.15  # Boost visual content
                boosted_scores.append(score)
            
            scored = sorted(zip(boosted_scores, docs), key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_n]]

# ==================== STRICT GROUNDING PROMPT ====================
STRICT_QA_PROMPT = PromptTemplate.from_template(
    """Answer the question using ONLY the provided context.

## RULES:
1. Use ONLY words from the context below. Do not use external knowledge.
2. Cite every claim as [1], [2], etc. immediately after the claim.
3. If information is not in the context, say "Not found in provided materials."
4. Prefer short, exact phrases from the text.

## CONTEXT:
{context}

## QUESTION:
{question}

## ANSWER (using only context words):"""
)

# ==================== CSV Loader ====================
class CSVDocLoader:
    @classmethod
    def load(cls, csv_path: str) -> List[Document]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
        df.columns = df.columns.str.strip()

        required = ["text", "source_file", "page_number", "chunk_id"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column. Found: {list(df.columns)}")

        docs = []
        for idx, row in df.iterrows():
            content = str(row.get("text", "")).strip()
            if not content:
                continue

            metadata = {
                "source_file": str(row.get("source_file", "")).strip() or "Unknown",
                "page_number": str(row.get("page_number", "")).strip() or "N/A",
                "chunk_id": str(row.get("chunk_id", "")).strip() or f"row-{idx}",
                "chunk_length": row.get("chunk_length", None),
                "section_path": str(row.get("section_path", "General")),
                "type": str(row.get("type", "text")),
                "image_ref": str(row.get("image_ref", "")) or None
            }
            docs.append(Document(page_content=content, metadata=metadata))

        logger.info(f"Loaded {len(docs)} documents from {csv_path}")
        return docs


# ==================== Tokenizer & Fusion ====================
def simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"\b\w+\b", text)

def rrf_fusion(bm25_docs: List[Document], faiss_docs: List[Document], k: int = 60) -> List[Document]:
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def cid(d: Document) -> str:
        return str(d.metadata.get("chunk_id", hash(d.page_content)))

    for rank, d in enumerate(bm25_docs, start=1):
        c = cid(d)
        doc_map.setdefault(c, d)
        score_map[c] = score_map.get(c, 0.0) + 1.0 / (k + rank)

    for rank, d in enumerate(faiss_docs, start=1):
        c = cid(d)
        doc_map.setdefault(c, d)
        score_map[c] = score_map.get(c, 0.0) + 1.0 / (k + rank)

    sorted_cids = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[c] for c, _ in sorted_cids]


# ==================== HyDE ====================
HYDE_PROMPT = PromptTemplate.from_template(
    """Write a detailed passage that answers the following question.
Include specific details and concepts that would be found in an educational textbook.

Question: {question}

Passage:"""
)

def generate_hypothetical_document(query: str, llm) -> str:
    """Generate a hypothetical answer to use for embedding retrieval."""
    try:
        hyde_text = llm.invoke(HYDE_PROMPT.format(question=query))
        # FIX: Correct DeepSeek <think> tag stripping
        if "</think>" in hyde_text:
            hyde_text = hyde_text.split("</think>")[-1].strip()
        return hyde_text.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, using original query")
        return query


# ==================== Source Consensus Detection ====================
def compute_source_consensus(context_docs: List[Document]) -> Dict[str, Any]:
    """Detects if retrieved sources agree or contradict each other."""
    if len(context_docs) < 2:
        return {
            "consensus_score": 1.0,
            "conflicts": [],
            "status": "SINGLE_SOURCE",
            "message": "Only one source retrieved."
        }

    try:
        doc_texts = [d.page_content for d in context_docs]
        doc_embs = np.array(embedder.embed_documents(doc_texts))

        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_embs = doc_embs / norms

        sim_matrix = np.dot(doc_embs, doc_embs.T)

        n = len(context_docs)
        pairwise_sims = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]

        avg_sim = np.mean(pairwise_sims) if pairwise_sims else 1.0
        min_sim = np.min(pairwise_sims) if pairwise_sims else 1.0

        conflicts = []
        conflict_threshold = 0.5

        if min_sim < conflict_threshold:
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i][j] < conflict_threshold:
                        conflicts.append({
                            "doc_a": {
                                "id": context_docs[i].metadata.get("chunk_id"),
                                "source": context_docs[i].metadata.get("source_file"),
                                "page": context_docs[i].metadata.get("page_number")
                            },
                            "doc_b": {
                                "id": context_docs[j].metadata.get("chunk_id"),
                                "source": context_docs[j].metadata.get("source_file"),
                                "page": context_docs[j].metadata.get("page_number")
                            },
                            "similarity": float(sim_matrix[i][j]),
                            "issue": "LOW_SIMILARITY"
                        })

        if conflicts:
            status = "CONFLICT_DETECTED"
            message = f"⚠️ Sources may contradict (min similarity: {min_sim:.2f}). Verify claims."
        elif avg_sim > 0.8:
            status = "HIGH_CONSENSUS"
            message = "✅ Sources agree closely on topic."
        else:
            status = "MODERATE_VARIANCE"
            message = "ℹ️ Sources cover different aspects of the topic."

        return {
            "consensus_score": float(avg_sim),
            "min_similarity": float(min_sim),
            "conflicts": conflicts,
            "status": status,
            "message": message
        }

    except Exception as e:
        logger.warning(f"Consensus calculation failed: {e}")
        return {
            "consensus_score": 1.0,
            "conflicts": [],
            "status": "ERROR",
            "message": "Could not analyze source consensus."
        }


# ==================== EXPLAINABILITY: Evidence Attribution ====================
class EvidenceAttributor:
    """Cross-encoder attribution: which chunks contributed most to the answer."""
    def __init__(self, model_name: str = RERANK_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def attribute(self, query: str, answer: str, documents: List[Document]) -> Dict:
        if not documents:
            return {"attributions": [], "primary_source": None}

        attributions = []

        with torch.no_grad():
            for i, doc in enumerate(documents):
                pairs = [[f"{query} {answer}", doc.page_content]]
                inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                        return_tensors="pt", max_length=512)
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits).item()

                answer_tokens = set(answer.lower().split())
                doc_tokens = set(doc.page_content.lower().split())
                overlap = len(answer_tokens & doc_tokens) / len(answer_tokens) if answer_tokens else 0

                combined = 0.7 * score + 0.3 * overlap

                attributions.append({
                    "chunk_id": doc.metadata.get("chunk_id", f"doc_{i}"),
                    "source": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "relevance_score": float(score),
                    "token_overlap": float(overlap),
                    "combined_score": float(combined),
                    "contribution_pct": 0.0
                })

        total = sum(a["combined_score"] for a in attributions) or 1.0
        for a in attributions:
            a["contribution_pct"] = (a["combined_score"] / total) * 100

        attributions.sort(key=lambda x: x["combined_score"], reverse=True)

        return {
            "attributions": attributions,
            "primary_source": attributions[0]["source"] if attributions else None,
            "method": "cross_encoder"
        }


# ==================== EXPLAINABILITY: Token Attribution ====================
class TokenAttributor:
    """
    Token-level grounding: measures how much of the answer is supported by retrieved context.
    FIX: grounding_ratio is calculated on content words only (stop words excluded)
    to avoid inflated scores from trivially matched function words.
    """

    def __init__(self):
        try:
            self.stemmer = PorterStemmer()
            self.use_stemming = True
        except:
            self.use_stemming = False

    def attribute(self, answer: str, context_docs: List[Document]) -> Dict:
        if not answer or not context_docs:
            return {"tokens": [], "grounding_ratio": 0.0}

        context_text = " ".join([d.page_content for d in context_docs]).lower()
        context_tokens = set(context_text.split())

        if self.use_stemming:
            context_stems = set(self.stemmer.stem(t) for t in context_tokens)

        tokens = re.findall(r'\w+|[^\w\s]', answer)

        heatmap = []
        for token in tokens:
            token_lower = token.lower()
            token_stem = self.stemmer.stem(token_lower) if self.use_stemming else token_lower

            if token_lower in context_tokens:
                score = 1.0          # Exact match
            elif self.use_stemming and token_stem in context_stems:
                score = 0.9          # Stem match (e.g. "running" vs "run")
            elif len(token) > 3 and token_lower in context_text:
                score = 0.8          # Substring match
            else:
                score = 0.2          # Likely hallucinated

            heatmap.append({
                "token": token,
                "grounded_score": score,
                "is_grounded": score > 0.7
            })

        # FIX: Only count content words for grounding_ratio
        # Stop words, punctuation, and short tokens match trivially and inflate the score
        meaningful = [
            h for h in heatmap
            if h['token'].lower() not in STOP_WORDS
            and h['token'].isalpha()
            and len(h['token']) > 2
        ]

        grounding_ratio = (
            sum(1 for h in meaningful if h['is_grounded']) / len(meaningful)
            if meaningful else 0.0
        )

        return {
            "tokens": heatmap,  # Full list kept for visualization
            "grounding_ratio": grounding_ratio,  # Content words only — academically honest
            "ungrounded_tokens": [
                h["token"] for h in heatmap
                if not h["is_grounded"]
                and h["token"].lower() not in STOP_WORDS
                and h["token"].isalpha()
            ][:15]
        }


# ==================== Citation Metrics ====================
class CitationMetrics:
    """Calculate citation validity and precision/recall."""

    def evaluate(self, answer: str, context_docs: List[Document],
                 ground_truth_citations: Optional[List[int]] = None) -> Dict:
        citations_found = re.findall(r'\[(\d+)\]', answer)
        cited_indices = [int(c) - 1 for c in citations_found]

        metrics = {
            "citation_count": len(citations_found),
            "unique_citations": len(set(citations_found)),
        }

        valid_citations = []
        invalid_citations = []

        for idx in cited_indices:
            if 0 <= idx < len(context_docs):
                valid_citations.append(idx)
            else:
                invalid_citations.append(idx + 1)

        # FIX: Return None when no citations present so evaluator can exclude from average
        metrics["valid_citation_rate"] = (
            len(valid_citations) / len(cited_indices)
            if cited_indices else None
        )
        metrics["invalid_citations"] = invalid_citations

        if ground_truth_citations:
            gt_set = set([c - 1 for c in ground_truth_citations])
            cited_set = set(valid_citations)

            tp = len(cited_set & gt_set)
            fp = len(cited_set - gt_set)
            fn = len(gt_set - cited_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics.update({
                "citation_precision": precision,
                "citation_recall": recall,
                "citation_f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            })

        return metrics


# ==================== Generation Metrics ====================
class GenerationMetrics:
    @staticmethod
    def exact_match(pred: str, ref: str) -> float:
        return float(pred.strip().lower() == ref.strip().lower())

    @staticmethod
    def f1_score(pred: str, ref: str) -> float:
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def rouge_l(pred: str, ref: str) -> float:
        try:
            result = get_rouge().compute(predictions=[pred], references=[ref])
            return result['rougeL']
        except:
            return 0.0

    @staticmethod
    def bertscore(pred: str, ref: str) -> float:
        try:
            result = get_bertscore().compute(predictions=[pred], references=[ref], lang="en")
            return float(np.mean(result['f1']))
        except:
            return 0.0


# ==================== Faithfulness ====================
def compute_faithfulness(answer: str, context_docs: List[Document]) -> float:
    """
    FIX: Score answer against each chunk individually and take max.
    Previous approach concatenated all chunks into one string,
    causing silent BERTScore truncation at 512 tokens.
    Taking max is semantically correct — answer is faithful if it
    matches at least one retrieved chunk well.
    """
    if not context_docs:
        return 0.0
    try:
        scores = []
        for doc in context_docs:
            score = GenerationMetrics.bertscore(answer, doc.page_content[:500])
            scores.append(score)
        return float(np.max(scores))
    except Exception as e:
        logger.warning(f"Faithfulness failed: {e}")
        return 0.0


# ==================== Hybrid Retriever ====================
class HybridRetriever(BaseRetriever, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    documents: List[Document]
    bm25: Any = None
    faiss: Any = None
    reranker: Any = None
    top_k: int = 5
    enable_hyde: bool = True   # FIX: field added so HyDE is actually used
    hyde_llm: Any = None       # FIX: field added to hold the HyDE language model

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tokenized = [d.page_content.lower().split() for d in self.documents]
        object.__setattr__(self, 'bm25', BM25Okapi(tokenized))
        object.__setattr__(self, 'reranker', DocumentReranker(RERANK_MODEL))

        if os.path.exists(FAISS_INDEX_PATH):
            faiss_obj = FAISS.load_local(FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
            object.__setattr__(self, 'faiss', faiss_obj)
        else:
            faiss_obj = FAISS.from_documents(self.documents, embedder)
            faiss_obj.save_local(FAISS_INDEX_PATH)
            object.__setattr__(self, 'faiss', faiss_obj)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        fetch_k = 50
        
        # Expand query for better visual matching
        visual_expansion = " diagram chart figure illustration visual graph table image"
        expanded_query = query + visual_expansion
        
        # BM25 uses expanded query
        bm25_scores = self.bm25.get_scores(expanded_query.lower().split())
        bm25_idx = np.argsort(bm25_scores)[-fetch_k:][::-1]
        bm25_docs = [self.documents[i] for i in bm25_idx if bm25_scores[i] > 0]
        
        # HyDE for dense retrieval
        if self.enable_hyde and self.hyde_llm is not None:
            hyde_query = generate_hypothetical_document(query, self.hyde_llm)
        else:
            hyde_query = query
        
        faiss_docs = self.faiss.similarity_search(hyde_query, k=fetch_k)
        
        fused = rrf_fusion(bm25_docs, faiss_docs)
        final_docs = self.reranker.rerank(query, fused, top_n=self.top_k)
        return final_docs



# ==================== Prompts & Utils ====================
DOCUMENT_PROMPT = (
    "[{idx}] Source: {source_file} | Section: {section_path} | Page: {page_number} | Chunk: {chunk_id} | Type: {type}\n{page_content}\n"
)

def combine_documents_for_prompt(docs: List[Document], max_chars: int = 2500) -> str:
    parts = []
    total = 0

    for idx, d in enumerate(docs, 1):
        content_type = d.metadata.get("type", "text")
        section = d.metadata.get("section_path", "Unknown")
        page = d.metadata.get("page_number", "N/A")
        source = d.metadata.get("source_file", "Unknown")

        visual_indicator = ""
        if content_type == "visual_content":
            visual_indicator = "[VISUAL CONTENT]\n"

        chunk = DOCUMENT_PROMPT.format(
            idx=idx,
            source_file=source,
            section_path=section,
            page_number=page,
            chunk_id=d.metadata.get("chunk_id", "N/A"),
            type=content_type.upper(),
            page_content=visual_indicator + d.page_content
        )

        parts.append(chunk)
        total += len(chunk)

        if total >= max_chars:
            logger.warning(f"Context truncated at {total} chars")
            break

    return "\n\n".join(parts)


def clean_llm_output(raw_text: str) -> str:
    """FIX: Correct DeepSeek <think> tag stripping using split instead of broken regex."""
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r'^(Answer|Response|Based on the context)[\s:]*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\(([0-9]+)\)', r'[\1]', text)
    return text


# ==================== RAG Response ====================
@dataclass
class RAGResponse:
    answer: str = ""
    context: List[Document] = field(default_factory=list)
    retrieval_score: float = 0.0
    faithfulness: float = 0.0
    consensus: Dict = field(default_factory=dict)
    evidence_attribution: Dict = field(default_factory=dict)
    token_attribution: Dict = field(default_factory=dict)
    citation_metrics: Dict = field(default_factory=dict)
    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    bertscore: float = 0.0


# ==================== Main Pipeline ====================
def run_rag_pipeline(
    query: str,
    retriever: HybridRetriever,
    ground_truth: Optional[str] = None,
    ground_truth_citations: Optional[List[int]] = None,
    llm=None,
    use_strict_prompt: bool = True,
    evidence_attributor=None,   # FIX: accept pre-initialized to avoid reloading per query
    token_attributor=None       # FIX: accept pre-initialized to avoid reloading per query
) -> RAGResponse:
    """Full RAG pipeline with always-on attribution."""
    response = RAGResponse()

    # 1. Retrieve
    context_docs = retriever._get_relevant_documents(query)
    response.context = context_docs

    # 2. Retrieval score
    if context_docs:
        try:
            q_emb = np.array(embedder.embed_query(query))
            doc_embs = np.array(embedder.embed_documents([d.page_content for d in context_docs]))
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(doc_embs, q_emb)
            response.retrieval_score = float(np.mean(sims))
        except Exception as e:
            logger.warning(f"Failed to calculate retrieval score: {e}")
            response.retrieval_score = 0.0
    else:
        response.retrieval_score = 0.0

    # 3. Source consensus
    response.consensus = compute_source_consensus(context_docs)

    # 4. Generate answer
    if llm is None:
        llm = OllamaLLM(model=PREFERRED_LLM, temperature=0.1)

    context_text = combine_documents_for_prompt(context_docs)

    if use_strict_prompt:
        filled_prompt = STRICT_QA_PROMPT.format(context=context_text, question=query)
    else:
        filled_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"

    raw_answer = llm.invoke(filled_prompt)
    response.answer = clean_llm_output(raw_answer)

    # 5. Faithfulness
    response.faithfulness = compute_faithfulness(response.answer, context_docs)

    # 6. Attribution (use pre-initialized if provided)
    ev_attr = evidence_attributor or EvidenceAttributor()
    response.evidence_attribution = ev_attr.attribute(query, response.answer, context_docs)

    tok_attr = token_attributor or TokenAttributor()
    response.token_attribution = tok_attr.attribute(response.answer, context_docs)

    # 7. Citation metrics
    cit_metrics = CitationMetrics()
    response.citation_metrics = cit_metrics.evaluate(response.answer, context_docs, ground_truth_citations)

    # 8. Generation metrics vs ground truth
    if ground_truth:
        response.exact_match = GenerationMetrics.exact_match(response.answer, ground_truth)
        response.f1 = GenerationMetrics.f1_score(response.answer, ground_truth)
        response.rouge_l = GenerationMetrics.rouge_l(response.answer, ground_truth)
        response.bertscore = GenerationMetrics.bertscore(response.answer, ground_truth)

    return response


# ==================== CLI Visual Helpers ====================
def print_colored_tokens(tokens_data: List[Dict], max_display: int = 50):
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    line = "   "
    count = 0

    for token_info in tokens_data[:max_display]:
        token = token_info['token']
        is_grounded = token_info['is_grounded']
        colored_token = f"{GREEN}{token}{RESET}" if is_grounded else f"{RED}{token}{RESET}"
        line += colored_token + " "
        count += 1

        if count % 10 == 0:
            print(line)
            line = "   "

    if line.strip():
        print(line)


# ==================== KNOWLEDGE MAP CLI ====================
def knowledge_map_cli(csv_path: str = "educational_knowledge_base.csv"):
    """CLI-based Knowledge Map — browse documents, sections, and chunks."""
    if not os.path.exists(csv_path):
        print(f"\n❌ Knowledge base not found: {csv_path}")
        return True

    try:
        df = pd.read_csv(csv_path)

        while True:
            print("\n" + "=" * 70)
            print("🔍 KNOWLEDGE MAP - Document Explorer")
            print("=" * 70)
            print("Commands: 'back' to return to QA, 'quit' to exit")
            print("-" * 70)

            sources = ["All Documents"] + df['source_file'].unique().tolist()

            print("\n📚 Available Documents:")
            for i, src in enumerate(sources, 0):
                doc_chunks = len(df[df['source_file'] == src]) if src != "All Documents" else len(df)
                print(f"  [{i}] {src[:50]}... ({doc_chunks} chunks)")

            try:
                choice = input("\nSelect document [number] or 'back': ").strip()
                if choice.lower() in ['back', 'b']:
                    return True
                if choice.lower() in ['quit', 'q']:
                    return False

                idx = int(choice)
                if idx < 0 or idx >= len(sources):
                    print("❌ Invalid selection")
                    continue

                selected_source = sources[idx]
            except ValueError:
                print("❌ Please enter a number")
                continue

            if selected_source != "All Documents":
                doc_df = df[df['source_file'] == selected_source]
                print(f"\n📄 Viewing: {selected_source}")
            else:
                doc_df = df
                print(f"\n📄 Viewing: All Documents ({len(doc_df)} total chunks)")

            sections = doc_df['section_path'].unique().tolist()
            if len(sections) > 1:
                print(f"\n📑 Found {len(sections)} sections:")
                for i, sec in enumerate(sections[:20], 1):
                    sec_count = len(doc_df[doc_df['section_path'] == sec])
                    print(f"  [{i}] {sec} ({sec_count} chunks)")

                try:
                    sec_choice = input("\nSelect section [number] or 'all': ").strip()
                    if sec_choice.lower() == 'all':
                        section_data = doc_df
                    else:
                        sec_idx = int(sec_choice) - 1
                        if 0 <= sec_idx < len(sections):
                            selected_section = sections[sec_idx]
                            section_data = doc_df[doc_df['section_path'] == selected_section]
                            print(f"\n📑 Section: {selected_section} ({len(section_data)} chunks)")
                        else:
                            section_data = doc_df
                except ValueError:
                    section_data = doc_df
            else:
                section_data = doc_df
                if len(sections) == 1:
                    print(f"\n📑 Single Section: {sections[0]}")

            print(f"\n📝 Content Preview:")
            print("-" * 70)

            visuals = section_data[section_data['type'] == 'visual_content']
            texts = section_data[section_data['type'] != 'visual_content']

            if len(visuals) > 0:
                print(f"\n🖼️ Visual Content ({len(visuals)} items):")
                for _, row in visuals.head(5).iterrows():
                    print(f"\n  • {row['chunk_id']}")
                    print(f"    Page: {row['page_number']} | Type: {row['type']}")
                    if pd.notna(row['image_ref']) and os.path.exists(row['image_ref']):
                        size_kb = os.path.getsize(row['image_ref']) / 1024
                        print(f"    Image: {row['image_ref']} ({size_kb:.1f} KB)")
                        print(f"    [Command: open {row['image_ref']}]")
                    else:
                        print(f"    Image: Not found")
                    desc = str(row['text'])[:80] if pd.notna(row['text']) else "No description"
                    print(f"    Desc: {desc}...")

            if len(texts) > 0:
                print(f"\n📄 Text Content ({len(texts)} chunks):")
                for i, (_, row) in enumerate(texts.head(10).iterrows(), 1):
                    print(f"\n  [{i}] {row['chunk_id']} (Page {row['page_number']})")
                    content = str(row['text'])[:150] if pd.notna(row['text']) else "No content"
                    print(f"  {content}...")
                    if len(str(row['text'])) > 150:
                        print(f"  ... [{len(str(row['text']))} chars total]")

            print("\n" + "-" * 70)
            action = input("Enter 'open <path>' to view image, 'back' to documents, 'quit': ").strip()

            if action.lower() in ['quit', 'q']:
                return False
            elif action.lower() == 'back':
                continue
            elif action.lower().startswith('open '):
                file_path = action[5:].strip()
                if os.path.exists(file_path):
                    print(f"📷 Opening {file_path}...")
                    if platform.system() == 'Windows':
                        os.startfile(file_path)
                    elif platform.system() == 'Darwin':
                        subprocess.run(['open', file_path])
                    else:
                        subprocess.run(['xdg-open', file_path])
                else:
                    print(f"❌ File not found: {file_path}")

    except Exception as e:
        print(f"\n❌ Error browsing knowledge map: {e}")
        import traceback
        traceback.print_exc()

    return True


# ==================== Factory Functions ====================
def get_hybrid_retriever_from_csv(
    csv_path: str,
    top_k: int = 5,
    enable_hyde: bool = True,
    hyde_llm=None
):
    docs = CSVDocLoader.load(csv_path)

    for doc in docs:
        if "section_path" not in doc.metadata:
            doc.metadata["section_path"] = "General"
        if "type" not in doc.metadata:
            doc.metadata["type"] = "text"
        if "image_ref" not in doc.metadata:
            doc.metadata["image_ref"] = None

    if hyde_llm is None and enable_hyde:
        hyde_llm = OllamaLLM(model=FALLBACK_LLM, temperature=0.1, max_tokens=256)

    return HybridRetriever(
        documents=docs,
        top_k=top_k,
        enable_hyde=enable_hyde,   # FIX: was not passed — HyDE was silently disabled
        hyde_llm=hyde_llm,         # FIX: was not passed — LLM was initialized but discarded
    )


def get_llm(preferred: str = PREFERRED_LLM, fallback: str = FALLBACK_LLM):
    try:
        return OllamaLLM(model=preferred, temperature=0.1, max_tokens=1024)
    except:
        return OllamaLLM(model=fallback, temperature=0.1, max_tokens=512)


# ==================== CLI Mode ====================
if __name__ == "__main__":
    CSV_PATH = "educational_knowledge_base.csv"

    retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5, enable_hyde=True)
    llm = get_llm()

    # FIX: Initialize attributors once, reuse across queries
    evidence_attributor = EvidenceAttributor()
    token_attributor = TokenAttributor()

    print("=" * 70)
    print(" Enhanced RAG: HyDE + Attribution + Knowledge Map (Always On)")
    print("=" * 70)
    print(" Commands:")
    print("   'map' or 'browse'  - Open Knowledge Map")
    print("   'eval <answer>'    - Evaluate against ground truth")
    print("   'quit' or 'exit'   - Exit")
    print("=" * 70)

    while True:
        try:
            query = input("\nQuery (or 'map' to browse, 'quit' to exit): ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if query.lower() in ['map', 'browse', 'kb']:
                should_continue = knowledge_map_cli(CSV_PATH)
                if should_continue is False:
                    break
                continue

            ground_truth = None
            if query.lower().startswith('eval '):
                ground_truth = query[5:].strip()
                query = input("Actual query: ").strip()
                if not query:
                    continue

            result = run_rag_pipeline(
                query, retriever,
                ground_truth=ground_truth,
                llm=llm,
                use_strict_prompt=True,
                evidence_attributor=evidence_attributor,
                token_attributor=token_attributor
            )

            print("\n" + "=" * 70)
            print(f"🧠 {result.consensus.get('message', 'Consensus: N/A')}")
            print("-" * 70)
            print(f"📘 ANSWER:\n{result.answer}")

            print(f"\n📊 EVIDENCE ATTRIBUTION:")
            if result.evidence_attribution.get('attributions'):
                for idx, attr in enumerate(result.evidence_attribution['attributions'][:3], 1):
                    src_display = attr['source'][:40] + "..." if len(attr['source']) > 40 else attr['source']
                    print(f"   {idx}. {src_display:<43} (p.{attr['page']}) - {attr['contribution_pct']:.1f}%")

            cm = result.citation_metrics
            cit_rate = cm.get('valid_citation_rate')
            cit_display = f"{cit_rate * 100:.0f}%" if cit_rate is not None else "N/A (no citations)"
            print(f"\n📚 CITATIONS: {cm.get('unique_citations', 0)} unique, {cit_display} valid")

            ta = result.token_attribution
            print(f"\n🔍 TOKEN GROUNDING ({ta.get('grounding_ratio', 0) * 100:.1f}% content words grounded):")
            print("   " + '\033[92m' + "Green=Grounded" + '\033[0m' + " | " + '\033[91m' + "Red=Ungrounded" + '\033[0m')
            if ta.get('tokens'):
                print_colored_tokens(ta['tokens'], max_display=50)

            if ta.get('ungrounded_tokens'):
                print(f"\n   ⚠️ Ungrounded: {', '.join(ta['ungrounded_tokens'][:8])}")

            visual_docs = [d for d in result.context if d.metadata.get("type") == "visual_content"]
            if visual_docs:
                print(f"\n🖼️ VISUAL EVIDENCE ({len(visual_docs)} image(s)):")
                for doc in visual_docs[:3]:
                    img_path = doc.metadata.get("image_ref", "N/A")
                    print(f"   • Page {doc.metadata.get('page_number')}: {img_path}")

            print(f"\n📊 METRICS: Retrieval={result.retrieval_score:.3f} | Faithfulness={result.faithfulness:.3f} | Consensus={result.consensus.get('consensus_score', 0):.3f}")

            if ground_truth:
                print(f"\n🎯 EVAL: EM={result.exact_match:.3f} | F1={result.f1:.3f} | ROUGE-L={result.rouge_l:.3f}")

            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()