"""
ablation_eval.py
Run this to generate ablation study results for the paper.
Produces three configurations:
  1. BM25 only (no FAISS, no reranker, no HyDE)
  2. Hybrid BM25+FAISS with RRF (no reranker, no HyDE)
  3. Full system — already in final_benchmark_results.json

Usage:
    python ablation_eval.py
"""

import pandas as pd
import numpy as np
import logging
import warnings
import evaluate
import json
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

# Suppress noise
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==================== Minimal BM25-only retriever ====================
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import pandas as pd_inner

def load_docs(csv_path="educational_knowledge_base.csv"):
    df = pd_inner.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
    docs = []
    for idx, row in df.iterrows():
        content = str(row.get("text","")).strip()
        if not content:
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "chunk_id": row.get("chunk_id", f"row-{idx}"),
                "source_file": row.get("source_file", "Unknown"),
                "page_number": row.get("page_number", "N/A"),
                "section_path": row.get("section_path", "General"),
                "type": row.get("type", "text"),
            }
        ))
    return docs

def bm25_retrieve(query, docs, bm25, top_k=5):
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[-top_k:][::-1]
    return [docs[i] for i in idx if scores[i] > 0]

def hybrid_retrieve_no_reranker(query, docs, bm25, faiss_store, top_k=5):
    from main_rag import rrf_fusion
    fetch_k = 20
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[-fetch_k:][::-1]
    bm25_docs = [docs[i] for i in idx if scores[i] > 0]
    faiss_docs = faiss_store.similarity_search(query, k=fetch_k)
    fused = rrf_fusion(bm25_docs, faiss_docs)
    return fused[:top_k]

# ==================== Metrics ====================
def f1_score(pred, ref):
    p_tok = set(pred.lower().split())
    r_tok = set(ref.lower().split())
    if not p_tok or not r_tok:
        return 0.0
    common = p_tok & r_tok
    prec = len(common)/len(p_tok)
    rec = len(common)/len(r_tok)
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def compute_faithfulness(answer, context_docs, bertscore_metric):
    if not context_docs: return 0.0
    scores = []
    for doc in context_docs:
        r = bertscore_metric.compute(
            predictions=[answer],
            references=[doc.page_content[:500]],
            lang="en"
        )
        scores.append(float(np.mean(r['f1'])))
    return float(np.max(scores)) if scores else 0.0

# ==================== Evaluate one config ====================
def evaluate_config(config_name, retriever_fn, eval_csv, bertscore_metric, rouge_metric, llm):
    try:
        df = pd.read_csv(eval_csv)
    except Exception as e:
        logger.error(f"Cannot load eval CSV: {e}")
        return {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Running config: {config_name} on {len(df)} queries")
    logger.info(f"{'='*60}")

    results = defaultdict(list)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=config_name):
        try:
            query = str(row["query"])
            ground_truth = str(row["answer"])
            target_id = str(row["target_chunk_id"]).strip()

            # Retrieve
            context_docs = retriever_fn(query)
            retrieved_ids = [d.metadata.get("chunk_id") for d in context_docs]

            # Generate
            from main_rag import combine_documents_for_prompt, STRICT_QA_PROMPT, clean_llm_output
            context_text = combine_documents_for_prompt(context_docs)
            filled_prompt = STRICT_QA_PROMPT.format(context=context_text, question=query)
            raw = llm.invoke(filled_prompt)
            answer = clean_llm_output(raw)

            # Retrieval metrics
            r1 = 1 if retrieved_ids and retrieved_ids[0] == target_id else 0
            r5 = 1 if target_id in retrieved_ids[:5] else 0
            mrr = 1.0/(retrieved_ids.index(target_id)+1) if target_id in retrieved_ids else 0.0

            # Generation metrics
            bs = bertscore_metric.compute(predictions=[answer], references=[ground_truth], lang="en")
            rouge = rouge_metric.compute(predictions=[answer], references=[ground_truth])

            # Faithfulness
            faith = compute_faithfulness(answer, context_docs, bertscore_metric)

            results["recall@1"].append(r1)
            results["recall@5"].append(r5)
            results["mrr"].append(mrr)
            results["bertscore_f1"].append(float(np.mean(bs['f1'])))
            results["rouge_l"].append(rouge['rougeL'])
            results["token_f1"].append(f1_score(answer, ground_truth))
            results["faithfulness"].append(faith)

        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            continue

    summary = {k: float(np.mean(v)) for k, v in results.items() if v}
    return summary

def evaluate_visual_retrieval(
    eval_csv="visual_evaluation_dataset.csv",
    csv_path="educational_knowledge_base.csv"
):
    """
    Evaluate retrieval performance specifically on visual queries.
    Tracks whether visual chunks are retrieved and their ranking.
    """
    
    logger.info("\n" + "="*60)
    logger.info("VISUAL RETRIEVAL EVALUATION")
    logger.info("="*60)
    
    if not os.path.exists(eval_csv):
        logger.error(f"Visual evaluation dataset not found: {eval_csv}")
        logger.info("Run: python generate_visual_ground_truth.py")
        return None
    
    # Load visual QA pairs
    visual_df = pd.read_csv(eval_csv)
    logger.info(f"Evaluating {len(visual_df)} visual queries")
    
    # Load full knowledge base to check chunk types
    kb_df = pd.read_csv(csv_path)
    chunk_type_map = dict(zip(kb_df['chunk_id'], kb_df['type']))
    
    # Load retriever (reuse existing)
    docs = load_docs(csv_path)
    tokenized = [d.page_content.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists("faiss_index"):
        faiss_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    else:
        faiss_store = FAISS.from_documents(docs, embedder)
    
    # Metrics to track
    results = {
        "visual_in_top5": 0,      # Visual chunk anywhere in top 5
        "visual_at_1": 0,         # Visual chunk ranked first
        "visual_at_2": 0,         # Visual chunk ranked second
        "visual_rank_sum": 0,     # For MRR calculation
        "visual_count": 0,        # Queries with visual chunks retrieved
        "total_queries": len(visual_df),
        "queries_with_visual_target": 0,
        "text_only_retrieval": 0   # No visual chunks in top 5
    }
    
    detailed_traces = []
    
    for idx, row in tqdm(visual_df.iterrows(), total=len(visual_df), desc="Visual eval"):
        query = str(row["query"])
        target_id = str(row["target_chunk_id"]).strip()
        target_type = chunk_type_map.get(target_id, "unknown")
        
        # Retrieve using hybrid method
        from main_rag import rrf_fusion
        
        # BM25
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_idx = np.argsort(bm25_scores)[-20:][::-1]
        bm25_docs = [docs[i] for i in bm25_idx if bm25_scores[i] > 0]
        
        # FAISS (no HyDE for fair comparison, or enable it)
        faiss_docs = faiss_store.similarity_search(query, k=20)
        
        # Fuse and rerank
        fused = rrf_fusion(bm25_docs, faiss_docs, k=60)
        
        # Rerank (simplified - use existing reranker if available)
        final_docs = fused[:5]
        
        # Analyze results
        retrieved_ids = [d.metadata.get("chunk_id") for d in final_docs]
        retrieved_types = [chunk_type_map.get(cid, "unknown") for cid in retrieved_ids]
        
        visual_positions = [i for i, t in enumerate(retrieved_types) if t == "visual_content"]
        
        trace = {
            "query": query[:80],
            "target_id": target_id,
            "target_type": target_type,
            "retrieved_ids": retrieved_ids,
            "retrieved_types": retrieved_types,
            "visual_positions": visual_positions,
            "target_in_results": target_id in retrieved_ids
        }
        detailed_traces.append(trace)
        
        # Update metrics
        if visual_positions:
            results["visual_count"] += 1
            results["visual_in_top5"] += 1
            best_pos = min(visual_positions) + 1  # 1-indexed
            results["visual_rank_sum"] += 1.0 / best_pos
            
            if best_pos == 1:
                results["visual_at_1"] += 1
            elif best_pos == 2:
                results["visual_at_2"] += 1
        else:
            results["text_only_retrieval"] += 1
        
        if target_type == "visual_content":
            results["queries_with_visual_target"] += 1
    
    # Calculate summary metrics
    n = results["total_queries"]
    
    summary = {
        "visual_recall@5": results["visual_in_top5"] / n,
        "visual_mrr": results["visual_rank_sum"] / n if n > 0 else 0,
        "visual_at_1_rate": results["visual_at_1"] / n,
        "visual_at_1_or_2_rate": (results["visual_at_1"] + results["visual_at_2"]) / n,
        "text_only_rate": results["text_only_retrieval"] / n,
        "queries_with_visual_target": results["queries_with_visual_target"],
        "avg_visuals_per_query": results["visual_count"] / n
    }
    
    # Print report
    print("\n" + "="*60)
    print("VISUAL RETRIEVAL RESULTS")
    print("="*60)
    print(f"Total visual queries: {n}")
    print(f"Queries targeting visual chunks: {results['queries_with_visual_target']}")
    print("-"*60)
    print(f"Visual Recall@5:      {summary['visual_recall@5']:.3f} ({results['visual_in_top5']}/{n})")
    print(f"Visual MRR:           {summary['visual_mrr']:.3f}")
    print(f"Visual at Rank 1:     {summary['visual_at_1_rate']:.3f} ({results['visual_at_1']}/{n})")
    print(f"Visual at Rank 1-2:   {summary['visual_at_1_or_2_rate']:.3f}")
    print(f"Text-only retrieval:  {summary['text_only_rate']:.3f} ({results['text_only_retrieval']}/{n})")
    print("="*60)
    
    # Save detailed results
    with open("visual_retrieval_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "raw_metrics": results,
            "traces": detailed_traces
        }, f, indent=2)
    
    logger.info("Detailed results saved to visual_retrieval_results.json")
    
    return summary

def print_config_results(name, results):
    print(f"\n--- {name} ---")
    print(f"  MRR:          {results.get('mrr', 0):.4f}")
    print(f"  Recall@1:     {results.get('recall@1', 0):.4f}")
    print(f"  Recall@5:     {results.get('recall@5', 0):.4f}")
    print(f"  BERTScore F1: {results.get('bertscore_f1', 0):.4f}")
    print(f"  ROUGE-L:      {results.get('rouge_l', 0):.4f}")
    print(f"  Token F1:     {results.get('token_f1', 0):.4f}")
    print(f"  Faithfulness: {results.get('faithfulness', 0):.4f}")

# ==================== Main ====================
if __name__ == "__main__":
    CSV_PATH = "educational_knowledge_base.csv"
    EVAL_CSV = "evaluation_dataset.csv"

    logger.info("Loading documents...")
    docs = load_docs(CSV_PATH)

    logger.info("Building BM25 index...")
    tokenized = [d.page_content.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    logger.info("Loading FAISS index...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    import os
    if os.path.exists("faiss_index"):
        faiss_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    else:
        logger.info("Building FAISS index from scratch (this takes a few minutes)...")
        faiss_store = FAISS.from_documents(docs, embedder)
        faiss_store.save_local("faiss_index")

    logger.info("Loading LLM (DeepSeek R1 8B)...")
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.1, max_tokens=1024)

    logger.info("Loading evaluation metrics...")
    bertscore_metric = evaluate.load("bertscore")
    rouge_metric = evaluate.load("rouge")

    all_results = {}

    # ---- Config 1: BM25 Only ----
    def bm25_retriever(query):
        return bm25_retrieve(query, docs, bm25, top_k=5)

    all_results["BM25 Only"] = evaluate_config(
        "BM25 Only", bm25_retriever, EVAL_CSV,
        bertscore_metric, rouge_metric, llm
    )
    print_config_results("BM25 Only", all_results["BM25 Only"])

    # ---- Config 2: Hybrid No Reranker ----
    def hybrid_retriever(query):
        return hybrid_retrieve_no_reranker(query, docs, bm25, faiss_store, top_k=5)

    all_results["Hybrid (No Reranker)"] = evaluate_config(
        "Hybrid (No Reranker)", hybrid_retriever, EVAL_CSV,
        bertscore_metric, rouge_metric, llm
    )
    print_config_results("Hybrid (No Reranker)", all_results["Hybrid (No Reranker)"])

    # ---- Config 3: Full (from existing results) ----
    try:
        with open("final_benchmark_results.json") as f:
            full_results = json.load(f)
        all_results["Full System"] = {
            "mrr": full_results.get("retrieval_mrr", 0),
            "recall@1": full_results.get("retrieval_recall@1", 0),
            "recall@5": full_results.get("retrieval_recall@5", 0),
            "bertscore_f1": full_results.get("generation_bertscore_f1", 0),
            "rouge_l": full_results.get("generation_rouge_l", 0),
            "token_f1": full_results.get("generation_f1", 0),
            "faithfulness": full_results.get("grounding_faithfulness", 0),
        }
        print_config_results("Full System (Hybrid + HyDE + Reranker)", all_results["Full System"])
    except:
        logger.warning("Could not load final_benchmark_results.json — run evaluation.py first")

    # ---- Save all ablation results ----
    with open("ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "="*60)
    print("ABLATION TABLE SUMMARY")
    print("="*60)
    print(f"{'Configuration':<30} {'MRR':>6} {'R@1':>6} {'R@5':>6} {'BERT':>6} {'Faith':>6}")
    print("-"*60)
    for name, res in all_results.items():
        print(f"{name:<30} {res.get('mrr',0):>6.4f} {res.get('recall@1',0):>6.4f} {res.get('recall@5',0):>6.4f} {res.get('bertscore_f1',0):>6.4f} {res.get('faithfulness',0):>6.4f}")
    print("="*60)
    print("\n✅ Results saved to ablation_results.json")