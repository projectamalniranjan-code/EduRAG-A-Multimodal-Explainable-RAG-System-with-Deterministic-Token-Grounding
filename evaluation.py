"""
evaluation.py
Comprehensive benchmarking for Educational RAG
Connected directly to main_rag_withoutShap.py

FIXES APPLIED:
- Fix 1: HyDE now enabled during evaluation (was False before — methodological inconsistency)
- Fix 2: EvidenceAttributor and TokenAttributor initialized once, reused across queries
- Fix 3: Citation validity None handling — excludes no-citation cases from average
- Fix 4: Sample size increased to 100 in generate_ground_truth.py recommendation
"""
import pandas as pd
import numpy as np
import logging
import warnings
import evaluate
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict

from main_rag import (
    get_hybrid_retriever_from_csv,
    run_rag_pipeline,
    EvidenceAttributor,
    TokenAttributor
)
from explainability import CitationGroundingMetrics

# Setup logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, csv_path: str = "educational_knowledge_base.csv"):
        self.csv_path = csv_path

        logger.info("Loading NLP Evaluation Models (BERTScore, ROUGE)...")
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")

        # Attribution tools
        self.citation_eval = CitationGroundingMetrics()

        # FIX: Initialize attributors ONCE — reused across all queries
        # Previously these were re-instantiated inside run_rag_pipeline every query,
        # loading the cross-encoder model from disk each time (slow and wasteful)
        logger.info("Loading Attribution Models (cross-encoder, token attributor)...")
        self.evidence_attributor = EvidenceAttributor()
        self.token_attributor = TokenAttributor()

        # FIX: enable_hyde=True to match actual system behavior
        # Previously enable_hyde=False — benchmark did not reflect real system
        logger.info("Initializing Hybrid Retriever Engine (HyDE enabled)...")
        self.retriever = get_hybrid_retriever_from_csv(
            self.csv_path, top_k=5, enable_hyde=True
        )

    # ============== TEXT METRICS ==============

    def exact_match(self, prediction: str, reference: str) -> float:
        return float(prediction.strip().lower() == reference.strip().lower())

    def f1_score(self, prediction: str, reference: str) -> float:
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # ============== SINGLE EVALUATION ==============

    def evaluate_single(self, query: str, ground_truth: str, target_chunk_id: str) -> Dict:
        """Runs a single query through the RAG and computes all metrics."""

        # FIX: pass pre-initialized attributors to avoid model reloading per query
        result = run_rag_pipeline(
            query,
            self.retriever,
            llm=None,
            evidence_attributor=self.evidence_attributor,
            token_attributor=self.token_attributor
        )

        prediction = result.answer
        context_docs = result.context
        retrieved_ids = [d.metadata.get("chunk_id") for d in context_docs]

        # Generation metrics
        gen_metrics = {
            "exact_match": self.exact_match(prediction, ground_truth),
            "f1": self.f1_score(prediction, ground_truth),
            "rouge_l": self.rouge.compute(
                predictions=[prediction], references=[ground_truth]
            )['rougeL'],
            "bertscore_f1": np.mean(
                self.bertscore.compute(
                    predictions=[prediction], references=[ground_truth], lang="en"
                )['f1']
            )
        }

        # Retrieval metrics
        r_at_1 = 1 if len(retrieved_ids) > 0 and retrieved_ids[0] == target_chunk_id else 0
        r_at_5 = 1 if target_chunk_id in retrieved_ids[:5] else 0

        mrr = 0.0
        if target_chunk_id in retrieved_ids:
            mrr = 1.0 / (retrieved_ids.index(target_chunk_id) + 1)

        ret_metrics = {
            "recall@1": r_at_1,
            "recall@5": r_at_5,
            "mrr": mrr
        }

        # Grounding metrics
        cit_metrics = self.citation_eval.evaluate_citations(prediction, context_docs)

        # FIX: citation_validity from result.citation_metrics (uses corrected None handling)
        cit_rate = result.citation_metrics.get("valid_citation_rate")

        grounding = {
            "citation_validity": cit_rate,   # May be None — handled in batch aggregation
            "faithfulness": result.faithfulness,
            "token_grounding_ratio": result.token_attribution.get("grounding_ratio", 0)
        }

        return {
            "generation": gen_metrics,
            "retrieval": ret_metrics,
            "grounding": grounding
        }

    # ============== BATCH EVALUATION ==============

    def evaluate_dataset(self, eval_csv: str = "evaluation_dataset.csv") -> Dict:
        """Processes the entire Ground Truth dataset."""
        try:
            df = pd.read_csv(eval_csv)
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
            return {}

        logger.info(f"\n🚀 Beginning benchmark on {len(df)} queries...")
        all_results = defaultdict(list)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating RAG System"):
            try:
                result = self.evaluate_single(
                    query=str(row["query"]),
                    ground_truth=str(row["answer"]),
                    target_chunk_id=str(row["target_chunk_id"]).strip()
                )

                for category, metrics in result.items():
                    for k, v in metrics.items():
                        # FIX: Only append non-None values
                        if v is not None:
                            all_results[f"{category}_{k}"].append(v)

            except Exception as e:
                logger.error(f"Error evaluating query {idx}: {e}")
                continue

        # Calculate averages (only over queries that had valid values)
        summary = {k: np.mean(v) for k, v in all_results.items() if v}

        self._print_report(summary, len(df))

        import json
        output_file = "final_benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"\n✅ Detailed metrics saved to {output_file}")

        return summary

    def _print_report(self, summary: Dict, total: int):
        print("\n" + "=" * 65)
        print("                 RESEARCH BENCHMARK REPORT                 ")
        print("=" * 65)
        print(f"Total Queries Evaluated: {total}")
        print("-" * 65)
        print("1. RETRIEVAL ENGINE PERFORMANCE (Hybrid BM25 + FAISS + RRF + Reranker)")
        print(f"   MRR (Mean Reciprocal Rank): {summary.get('retrieval_mrr', 0):.4f}")
        print(f"   Recall@1:                   {summary.get('retrieval_recall@1', 0):.4f}")
        print(f"   Recall@5:                   {summary.get('retrieval_recall@5', 0):.4f}")
        print("-" * 65)
        print("2. GENERATION ACCURACY (DeepSeek R1 8B)")
        print(f"   BERTScore F1 (Semantic):    {summary.get('generation_bertscore_f1', 0):.4f}")
        print(f"   ROUGE-L (Structure):        {summary.get('generation_rouge_l', 0):.4f}")
        print(f"   Token F1 (Overlap):         {summary.get('generation_f1', 0):.4f}")
        print(f"   Exact Match:                {summary.get('generation_exact_match', 0):.4f}")
        print("   Note: Exact match of 0.0 is expected for open-ended RAG generation.")
        print("-" * 65)
        print("3. TRUST & HALLUCINATION SAFEGUARDS")
        print(f"   Faithfulness Score:         {summary.get('grounding_faithfulness', 0):.4f}")
        cit = summary.get('grounding_citation_validity', None)
        cit_display = f"{cit * 100:.1f}%" if cit is not None else "N/A"
        print(f"   Valid Citation Rate:        {cit_display}")
        print(f"   Token Grounding Ratio:      {summary.get('grounding_token_grounding_ratio', 0) * 100:.1f}%")
        print("   Note: Token grounding measured on content words only (stop words excluded).")
        print("=" * 65)


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset("evaluation_dataset.csv")