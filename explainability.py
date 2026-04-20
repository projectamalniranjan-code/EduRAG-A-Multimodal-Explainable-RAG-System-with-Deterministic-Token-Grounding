"""
explainability.py
Attribution and grounding metrics for Educational RAG.

FIXES APPLIED:
- Fix 1: Duplicate EvidenceAttributor and TokenAttributor classes removed.
  Both now imported from main_rag_withoutShap.py to ensure evaluation
  and pipeline use identical logic (previously they differed silently).
- Fix 2: CitationGroundingMetrics._verify_citation_support placeholder
  now uses token overlap instead of hardcoded 1.0.
- Fix 3: valid_citation_rate returns None when no citations present.
"""

import numpy as np
import re
from typing import List, Dict, Optional
from langchain_core.documents import Document

# FIX: Import from main_rag_withoutShap instead of redefining
# This ensures evaluation uses the same logic as the pipeline
from main_rag import EvidenceAttributor, TokenAttributor


class CitationGroundingMetrics:
    """
    Citation validity and precision/recall metrics.
    Used by evaluation.py to cross-check citation quality.
    """

    def evaluate_citations(self, answer: str, context_docs: List[Document],
                           ground_truth_citations: Optional[List[int]] = None) -> Dict:
        citations_found = re.findall(r'\[(\d+)\]', answer)
        cited_indices = [int(c) - 1 for c in citations_found]

        metrics = {
            "citation_count": len(citations_found),
            "unique_citations": len(set(citations_found)),
            "citations": citations_found
        }

        valid_citations = []
        invalid_citations = []

        for idx in cited_indices:
            if 0 <= idx < len(context_docs):
                valid_citations.append(idx)
            else:
                invalid_citations.append(idx + 1)

        # FIX: Return None when no citations — don't count as 0 in average
        metrics["valid_citation_rate"] = (
            len(valid_citations) / len(cited_indices)
            if cited_indices else None
        )
        metrics["invalid_citations"] = invalid_citations

        if ground_truth_citations is not None:
            cited_set = set(cited_indices)
            gt_set = set([c - 1 for c in ground_truth_citations])

            true_positives = len(cited_set & gt_set)
            false_positives = len(cited_set - gt_set)
            false_negatives = len(gt_set - cited_set)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics.update({
                "citation_precision": precision,
                "citation_recall": recall,
                "citation_f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            })
        else:
            metrics.update(self._verify_citation_support(answer, context_docs, cited_indices))

        return metrics

    def _verify_citation_support(self, answer: str, context_docs: List[Document],
                                  cited_indices: List[int]) -> Dict:
        """
        Verify if cited documents support the claims via token overlap.
        NOTE: A production implementation would use an NLI model
        (e.g., cross-encoder/nli-deberta-v3-small) for entailment checking.
        FIX: Returns real overlap score instead of hardcoded 1.0 placeholder.
        """
        support_scores = []

        sentences = re.split(r'(?<=[.!?])\s+', answer)
        for sent in sentences:
            cited_in_sent = re.findall(r'\[(\d+)\]', sent)
            if not cited_in_sent:
                continue

            sent_text = re.sub(r'\[\d+\]', '', sent).strip()
            if len(sent_text) < 10:
                continue

            for c_str in cited_in_sent:
                idx = int(c_str) - 1
                if 0 <= idx < len(context_docs):
                    sent_tokens = set(sent_text.lower().split())
                    doc_tokens = set(context_docs[idx].page_content[:500].lower().split())
                    overlap = len(sent_tokens & doc_tokens) / len(sent_tokens) if sent_tokens else 0
                    support_scores.append(overlap)

        avg_support = np.mean(support_scores) if support_scores else 0.0
        return {
            "citation_support_score": avg_support,
            "unsupported_claims": sum(1 for s in support_scores if s < 0.3)
        }


def get_attributors(embedding_model=None):
    """Get configured attributors."""
    return {
        "evidence": EvidenceAttributor(),
        "token": TokenAttributor(),
        "citation": CitationGroundingMetrics()
    }