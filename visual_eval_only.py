"""
visual_eval_only.py
Standalone visual retrieval evaluation.
Assumes text evaluation (ablation_results.json) already exists.
Generates visual dataset and evaluates multimodal retrieval performance.

Usage:
    python visual_eval_only.py
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import re
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from main_rag import (
    rrf_fusion,
    DocumentReranker,
    generate_hypothetical_document,
    combine_documents_for_prompt,
    STRICT_QA_PROMPT,
    clean_llm_output,
    compute_faithfulness,
    EvidenceAttributor,
    TokenAttributor,
    CitationMetrics
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ==================== Visual Dataset Generation ====================

def generate_visual_dataset(
    kb_path: str = "educational_knowledge_base.csv",
    output_path: str = "visual_evaluation_dataset.csv",
    num_questions: int = 15,
    model_name: str = "deepseek-r1:8b"
) -> Optional[pd.DataFrame]:
    """Generate QA pairs from VLM-captioned visual chunks."""
    
    logger.info("=" * 70)
    logger.info("GENERATING VISUAL EVALUATION DATASET")
    logger.info("=" * 70)
    
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        return None
    
    df = pd.read_csv(kb_path, dtype=str, keep_default_na=False)
    
    # Filter for visual chunks with meaningful captions
    visual_chunks = df[
        (df['type'] == 'visual_content') &
        (df['text'].str.len() > 80) &
        (df['text'].str.len() < 800) &
        (~df['text'].str.contains('logo|decorative|banner|header', case=False, na=False)) &
        (df['image_ref'].notna()) &
        (df['image_ref'] != '')
    ].copy()
    
    if len(visual_chunks) == 0:
        logger.error("No valid visual chunks found")
        return None
    
    logger.info(f"Found {len(visual_chunks)} visual chunks")
    
    # Sample with diversity
    if len(visual_chunks) > num_questions:
        # Try to get one from each source first
        sources = visual_chunks['source_file'].unique()
        sampled = []
        
        for source in sources[:num_questions]:
            source_chunks = visual_chunks[visual_chunks['source_file'] == source]
            if len(source_chunks) > 0:
                sampled.append(source_chunks.iloc[0].to_dict())
        
        # Fill remaining randomly
        if len(sampled) < num_questions:
            remaining = visual_chunks[~visual_chunks.index.isin([s.get('index') for s in sampled])]
            needed = num_questions - len(sampled)
            if len(remaining) > 0:
                additional = remaining.sample(n=min(needed, len(remaining)), random_state=42)
                sampled.extend(additional.to_dict('records'))
        
        visual_chunks = pd.DataFrame(sampled).head(num_questions)
    else:
        visual_chunks = visual_chunks.head(num_questions)
    
    logger.info(f"Selected {len(visual_chunks)} visual chunks for evaluation")
    
    # Initialize LLM
    try:
        llm = OllamaLLM(model=model_name, temperature=0.2, max_tokens=512)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        return None
    
    # Prompt for diagram-specific questions
    prompt_template = """You are an expert professor analyzing an educational diagram.

DIAGRAM DESCRIPTION:
{diagram_description}

SOURCE: {source_file}, Page {page_number}

Write ONE specific exam question that requires understanding this diagram.
The question should ask about concepts VISUALLY represented (structure, architecture, components, relationships).

Requirements:
- Question must reference the diagram/figure explicitly or implicitly
- Answer must be derivable from the description provided
- Be specific to visual elements (not generic text knowledge)

Format exactly as:
QUESTION: [Your specific question about the diagram]
ANSWER: [Comprehensive answer based on the description]

Do not include any other text."""

    dataset = []
    
    for _, chunk in tqdm(visual_chunks.iterrows(), total=len(visual_chunks), desc="Generating visual QA"):
        filled_prompt = prompt_template.format(
            diagram_description=chunk['text'],
            source_file=chunk.get('source_file', 'Unknown'),
            page_number=chunk.get('page_number', 'N/A')
        )
        
        try:
            response = llm.invoke(filled_prompt)
            
            # Clean DeepSeek tags
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            
            # Parse
            q_match = re.search(r'QUESTION:\s*(.*?)\n\s*ANSWER:', response, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'ANSWER:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            
            if q_match and a_match:
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                
                if len(question) < 20 or len(answer) < 50:
                    continue
                
                dataset.append({
                    "query": question,
                    "answer": answer,
                    "target_chunk_id": chunk.get('chunk_id'),
                    "source_file": chunk.get('source_file', 'Unknown'),
                    "page_number": chunk.get('page_number', 'N/A'),
                    "image_ref": chunk.get('image_ref', ''),
                    "vlm_caption": chunk.get('text', '')[:300]
                })
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('chunk_id')}: {e}")
            continue
    
    if not dataset:
        logger.error("No valid QA pairs generated")
        return None
    
    out_df = pd.DataFrame(dataset)
    out_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Generated {len(dataset)} visual QA pairs")
    logger.info(f"Saved to: {output_path}")
    
    # Show samples
    logger.info("\nSample queries:")
    for i, row in out_df.head(3).iterrows():
        logger.info(f"  {i+1}. {row['query'][:70]}...")
        logger.info(f"     Target: {row['target_chunk_id']} (Page {row['page_number']})")
    
    return out_df


# ==================== Visual Retrieval Evaluation ====================

class VisualRetriever:
    """Full system retriever for visual evaluation."""
    
    def __init__(self, docs: List[Document], faiss_store, hyde_llm):
        self.docs = docs
        self.faiss = faiss_store
        self.hyde_llm = hyde_llm
        self.reranker = DocumentReranker()
        
        tokenized = [d.page_content.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        fetch_k = 20
        
        # BM25 with original query
        scores = self.bm25.get_scores(query.lower().split())
        idx = np.argsort(scores)[-fetch_k:][::-1]
        bm25_docs = [self.docs[i] for i in idx if scores[i] > 0]
        
        # FAISS with HyDE
        hyde_query = generate_hypothetical_document(query, self.hyde_llm)
        faiss_docs = self.faiss.similarity_search(hyde_query, k=fetch_k)
        
        # RRF fusion
        fused = rrf_fusion(bm25_docs, faiss_docs, k=60)
        
        # Rerank
        reranked = self.reranker.rerank(query, fused, top_n=top_k)
        return reranked


def evaluate_visual_only(
    visual_csv: str = "visual_evaluation_dataset.csv",
    kb_csv: str = "educational_knowledge_base.csv"
) -> Optional[Dict]:
    """Evaluate visual retrieval performance only."""
    
    logger.info("\n" + "=" * 70)
    logger.info("VISUAL RETRIEVAL EVALUATION")
    logger.info("=" * 70)
    
    if not os.path.exists(visual_csv):
        logger.error(f"Visual dataset not found: {visual_csv}")
        logger.info("Run generate_visual_dataset() first")
        return None
    
    # Load data
    visual_df = pd.read_csv(visual_csv)
    kb_df = pd.read_csv(kb_csv)
    
    logger.info(f"Evaluating {len(visual_df)} visual queries")
    
    # Build type map
    chunk_type_map = dict(zip(kb_df['chunk_id'], kb_df['type']))
    
    # Load documents and indices
    logger.info("Loading documents and building indices...")
    docs = []
    for idx, row in kb_df.iterrows():
        content = str(row.get("text", "")).strip()
        if not content:
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "chunk_id": row.get("chunk_id", f"row-{idx}"),
                "source_file": row.get("source_file", "Unknown"),
                "page_number": row.get("page_number", "N/A"),
                "type": row.get("type", "text")
            }
        ))
    
    # Build FAISS
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists("faiss_index"):
        faiss_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    else:
        faiss_store = FAISS.from_documents(docs, embedder)
    
    # Setup retriever
    hyde_llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1, max_tokens=256)
    retriever = VisualRetriever(docs, faiss_store, hyde_llm)
    
    # Setup generation
    llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.1, max_tokens=1024)
    evidence_attributor = EvidenceAttributor()
    token_attributor = TokenAttributor()
    
    # Metrics
    total = len(visual_df)
    visual_in_top5 = 0
    visual_at_1 = 0
    visual_at_2 = 0
    visual_at_3 = 0
    rank_sum = 0.0
    target_in_top5 = 0
    target_at_1 = 0
    
    # For generation quality (optional)
    faithfulness_scores = []
    
    detailed_results = []
    
    for idx, row in tqdm(visual_df.iterrows(), total=total, desc="Evaluating"):
        query = str(row["query"])
        target_id = str(row["target_chunk_id"]).strip()
        
        # Retrieve
        context_docs = retriever.retrieve(query, top_k=5)
        retrieved_ids = [d.metadata.get("chunk_id") for d in context_docs]
        retrieved_types = [chunk_type_map.get(cid, "unknown") for cid in retrieved_ids]
        
        # Visual analysis
        visual_positions = [i for i, t in enumerate(retrieved_types) if t == "visual_content"]
        
        has_visual = len(visual_positions) > 0
        best_visual_rank = min(visual_positions) + 1 if visual_positions else None
        
        # Target analysis
        target_rank = retrieved_ids.index(target_id) + 1 if target_id in retrieved_ids else None
        target_in_results = target_id in retrieved_ids[:5]
        target_first = retrieved_ids[0] == target_id if retrieved_ids else False
        
        # Update metrics
        if has_visual:
            visual_in_top5 += 1
            rank_sum += 1.0 / best_visual_rank
            
            if best_visual_rank == 1:
                visual_at_1 += 1
            elif best_visual_rank == 2:
                visual_at_2 += 1
            elif best_visual_rank == 3:
                visual_at_3 += 1
        
        if target_in_results:
            target_in_top5 += 1
            if target_first:
                target_at_1 += 1
        
        # Optional: generate and evaluate answer
        try:
            context_text = combine_documents_for_prompt(context_docs)
            filled_prompt = STRICT_QA_PROMPT.format(context=context_text, question=query)
            raw = llm.invoke(filled_prompt)
            answer = clean_llm_output(raw)
            
            faith = compute_faithfulness(answer, context_docs)
            faithfulness_scores.append(faith)
            
            # Token grounding
            tok_attr = token_attributor.attribute(answer, context_docs)
            grounding_ratio = tok_attr.get("grounding_ratio", 0)
            
            # Evidence attribution
            ev_attr = evidence_attributor.attribute(query, answer, context_docs)
            top_contrib = ev_attr["attributions"][0] if ev_attr.get("attributions") else None
            
        except Exception as e:
            logger.warning(f"Generation failed for query {idx}: {e}")
            faith = 0
            grounding_ratio = 0
            top_contrib = None
            answer = ""
        
        # Store detailed result
        detailed_results.append({
            "query": query,
            "target_id": target_id,
            "target_type": chunk_type_map.get(target_id, "unknown"),
            "retrieved_ids": retrieved_ids,
            "retrieved_types": retrieved_types,
            "visual_positions": visual_positions,
            "has_visual": has_visual,
            "best_visual_rank": best_visual_rank,
            "target_rank": target_rank,
            "target_in_top5": target_in_results,
            "faithfulness": faith,
            "token_grounding": grounding_ratio,
            "top_evidence_source": top_contrib["source"] if top_contrib else None,
            "top_evidence_type": chunk_type_map.get(top_contrib["chunk_id"], "unknown") if top_contrib else None,
            "answer_preview": answer[:100] if answer else ""
        })
    
    # Calculate summary metrics
    results = {
        "total_queries": total,
        "visual_recall@5": visual_in_top5 / total if total > 0 else 0,
        "visual_mrr": rank_sum / total if total > 0 else 0,
        "visual_at_1": visual_at_1 / total if total > 0 else 0,
        "visual_at_2": visual_at_2 / total if total > 0 else 0,
        "visual_at_3": visual_at_3 / total if total > 0 else 0,
        "visual_at_1_2_3": (visual_at_1 + visual_at_2 + visual_at_3) / total if total > 0 else 0,
        "target_recall@5": target_in_top5 / total if total > 0 else 0,
        "target_mrr": sum(1.0/r for r in [d["target_rank"] for d in detailed_results if d["target_rank"]]) / total if total > 0 else 0,
        "target_at_1": target_at_1 / total if total > 0 else 0,
        "avg_faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
        "queries_with_visual": visual_in_top5,
        "queries_with_target": target_in_top5,
        "raw_counts": {
            "visual_in_top5": visual_in_top5,
            "visual_at_1": visual_at_1,
            "visual_at_2": visual_at_2,
            "visual_at_3": visual_at_3,
            "target_in_top5": target_in_top5,
            "target_at_1": target_at_1
        }
    }
    
    # Print report
    print("\n" + "=" * 70)
    print("VISUAL RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"Total visual queries:           {total}")
    print(f"Queries with visual in top-5:   {visual_in_top5} ({results['visual_recall@5']*100:.1f}%)")
    print(f"Visual at Rank 1:               {visual_at_1} ({results['visual_at_1']*100:.1f}%)")
    print(f"Visual at Rank 2:               {visual_at_2} ({results['visual_at_2']*100:.1f}%)")
    print(f"Visual at Rank 3:               {visual_at_3} ({results['visual_at_3']*100:.1f}%)")
    print(f"Visual at Rank 1-3:             {visual_at_1+visual_at_2+visual_at_3} ({results['visual_at_1_2_3']*100:.1f}%)")
    print(f"Visual MRR:                     {results['visual_mrr']:.4f}")
    print("-" * 70)
    print(f"Target chunk retrieved (top-5): {target_in_top5} ({results['target_recall@5']*100:.1f}%)")
    print(f"Target at Rank 1:               {target_at_1} ({results['target_at_1']*100:.1f}%)")
    print(f"Target MRR:                     {results['target_mrr']:.4f}")
    print("-" * 70)
    print(f"Avg Faithfulness:               {results['avg_faithfulness']:.4f}")
    print("=" * 70)
    
    # Save results
    with open("visual_evaluation_results.json", "w") as f:
        json.dump({
            "summary": results,
            "detailed_traces": detailed_results
        }, f, indent=2)
    
    logger.info("Results saved to visual_evaluation_results.json")
    
    # Print examples
    print("\n" + "=" * 70)
    print("EXAMPLE TRACES")
    print("=" * 70)
    
    # Example 1: Visual retrieved first
    visual_first = [d for d in detailed_results if d["best_visual_rank"] == 1]
    if visual_first:
        ex = visual_first[0]
        print(f"\n[Visual Ranked First]")
        print(f"Query: {ex['query']}")
        print(f"Target: {ex['target_id']} ({ex['target_type']})")
        print(f"Visual positions in top-5: {ex['visual_positions']}")
        print(f"Top evidence: {ex['top_evidence_source']} ({ex['top_evidence_type']})")
        print(f"Faithfulness: {ex['faithfulness']:.3f}")
    
    # Example 2: Target retrieved but visual not prominent
    target_no_visual = [d for d in detailed_results 
                       if d["target_in_top5"] and not d["has_visual"]]
    if target_no_visual:
        ex = target_no_visual[0]
        print(f"\n[Target Retrieved, No Visual in Top-5]")
        print(f"Query: {ex['query']}")
        print(f"Retrieved types: {ex['retrieved_types']}")
        print(f"Target rank: {ex['target_rank']}")
    
    # Example 3: Visual-heavy retrieval
    visual_heavy = [d for d in detailed_results if len(d["visual_positions"]) >= 2]
    if visual_heavy:
        ex = visual_heavy[0]
        print(f"\n[Multiple Visuals in Top-5]")
        print(f"Query: {ex['query']}")
        print(f"Visual positions: {ex['visual_positions']}")
        print(f"Retrieved types: {ex['retrieved_types']}")
    
    print("=" * 70)
    
    return results


# ==================== Main ====================

def main():
    """Run visual evaluation only."""
    
    kb_path = "educational_knowledge_base.csv"
    visual_csv = "visual_evaluation_dataset.csv"
    
    # Check knowledge base
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        return
    
    # Generate visual dataset if needed
    if not os.path.exists(visual_csv):
        logger.info("Visual dataset not found, generating...")
        df = generate_visual_dataset(kb_path, visual_csv, num_questions=15)
        if df is None:
            logger.error("Failed to generate visual dataset")
            return
    else:
        logger.info(f"Using existing visual dataset: {visual_csv}")
    
    # Run evaluation
    results = evaluate_visual_only(visual_csv, kb_path)
    
    if results:
        # Print paper-ready summary
        print("\n" + "=" * 70)
        print("PAPER-READY SUMMARY")
        print("=" * 70)
        print(f"For queries explicitly referencing diagrams (n={results['total_queries']}):")
        print(f"  • Visual chunks appear in top-5: {results['visual_recall@5']*100:.1f}% of queries")
        print(f"  • Visual chunks rank first: {results['visual_at_1']*100:.1f}% of queries")
        print(f"  • Visual MRR: {results['visual_mrr']:.3f}")
        print(f"  • Target visual retrieved: {results['target_recall@5']*100:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()