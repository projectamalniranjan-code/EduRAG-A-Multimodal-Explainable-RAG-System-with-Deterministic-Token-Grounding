"""
generate_visual_ground_truth.py
Generates evaluation dataset from VLM-captioned visual chunks.
Uses LLM-as-Teacher approach consistent with text evaluation.
"""

import pandas as pd
import random
import os
import logging
import re
from tqdm import tqdm
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_visual_evaluation_dataset(
    kb_path="educational_knowledge_base.csv",
    output_path="visual_evaluation_dataset.csv",
    num_questions=15,
    model_name="deepseek-r1:8b"
):
    """
    Generate QA pairs from VLM-captioned visual chunks.
    Filters for substantive educational diagrams (not logos/decorative).
    """
    
    logger.info("--- Visual Ground Truth Generation ---")
    
    if not os.path.exists(kb_path):
        logger.error(f"Knowledge base not found: {kb_path}")
        return
    
    # Load knowledge base
    df = pd.read_csv(kb_path, dtype=str, keep_default_na=False)
    
    # Filter for visual chunks with meaningful captions
    # Exclude: short captions, generic labels, decorative images
    visual_chunks = df[
        (df['type'] == 'visual_content') &
        (df['text'].str.len() > 80) &  # Substantial caption
        (df['text'].str.len() < 800) &  # Not too long (VLM truncation)
        (~df['text'].str.contains('logo|decorative|banner|header', case=False, na=False)) &
        (df['image_ref'].notna()) &
        (df['image_ref'] != '')
    ].copy()
    
    if len(visual_chunks) == 0:
        logger.error("No valid visual chunks found. Check VLM processing.")
        return
    
    logger.info(f"Found {len(visual_chunks)} visual chunks with captions")
    
    # Sample with diversity across sources
    if len(visual_chunks) > num_questions:
        # Try to get diversity across source files
        sources = visual_chunks['source_file'].unique()
        sampled = []
        
        # At least one from each source if possible
        for source in sources:
            source_chunks = visual_chunks[visual_chunks['source_file'] == source]
            if len(source_chunks) > 0:
                n_samples = min(2, len(source_chunks), num_questions - len(sampled))
                sampled.extend(source_chunks.sample(n=n_samples, random_state=42).to_dict('records'))
                if len(sampled) >= num_questions:
                    break
        
        # Fill remaining randomly
        if len(sampled) < num_questions:
            remaining = visual_chunks[~visual_chunks.index.isin([s.get('index') for s in sampled])]
            needed = num_questions - len(sampled)
            if len(remaining) > 0:
                sampled.extend(remaining.sample(n=min(needed, len(remaining)), random_state=42).to_dict('records'))
        
        visual_chunks = pd.DataFrame(sampled).head(num_questions)
    else:
        visual_chunks = visual_chunks.to_dict('records')
    
    actual_n = len(visual_chunks) if isinstance(visual_chunks, pd.DataFrame) else len(visual_chunks)
    logger.info(f"Selected {actual_n} visual chunks for evaluation")
    
    # Initialize LLM
    logger.info(f"Loading LLM ({model_name})...")
    try:
        llm = OllamaLLM(model=model_name, temperature=0.2, max_tokens=512)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        return
    
    # Prompt template - designed to generate diagram-specific questions
    prompt_template = """You are an expert professor analyzing an educational diagram.

DIAGRAM DESCRIPTION:
{diagram_description}

SOURCE: {source_file}, Page {page_number}

Your task: Write ONE specific exam question that requires understanding this diagram.
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
    
    chunks_iter = visual_chunks.iterrows() if isinstance(visual_chunks, pd.DataFrame) else enumerate(visual_chunks)
    
    for idx, chunk in tqdm(list(chunks_iter), desc="Generating visual QA"):
        if isinstance(visual_chunks, pd.DataFrame):
            chunk = chunk.to_dict()
        
        # Build prompt
        filled_prompt = prompt_template.format(
            diagram_description=chunk['text'],
            source_file=chunk.get('source_file', 'Unknown'),
            page_number=chunk.get('page_number', 'N/A')
        )
        
        try:
            response = llm.invoke(filled_prompt)
            
            # Clean DeepSeek tags if present
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            
            # Parse QUESTION and ANSWER
            q_match = re.search(r'QUESTION:\s*(.*?)\n\s*ANSWER:', response, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'ANSWER:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
            
            if q_match and a_match:
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                
                # Validate quality
                if len(question) < 20 or len(answer) < 50:
                    logger.debug(f"Skipping low-quality output for chunk {chunk.get('chunk_id')}")
                    continue
                
                dataset.append({
                    "query": question,
                    "answer": answer,
                    "target_chunk_id": chunk.get('chunk_id'),
                    "source_file": chunk.get('source_file', 'Unknown'),
                    "page_number": chunk.get('page_number', 'N/A'),
                    "image_ref": chunk.get('image_ref', ''),
                    "vlm_caption": chunk.get('text', '')[:300],  # Truncated for reference
                    "caption_length": len(chunk.get('text', ''))
                })
                
                logger.info(f"✓ Generated: {question[:60]}...")
            else:
                logger.warning(f"Failed to parse LLM output for chunk {chunk.get('chunk_id')}")
                logger.debug(f"Raw output: {response[:200]}...")
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('chunk_id')}: {e}")
            continue
    
    # Save dataset
    if not dataset:
        logger.error("No valid QA pairs generated. Check LLM output.")
        return
    
    out_df = pd.DataFrame(dataset)
    out_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Successfully generated {len(dataset)} visual QA pairs!")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"\nSample queries:")
    for i, row in out_df.head(3).iterrows():
        logger.info(f"  {i+1}. {row['query'][:70]}...")
        logger.info(f"     Target: {row['target_chunk_id']} (Page {row['page_number']})")
    
    return out_df


def validate_visual_dataset(dataset_path="visual_evaluation_dataset.csv"):
    """Quick validation of generated dataset."""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path)
    
    logger.info(f"\n--- Dataset Validation ---")
    logger.info(f"Total pairs: {len(df)}")
    logger.info(f"Unique sources: {df['source_file'].nunique()}")
    logger.info(f"Page distribution: {df['page_number'].value_counts().head()}")
    
    # Check if images exist
    existing_images = df['image_ref'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False).sum()
    logger.info(f"Images found: {existing_images}/{len(df)}")
    
    # Sample quality check
    logger.info(f"\n--- Sample Entry ---")
    sample = df.iloc[0]
    logger.info(f"Query: {sample['query']}")
    logger.info(f"Answer: {sample['answer'][:150]}...")
    logger.info(f"Target chunk: {sample['target_chunk_id']}")
    
    return True


if __name__ == "__main__":
    # Generate dataset
    df = generate_visual_evaluation_dataset(
        kb_path="educational_knowledge_base.csv",
        output_path="visual_evaluation_dataset.csv",
        num_questions=15
    )
    
    # Validate
    if df is not None:
        validate_visual_dataset("visual_evaluation_dataset.csv")