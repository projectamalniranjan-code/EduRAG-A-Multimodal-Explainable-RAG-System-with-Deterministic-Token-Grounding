import pandas as pd
import random
import os
import logging
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def generate_evaluation_dataset(
    kb_path="educational_knowledge_base.csv", 
    output_path="evaluation_dataset.csv", 
    num_questions=100  # Change this to 100 later if want a larger test set
):
    logger.info("--- Starting Ground Truth Dataset Generation ---")
    
    # 1. Load the Knowledge Base
    if not os.path.exists(kb_path):
        logger.error(f"Cannot find {kb_path}. Please ensure your database exists.")
        return
        
    df = pd.read_csv(kb_path, dtype=str, keep_default_na=False)
    
    # 2. Filter for high-quality text chunks
    # We strictly ignore 'visual_content' here to prevent the LLM from trying 
    # to write questions based on image descriptions.
    valid_chunks = df[
        (df['type'].isin(['text', 'table'])) & 
        (df['text'].str.len() > 150) & 
        (df['text'].str.len() < 1500)
    ].to_dict('records')
    
    if len(valid_chunks) < num_questions:
        logger.warning(f"Only found {len(valid_chunks)} valid chunks. Adjusting question count.")
        num_questions = len(valid_chunks)
        
    # Randomly sample the chunks
    sampled_chunks = random.sample(valid_chunks, num_questions)
    
    # 3. Initialize the "Teacher" LLM
    # We use a temperature of 0.3 so it is creative enough to write good questions,
    # but grounded enough to extract the exact answer.
    logger.info("Loading LLM (deepseek-r1:8b)...")
    try:
        llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.3)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}. Make sure Ollama is running.")
        return
    
    dataset = []
    
    prompt_template = """You are an expert college professor. I will provide you with a paragraph from a textbook.
Your task is to write ONE specific exam question that can be answered using ONLY the provided text.
Then, provide a COMPREHENSIVE, detailed answer based on the text.

GUIDELINES FOR THE ANSWER:
1. Write 2-4 complete sentences, not just a phrase.
2. Explain the reasoning or context fully.
3. Include all relevant details from the text.
4. Do not be concise; be thorough and educational.

Format your response EXACTLY like this:
QUESTION: [Your question here]
ANSWER: [Your answer here]

Textbook Paragraph:
{text}
"""

    # 4. Generate the pairs
    for idx, chunk in enumerate(tqdm(sampled_chunks, desc="Generating QA Pairs")):
        text = chunk['text']
        chunk_id = chunk.get('chunk_id', f"chunk_{idx}")
        
        try:
            filled_prompt = prompt_template.format(text=text)
            response = llm.invoke(filled_prompt)
            
            # Clean up <think> tags if deepseek uses them
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            
            # Parse the QUESTION and ANSWER using regex
            q_match = re.search(r'QUESTION:\s*(.*?)\nANSWER:', response, re.DOTALL)
            a_match = re.search(r'ANSWER:\s*(.*)', response, re.DOTALL)
            
            if q_match and a_match:
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                
                dataset.append({
                    "query": question,
                    "answer": answer,
                    "target_chunk_id": chunk_id,
                    "source_file": chunk.get("source_file", "Unknown")
                })
            else:
                logger.debug(f"Failed to parse LLM output for chunk {chunk_id}")
                
        except Exception as e:
            logger.error(f"Error generating for chunk {chunk_id}: {e}")
            
    # 5. Save to CSV
    if not dataset:
        logger.error("Failed to generate any valid QA pairs. Check your LLM output format.")
        return

    out_df = pd.DataFrame(dataset)
    out_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Successfully generated {len(dataset)} Ground Truth pairs!")
    logger.info(f"Saved to: {output_path}")
    logger.info("You are now ready to run evaluation.py to benchmark your system!")

if __name__ == "__main__":
    generate_evaluation_dataset()