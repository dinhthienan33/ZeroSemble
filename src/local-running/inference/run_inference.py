import os
import sys
import json
import argparse
from typing import List, Dict, Any
import time

# Add the parent directory to the path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm import ModelManager

def format_prompt(sample: Dict[str, Any], domain: str = None) -> List[Dict[str, str]]:
    """
    Format a sample document into a prompt for information extraction.
    
    Args:
        sample: Document sample containing text, title, etc.
        domain: Optional domain specification for the document
        
    Returns:
        Formatted messages for the model
    """
    # Use the domain from the sample if not explicitly provided
    if domain is None and "domain" in sample:
        domain = sample["domain"]
    
    prompt = f"""
        You are an advanced information extraction model specializing in Named Entity Recognition (NER) and Relation Extraction (RE).  
        Your specific domain is {domain}.  
        Extract named entities and relationships from the given document.  
        Return only the extracted JSON output without any extra text.  
        Extract relevant named entities and their relationships based on predefined NER and RE labels.  
        Find all entities that you can find.  

        ### Input:  
        {json.dumps(sample, ensure_ascii=False)}  

        ### Output Format:
        {{
            "{sample['id']}": {{
                "title": "{sample['title']}",
                "entities": [
                    {{
                        "mentions": ["<Entity Text>"],
                        "type": "<NER Label>"
                    }}
                ]
            }}
        }}
        """
    
    return [{"role": "user", "content": prompt}]

def format_triple_prompt(sample: Dict[str, Any], entities_list: List[str], doc_id: str = None) -> List[Dict[str, str]]:
    """
    Format a prompt to extract triples based on already extracted entities.
    
    Args:
        sample: Document sample containing text, title, etc.
        entities_list: List of already extracted entities
        doc_id: Optional document ID
        
    Returns:
        Formatted messages for the model
    """
    # Use the doc_id from the sample if not explicitly provided
    if doc_id is None and "id" in sample:
        doc_id = sample["id"]
    
    # Create a sample without NER for the triple extraction
    sample_without_ner = {k: v for k, v in sample.items() if k != "entities"}
    
    prompt = f"""
        You are an advanced information extraction model specializing in Relation Extraction (RE). 
        Your specific domain is {sample['domain']}.
        Extract relationships from the given document with a focus on the provided entities. 
        Based on the document id '{doc_id}' and its corresponding entities {entities_list}, please identify the relation triples where the 'head' and 'tail' are among these entities.
        Return only the extracted JSON output without any extra text.
        Extract relevant named entities and their relationships based on predefined RE labels.
        Try to find exactly.

        ### Input:
        {json.dumps(sample_without_ner, ensure_ascii=False)}

        ### Output Format:
        {{
            "{doc_id}": {{
                "title": "{sample['title']}",
                 "entities": [
                    {{
                        "mentions": ["<Entity Text>"],
                        "type": "<NER Label>"
                    }}
                ],
                "triples": [
                    {{
                        "head": "<Entity 1>",
                        "relation": "<Relationship>",
                        "tail": "<Entity 2>"
                    }}
                ]
            }}
        }}
        """
    
    return [{"role": "user", "content": prompt}]

def extract_entities_and_relations(manager: ModelManager, samples: List[Dict[str, Any]], 
                                  use_two_stage: bool = True,
                                  verbose: bool = False) -> Dict[str, Any]:
    """
    Run the extraction pipeline on a list of document samples.
    
    Args:
        manager: ModelManager instance with loaded models
        samples: List of document samples
        use_two_stage: Whether to use the two-stage pipeline with separate entity and relation extraction
        verbose: Whether to print detailed information during extraction
        
    Returns:
        Dictionary of extraction results for each sample
    """
    results = {}
    
    for i, sample in enumerate(samples):
        doc_id = sample["id"]
        if verbose:
            print(f"Processing document {i+1}/{len(samples)}: {doc_id}")
        
        # Stage 1: Extract entities and initial triples from all models
        messages = format_prompt(sample)
        initial_results = manager.generate_from_all(messages)
        
        # Process and merge results from different models (entity ensemble)
        # For simplicity, we'll just use one model's results in this example
        # In a real implementation, you'd merge entities and maybe initial triples
        entities_results = initial_results["qwen"]  # In reality, you'd process and combine all models
        
        if not use_two_stage:
            results[doc_id] = entities_results
            continue
            
        # Stage 2: Extract triples based on entities using Qwen model
        # In a real implementation, you'd extract and clean entities from stage 1 results
        entities_list = ["entity1", "entity2", "entity3"]  # Placeholder - extract from stage 1
        
        if verbose:
            print(f"  - Running second stage with {len(entities_list)} entities")
            
        triple_messages = format_triple_prompt(sample, entities_list, doc_id)
        triple_results = manager.generate(triple_messages, model_name="qwen")
        
        # Combine results (in a real implementation, merge entities from stage 1 and relations from stage 2)
        results[doc_id] = triple_results
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference for information extraction")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file with samples")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file for results")
    parser.add_argument("--models", type=str, nargs="+", default=["qwen"], 
                        choices=["qwen", "deepseek", "llama"],
                        help="Models to use for inference")
    parser.add_argument("--all-models", action="store_true", help="Use all available models")
    parser.add_argument("--single-stage", action="store_true", 
                        help="Use single-stage extraction instead of two-stage pipeline")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    if isinstance(samples, dict):
        samples = [samples]  # Convert single sample to list
        
    # Initialize model manager
    if args.verbose:
        print(f"Loading models: {'all' if args.all_models else args.models}")
        
    manager = ModelManager(use_models=args.models, load_all=args.all_models)
    
    # Run extraction
    start_time = time.time()
    results = extract_entities_and_relations(
        manager, 
        samples, 
        use_two_stage=not args.single_stage,
        verbose=args.verbose
    )
    end_time = time.time()
    
    if args.verbose:
        print(f"Extraction completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    if args.verbose:
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 