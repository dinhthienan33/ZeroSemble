#!/usr/bin/env python3
"""
Document-Level Information Extraction Pipeline

This script runs the complete information extraction pipeline:
1. Extracts entities and initial triples from multiple models
2. Ensembles the results from different models
3. Uses a second stage to improve triple extraction based on entities
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any

from models.llm import ModelManager
from utils.ensemble import (
    extract_entities_from_results, 
    combine_models_output, 
    refine_triples_with_entities
)
from inference.run_inference import format_prompt, format_triple_prompt

def process_document(
    manager: ModelManager,
    sample: Dict[str, Any],
    use_two_stage: bool = True,
    entity_min_votes: int = 2,
    triple_min_votes: int = 2,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process a single document through the extraction pipeline.
    
    Args:
        manager: ModelManager instance with loaded models
        sample: Document sample to process
        use_two_stage: Whether to use the two-stage pipeline
        entity_min_votes: Minimum votes needed for entity consensus
        triple_min_votes: Minimum votes needed for triple consensus
        verbose: Whether to print detailed processing information
        
    Returns:
        Extraction results for the document
    """
    doc_id = sample.get("id", "unknown")
    title = sample.get("title", "")
    
    if verbose:
        print(f"\nProcessing document: {doc_id} - {title}")
        print("=" * 50)
    
    # Stage 1: Extract entities and initial triples from all models
    if verbose:
        print("Stage 1: Extracting entities and initial triples...")
    
    messages = format_prompt(sample)
    model_outputs = manager.generate_from_all(messages)
    
    # Ensemble the results from different models
    if verbose:
        print(f"Combining outputs from {len(model_outputs)} models...")
    
    combined_results = combine_models_output(
        model_outputs, 
        doc_id, 
        title,
        entity_min_votes=entity_min_votes,
        triple_min_votes=triple_min_votes
    )
    
    # If not using two-stage approach, return the combined results
    if not use_two_stage:
        return combined_results
    
    # Stage 2: Use entities from Stage 1 to extract better triples
    if verbose:
        print("Stage 2: Refining triples based on extracted entities...")
    
    # Extract entities from the combined results
    entities = combined_results[doc_id]["entities"]
    entity_mentions = []
    for entity in entities:
        entity_mentions.extend(entity.get("mentions", []))
    
    if verbose:
        print(f"  - Using {len(entity_mentions)} entity mentions for triple extraction")
    
    # Generate a new prompt for triple extraction using the entities
    triple_messages = format_triple_prompt(sample, entity_mentions, doc_id)
    
    # Use Qwen model for the second stage (could be configurable)
    triple_model = "qwen"  # Could be made configurable
    if verbose:
        print(f"  - Using {triple_model} model for triple refinement")
    
    triple_results = manager.generate(triple_messages, model_name=triple_model)
    
    # Extract triples from the result
    refined_triples = []
    try:
        # Parse the result if it's a string
        if isinstance(triple_results, str):
            triple_results = json.loads(triple_results)
        
        # Extract triples if the parsing succeeded
        if doc_id in triple_results and "triples" in triple_results[doc_id]:
            refined_triples = triple_results[doc_id]["triples"]
        
        # Filter triples to only include those with head and tail in the entities
        refined_triples = refine_triples_with_entities(refined_triples, entities)
        
        if verbose:
            print(f"  - Extracted {len(refined_triples)} refined triples")
    except Exception as e:
        if verbose:
            print(f"  - Error processing triple results: {e}")
    
    # Create the final result by combining entities from Stage 1 and triples from Stage 2
    final_result = {
        doc_id: {
            "title": title,
            "entities": entities,
            "triples": refined_triples
        }
    }
    
    return final_result

def main():
    parser = argparse.ArgumentParser(description="Document-Level Information Extraction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--models", type=str, nargs="+", default=["qwen", "deepseek", "llama"], 
                       choices=["qwen", "deepseek", "llama"],
                       help="Models to use for inference")
    parser.add_argument("--single-stage", action="store_true", 
                       help="Use single-stage extraction instead of two-stage pipeline")
    parser.add_argument("--entity-votes", type=int, default=2,
                       help="Minimum number of models that must agree on an entity")
    parser.add_argument("--triple-votes", type=int, default=2,
                       help="Minimum number of models that must agree on a triple")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model manager with requested models
    if args.verbose:
        print(f"Loading models: {args.models}")
    
    manager = ModelManager(use_models=args.models)
    
    # Process input (file or directory)
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process a single input file
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both single samples and lists of samples
        samples = data if isinstance(data, list) else [data]
        
        total_start_time = time.time()
        all_results = {}
        
        for i, sample in enumerate(samples):
            if args.verbose:
                print(f"\nProcessing sample {i+1}/{len(samples)}")
            
            start_time = time.time()
            result = process_document(
                manager,
                sample,
                use_two_stage=not args.single_stage,
                entity_min_votes=args.entity_votes,
                triple_min_votes=args.triple_votes,
                verbose=args.verbose
            )
            end_time = time.time()
            
            if args.verbose:
                doc_id = next(iter(result.keys()))
                num_entities = len(result[doc_id]["entities"])
                num_triples = len(result[doc_id]["triples"])
                print(f"Extracted {num_entities} entities and {num_triples} triples in {end_time - start_time:.2f} seconds")
            
            # Add result to all results
            all_results.update(result)
        
        # Save the results
        output_file = output_dir / "extraction_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        total_end_time = time.time()
        if args.verbose:
            print(f"\nProcessed {len(samples)} samples in {total_end_time - total_start_time:.2f} seconds")
            print(f"Results saved to {output_file}")
    
    elif input_path.is_dir():
        # Process all JSON files in the directory
        json_files = list(input_path.glob("*.json"))
        
        if args.verbose:
            print(f"Found {len(json_files)} JSON files in {input_path}")
        
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return
        
        total_start_time = time.time()
        total_samples = 0
        
        for json_file in json_files:
            if args.verbose:
                print(f"\nProcessing file: {json_file.name}")
            
            # Load samples from the file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both single samples and lists of samples
            samples = data if isinstance(data, list) else [data]
            total_samples += len(samples)
            
            all_results = {}
            for i, sample in enumerate(samples):
                if args.verbose:
                    print(f"Processing sample {i+1}/{len(samples)}")
                
                result = process_document(
                    manager,
                    sample,
                    use_two_stage=not args.single_stage,
                    entity_min_votes=args.entity_votes,
                    triple_min_votes=args.triple_votes,
                    verbose=args.verbose
                )
                
                # Add result to all results
                all_results.update(result)
            
            # Save the results for this file
            output_file = output_dir / f"results_{json_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            if args.verbose:
                print(f"Results for {json_file.name} saved to {output_file}")
        
        total_end_time = time.time()
        if args.verbose:
            print(f"\nProcessed {total_samples} samples from {len(json_files)} files in {total_end_time - total_start_time:.2f} seconds")
    
    else:
        print(f"Input path {input_path} does not exist or is not a file or directory")

if __name__ == "__main__":
    main() 