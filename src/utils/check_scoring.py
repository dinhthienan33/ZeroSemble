import json
import os
import sys
from pathlib import Path

def check_mention_list_types(file_path, description):
    """Check if mention lists are properly formatted as sets"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for doc_id, doc in data.items():
                # First check if doc is a dictionary
                if not isinstance(doc, dict):
                    print(f"❌ Document {doc_id}: document should be a dictionary, got {type(doc)}")
                    return False
                
                # Check if entities is a list
                if not isinstance(doc.get('entities', []), list):
                    print(f"❌ Document {doc_id}: 'entities' should be a list, got {type(doc.get('entities'))}")
                    return False
                
                # Check entities
                for i, entity in enumerate(doc.get('entities', [])):
                    if not isinstance(entity, dict):
                        print(f"❌ Document {doc_id}: entity at index {i} should be a dictionary, got {type(entity)}")
                        return False
                    
                    mentions = entity.get('mentions', [])
                    if not isinstance(mentions, list):
                        print(f"❌ Document {doc_id}: mentions should be a list, got {type(mentions)}")
                        return False
                    
                    # Check if all mentions are strings
                    for j, mention in enumerate(mentions):
                        if not isinstance(mention, str):
                            print(f"❌ Document {doc_id}: mention at index {j} should be a string, got {type(mention)}")
                            return False
                
                # Check if triples is a list
                if not isinstance(doc.get('triples', []), list):
                    print(f"❌ Document {doc_id}: 'triples' should be a list, got {type(doc.get('triples'))}")
                    return False
                
                # Check triples
                for i, triple in enumerate(doc.get('triples', [])):
                    if not isinstance(triple, dict):
                        print(f"❌ Document {doc_id}: triple at index {i} should be a dictionary, got {type(triple)}")
                        return False
                    
                    head = triple.get('head', '')
                    tail = triple.get('tail', '')
                    if not isinstance(head, str) or not isinstance(tail, str):
                        print(f"❌ Document {doc_id}: head and tail in triples should be strings")
                        return False
            
            print(f"✅ {description} file has correct mention list types")
            return True
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f"❌ Error checking mention list types at line {line_number}: {str(e)}")
        if "'str' object has no attribute 'get'" in str(e):
            # Try to identify the problematic document
            try:
                for doc_id, doc in data.items():
                    if not isinstance(doc, dict):
                        print(f"  Problem identified in document ID: {doc_id}, which is a {type(doc).__name__} instead of a dictionary")
                    elif 'entities' in doc and not isinstance(doc['entities'], list):
                        print(f"  Problem identified in document ID: {doc_id}, 'entities' is a {type(doc['entities']).__name__} instead of a list")
                    elif 'triples' in doc and not isinstance(doc['triples'], list):
                        print(f"  Problem identified in document ID: {doc_id}, 'triples' is a {type(doc['triples']).__name__} instead of a list")
            except Exception as inner_e:
                print(f"  Could not fully identify the problem: {str(inner_e)}")
        return False

def validate_file(file_path, description):
    """Validate if a file exists and is valid JSON"""
    if not os.path.exists(file_path):
        print(f"❌ Error: {description} file not found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ {description} file is valid JSON")
            return True
    except json.JSONDecodeError:
        print(f"❌ Error: {description} file is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading {description} file: {str(e)}")
        return False

def check_file_structure(file_path, description):
    """Check if the file has the expected structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Check if it's a dictionary
            if not isinstance(data, dict):
                print(f"❌ {description} file should be a dictionary")
                return False
            
            # Check for required keys in each document
            for doc_id, doc in data.items():
                if not isinstance(doc, dict):
                    print(f"❌ Document {doc_id} should be a dictionary")
                    return False
                
                required_keys = ['entities', 'triples']
                for key in required_keys:
                    if key not in doc:
                        print(f"❌ Document {doc_id} missing required key: {key}")
                        return False
                
                # Check entities structure
                for entity in doc['entities']:
                    if not all(k in entity for k in ['mentions', 'type']):
                        print(f"❌ Entity in document {doc_id} missing required keys: mentions or type")
                        return False
                
                # Check triples structure
                for triple in doc['triples']:
                    if not all(k in triple for k in ['head', 'relation', 'tail']):
                        print(f"❌ Triple in document {doc_id} missing required keys: head, relation, or tail")
                        return False
            
            print(f"✅ {description} file has correct structure")
            return True
    except Exception as e:
        print(f"❌ Error checking {description} file structure: {str(e)}")
        return False

def compare_document_ids(pred_file, ref_file):
    """Compare document IDs between prediction and reference files"""
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        with open(ref_file, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
        
        pred_ids = set(pred_data.keys())
        ref_ids = set(ref_data.keys())
        
        missing_in_pred = ref_ids - pred_ids
        missing_in_ref = pred_ids - ref_ids
        
        if missing_in_pred:
            print(f"❌ Documents in reference but missing in prediction: {missing_in_pred}")
        if missing_in_ref:
            print(f"❌ Documents in prediction but missing in reference: {missing_in_ref}")
        
        if not missing_in_pred and not missing_in_ref:
            print("✅ All document IDs match between prediction and reference files")
            return True
        return False
    except Exception as e:
        print(f"❌ Error comparing document IDs: {str(e)}")
        return False

def analyze_and_combine_data(file_path, description):
    """Analyze file by combining entities/triples and removing duplicates."""
    print(f"\n--- Analyzing {description} File ({os.path.basename(file_path)}) ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_entities = []
        all_triples = []
        total_entities_before = 0
        total_triples_before = 0
        
        for doc_id, doc in data.items():
            entities = doc.get('entities', [])
            triples = doc.get('triples', [])
            
            total_entities_before += len(entities)
            total_triples_before += len(triples)
            
            all_entities.extend(entities)
            all_triples.extend(triples)
            
        # Deduplicate entities
        unique_entities = set()
        for entity in all_entities:
            # Use a frozenset of mentions for hashing, ignore order
            mentions_tuple = frozenset(entity.get('mentions', []))
            entity_type = entity.get('type', '')
            unique_entities.add((mentions_tuple, entity_type))
            
        # Deduplicate triples
        unique_triples = set()
        for triple in all_triples:
            head = triple.get('head', '')
            relation = triple.get('relation', '')
            tail = triple.get('tail', '')
            unique_triples.add((head, relation, tail))
            
        total_entities_after = len(unique_entities)
        total_triples_after = len(unique_triples)
        
        print("Entities Report:")
        print(f"  - Total entities across all documents (before deduplication): {total_entities_before}")
        print(f"  - Unique entities across all documents (after deduplication):   {total_entities_after}")
        print(f"  - Duplicate entities removed: {total_entities_before - total_entities_after}")
        
        print("\nTriples Report:")
        print(f"  - Total triples across all documents (before deduplication): {total_triples_before}")
        print(f"  - Unique triples across all documents (after deduplication):  {total_triples_after}")
        print(f"  - Duplicate triples removed: {total_triples_before - total_triples_after}")
        print("-------------------------------------")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {description} file: {str(e)}")
        return False

def main():
    # Configuration
    res_file = "combined_results.json"  # Prediction file
    ref_file = "llama-3.3-70b-versatile/results.json"  # Reference file
    output_dir = "source"  # Output directory
    
    print("\n=== Starting Scoring Validation ===\n")
    
    all_checks_passed = True
    
    # Check if files exist and are valid JSON
    print("-- Basic File Validation --")
    pred_valid = validate_file(res_file, "Prediction")
    ref_valid = validate_file(ref_file, "Reference")
    if not (pred_valid and ref_valid):
        all_checks_passed = False
    
    # Check mention list types
    print("\n-- Mention List Type Validation --")
    pred_mentions_valid = check_mention_list_types(res_file, "Prediction")
    ref_mentions_valid = check_mention_list_types(ref_file, "Reference")
    if not (pred_mentions_valid and ref_mentions_valid):
        all_checks_passed = False

    # Check file structures
    print("\n-- File Structure Validation --")
    pred_structure = check_file_structure(res_file, "Prediction")
    ref_structure = check_file_structure(ref_file, "Reference")
    if not (pred_structure and ref_structure):
        all_checks_passed = False
    
    # Compare document IDs
    print("\n-- Document ID Comparison --")
    ids_match = compare_document_ids(res_file, ref_file)
    if not ids_match:
        all_checks_passed = False
        
    # Perform combined analysis if basic checks passed
    if pred_valid and ref_valid: # Only run if files are valid JSON
        print("\n-- Combined Data Analysis --")
        analyze_and_combine_data(res_file, "Prediction")
        analyze_and_combine_data(ref_file, "Reference")
    else:
        print("\n-- Skipping Combined Data Analysis due to invalid JSON files --")

    # Final result
    if all_checks_passed:
        # Check output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n✅ Created output directory: {output_dir}")
        print("\n=== All validations passed! You can proceed with scoring. ===")
    else:
        print("\n❌ Validation failed. Please fix the reported issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 