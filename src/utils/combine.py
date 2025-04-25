import json
import os
from collections import defaultdict

def read_json_file(file_path):
    """Reads a JSON file and returns its content, handling errors."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}. Skipping.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                print(f"Warning: File content is not a dictionary - {file_path}. Skipping.")
                return None
            return data
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format - {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: Error reading file {file_path}: {str(e)}. Skipping.")
        return None

def combine_documents(input_files):
    """Combines entities and triples for each document ID across multiple files, deduplicating within each document."""
    combined_docs = defaultdict(lambda: {
        'title': None, 
        'entities': set(),  # Store hashable representations for deduplication
        'triples': set()    # Store hashable representations for deduplication
    })
    valid_input_files = []
    processed_doc_ids = set()

    print("Processing input files...")
    for file_path in input_files:
        data = read_json_file(file_path)
        if data:
            print(f"  - Processing {os.path.basename(file_path)}")
            valid_input_files.append(os.path.basename(file_path))
            for doc_id, doc in data.items():
                processed_doc_ids.add(doc_id)
                
                # Store title (use first encountered)
                if combined_docs[doc_id]['title'] is None:
                    combined_docs[doc_id]['title'] = doc.get('title', f'Title not found in source files for {doc_id}')

                # Add entities (deduplicated by set)
                entities = doc.get('entities', [])
                for entity in entities:
                    try:
                        # Use frozenset for mentions to handle order difference
                        mentions_key = frozenset(entity.get('mentions', []))
                        entity_type = entity.get('type', '')
                        entity_key = (mentions_key, entity_type)
                        combined_docs[doc_id]['entities'].add(entity_key)
                    except TypeError:
                        print(f"    Warning: Skipping entity in doc {doc_id} due to unhashable mention: {entity.get('mentions')}")
                
                # Add triples (deduplicated by set)
                triples = doc.get('triples', [])
                for triple in triples:
                    try:
                        head = triple.get('head', '')
                        relation = triple.get('relation', '')
                        tail = triple.get('tail', '')
                        triple_key = (head, relation, tail)
                        combined_docs[doc_id]['triples'].add(triple_key)
                    except TypeError:
                        print(f"    Warning: Skipping triple in doc {doc_id} due to unhashable part: {triple}")

    # Convert sets back to lists of dictionaries for the final output
    final_output_data = {}
    total_entities_after = 0
    total_triples_after = 0
    print("\nConsolidating results...")
    for doc_id, combined_info in combined_docs.items():
        final_entities = []
        for entity_key in combined_info['entities']:
            mentions_set, entity_type = entity_key
            final_entities.append({
                'mentions': sorted(list(mentions_set)),  # Store consistently sorted
                'type': entity_type
            })
        
        final_triples = []
        for triple_key in combined_info['triples']:
            head, relation, tail = triple_key
            final_triples.append({
                'head': head,
                'relation': relation,
                'tail': tail
            })
            
        final_output_data[doc_id] = {
            'title': combined_info['title'],
            'entities': sorted(final_entities, key=lambda x: (x['type'], x['mentions'][0] if x['mentions'] else '')),  # Sort for consistency
            'triples': sorted(final_triples, key=lambda x: (x['head'], x['relation'], x['tail']))  # Sort for consistency
        }
        total_entities_after += len(final_entities)
        total_triples_after += len(final_triples)

    report = {
        "processed_files": valid_input_files,
        "unique_documents_combined": len(processed_doc_ids),
        "total_unique_entities_across_docs": total_entities_after,
        "total_unique_triples_across_docs": total_triples_after
    }

    return report, final_output_data

def save_combined_data(data, output_file):
    """Saves the combined document data to a JSON file."""
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Combined document data successfully saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving combined data to {output_file}: {str(e)}")

def print_report(report):
    """Prints the summary report for document combination."""
    print("\n=== Document Combination Report ===")
    print(f"Processed Files ({len(report['processed_files'])}):")
    for filename in report['processed_files']:
        print(f"  - {filename}")
    print(f"Unique Documents Combined: {report['unique_documents_combined']}")
    print(f"Total Unique Entities (across all docs): {report['total_unique_entities_across_docs']}")
    print(f"Total Unique Triples (across all docs): {report['total_unique_triples_across_docs']}")
    print("=================================")

# Example usage in Jupyter Notebook:
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_files = [
        "deepseek/results.json",
        "llama-3.3-70b-versatile/results.json",
        "qwen-2.5/results.json",
        # Add more files as needed
    ]
    
    output_file = "ensemble1/results.json"  # Change this to your desired output path
    
    # Process the files
    report, combined_data = combine_documents(input_files)
    
    # Print the report
    print_report(report)
    
    # Save the output
    save_combined_data(combined_data, output_file)
    
    # The combined data is also available in the combined_data variable