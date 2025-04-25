import json
import os
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from statistics import mean, median, stdev

def load_data(file_path):
    """Loads the JSON data file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {file_path}")
        return data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def extract_document_data(data):
    """Extract document data from JSON, handling different formats."""
    doc_data = {}
    global_metadata = {}
    
    # Check for specific metadata keys at top level
    if isinstance(data, dict):
        # Check for label sets
        label_keys = ['NER_label_set', 'RE_label_set', 'entity_label_set', 'relation_label_set']
        for key in label_keys:
            if key in data:
                global_metadata[key] = data[key]
                
        print("Input data is a dictionary. Processing documents...")
        for key, value in data.items():
            # Skip keys we already processed as metadata
            if key in global_metadata:
                continue
                
            if isinstance(value, dict):
                # For test data, look for entries with document/doc, text, or content fields
                if 'document' in value or 'doc' in value or 'text' in value or 'content' in value or 'title' in value:
                    doc_data[key] = value
                # Check if this might be a metadata key
                elif key.endswith('_set') or key.endswith('_labels'):
                    global_metadata[key] = value
    elif isinstance(data, list):
        print("Input data is a list. Processing list items as documents...")
        for index, item in enumerate(data):
            if isinstance(item, dict):
                # Check if this is a metadata item first
                if 'label_set' in item or 'labels' in item:
                    for k, v in item.items():
                        global_metadata[k] = v
                # Otherwise process as document
                elif 'document' in item or 'doc' in item or 'text' in item or 'content' in item or 'title' in item:
                    doc_id = item.get('id', f"item_{index}")
                    if doc_id in doc_data:
                        print(f"Warning: Duplicate document ID '{doc_id}' found. Overwriting previous entry.")
                    doc_data[doc_id] = item
            else:
                print(f"Warning: Item at index {index} in list is not a dictionary. Skipping.")
    else:
        print("Error: Expected input data to be a dictionary or a list.")
        return None, None
            
    print(f"Extracted {len(doc_data)} document entries and {len(global_metadata)} metadata items.")
    return doc_data, global_metadata

def plot_distribution(counts, title, xlabel, ylabel, top_n=20, filename="plot.png"):
    """Generates and saves a bar plot for the top N items in a Counter."""
    if not counts:
        print(f"No data to plot for {title}")
        return
        
    labels, values = zip(*counts.most_common(top_n))
    
    plt.figure(figsize=(12, max(6, len(labels) * 0.4)))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values, align='center')
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.title(f'{title} (Top {top_n})')
    
    # Add counts on bars
    for index, value in enumerate(values):
        plt.text(value, index, f' {value}', va='center')
        
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()

def plot_histogram(values, title, xlabel, ylabel, bins=20, filename="histogram.png"):
    """Generates and saves a histogram."""
    if not values:
        print(f"No data to plot for {title}")
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Histogram saved to {filename}")
    except Exception as e:
        print(f"Error saving histogram {filename}: {e}")
    plt.close()

def extract_entity_relation_data(doc_data):
    """Extract entity and relation data from documents if available."""
    entity_types = Counter()
    relation_types = Counter()
    entity_per_doc = defaultdict(Counter)
    relation_per_doc = defaultdict(Counter)
    
    for doc_id, doc in doc_data.items():
        # Try to extract entities
        entities = doc.get('entities', [])
        for entity in entities:
            entity_type = entity.get('type', 'Unknown')
            entity_types[entity_type] += 1
            entity_per_doc[doc_id][entity_type] += 1
            
        # Try to extract relations
        relations = doc.get('triples', doc.get('relations', []))
        for relation in relations:
            relation_type = relation.get('relation', relation.get('type', 'Unknown'))
            relation_types[relation_type] += 1
            relation_per_doc[doc_id][relation_type] += 1
    
    return {
        'entity_types': entity_types,
        'relation_types': relation_types,
        'entity_per_doc': entity_per_doc,
        'relation_per_doc': relation_per_doc
    }

def analyze_label_sets(metadata):
    """Analyze label sets (NER and RE) if available in metadata."""
    ner_labels = []
    re_labels = []
    
    # Look for label sets with different possible key names
    for key, value in metadata.items():
        if isinstance(value, list):
            key_lower = key.lower()
            if 'ner' in key_lower or 'entity' in key_lower:
                ner_labels.extend(value)
            elif 're' in key_lower or 'relation' in key_lower:
                re_labels.extend(value)
    
    # Remove duplicates while preserving order
    ner_labels = list(dict.fromkeys(ner_labels))
    re_labels = list(dict.fromkeys(re_labels))
    
    return {
        'ner_labels': ner_labels,
        're_labels': re_labels
    }

def save_doc_stats_to_csv(doc_stats, output_dir):
    """Save document statistics to a CSV file."""
    csv_path = os.path.join(output_dir, "document_statistics.csv")
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['doc_id', 'domain', 'char_length', 'word_count', 'sentence_count', 'title_length', 'title', 'entity_count', 'relation_count'])
            
            # Write data rows
            for doc_id, stats in doc_stats.items():
                writer.writerow([
                    doc_id,
                    stats.get('domain', 'Unknown'),
                    stats.get('char_length', 0),
                    stats.get('word_count', 0),
                    stats.get('sentence_count', 0),
                    stats.get('title_length', 0),
                    stats.get('title', ''),
                    stats.get('entity_count', 0),
                    stats.get('relation_count', 0)
                ])
        print(f"Document statistics saved to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error saving document statistics to CSV: {e}")
        return None

def find_document_info(doc_id, doc_stats):
    """Find and display detailed information about a specific document."""
    if doc_id not in doc_stats:
        print(f"Document ID '{doc_id}' not found in the dataset.")
        return
    
    stats = doc_stats[doc_id]
    
    print(f"\n=== Document {doc_id} Information ===")
    print(f"Domain: {stats.get('domain', 'Unknown')}")
    
    title = stats.get('title', '')
    if title:
        print(f"Title: \"{title}\"")
        print(f"Title Length: {stats.get('title_length', 0)} characters")
    else:
        print("Title: Not available")
    
    print(f"\nDocument Content:")
    print(f"Character Length: {stats.get('char_length', 0)}")
    print(f"Word Count: {stats.get('word_count', 0)}")
    print(f"Sentence Count: {stats.get('sentence_count', 0)}")
    
    # Display entity and relation info if available
    entity_count = stats.get('entity_count', 0)
    if entity_count > 0:
        print(f"\nEntities: {entity_count}")
        if 'entity_types' in stats:
            print("Entity Types:")
            for entity_type, count in stats['entity_types'].items():
                print(f"  - {entity_type}: {count}")
    
    relation_count = stats.get('relation_count', 0)
    if relation_count > 0:
        print(f"\nRelations: {relation_count}")
        if 'relation_types' in stats:
            print("Relation Types:")
            for relation_type, count in stats['relation_types'].items():
                print(f"  - {relation_type}: {count}")
    
    excerpt = stats.get('excerpt', '')
    if excerpt:
        print(f"\nExcerpt: \"{excerpt}\"")
    
    return stats

def analyze_test_data(doc_data, metadata, output_dir, target_doc_id=None):
    """Analyzes test data focusing on domains, titles, and document content."""
    if not doc_data:
        print("No document data found to analyze.")
        return

    num_documents = len(doc_data)
    
    # Counters for domain and title analysis
    domains = Counter()
    title_lengths = []
    title_words = Counter()
    title_word_pattern = re.compile(r'\b\w+\b')
    
    # Store title info
    titles = []  # Store (length, title_text, doc_id) tuples
    
    # Document text analysis counters
    doc_lengths = []
    doc_word_counts = []
    sentences_per_doc = []
    sentence_pattern = re.compile(r'[.!?]+')
    word_pattern = re.compile(r'\b\w+\b')
    
    # Store document content info
    doc_contents = []  # Store (length, word_count, doc_text, doc_id) tuples
    
    # Dictionary to store stats for each document - used for CSV export and document lookups
    doc_stats = {}
    
    # Initialize variables that might be referenced later
    shortest_doc_info = None
    longest_doc_info = None
    wordiest_doc_info = None
    least_wordy_doc_info = None
    avg_doc_length = 0
    avg_word_count = 0
    
    # Extract entity and relation info if available
    er_data = extract_entity_relation_data(doc_data)
    entity_types = er_data['entity_types']
    relation_types = er_data['relation_types']
    entity_per_doc = er_data['entity_per_doc']
    relation_per_doc = er_data['relation_per_doc']
    
    # Analyze label sets from metadata
    label_data = analyze_label_sets(metadata)
    ner_labels = label_data['ner_labels']
    re_labels = label_data['re_labels']

    print("\nAnalyzing test documents...")
    # Iterate through documents
    for doc_id, doc in doc_data.items(): 
        # Initialize stats for this document
        doc_stats[doc_id] = {'doc_id': doc_id}
        
        # Extract domain
        domain = doc.get('domain', 'Unknown')
        domains[domain] += 1
        doc_stats[doc_id]['domain'] = domain
        
        # Extract and analyze title
        title = doc.get('title', '')
        if title:
            title_length = len(title)
            title_lengths.append(title_length)
            titles.append((title_length, title, doc_id))
            words = title_word_pattern.findall(title.lower())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    title_words[word] += 1
            
            doc_stats[doc_id]['title'] = title
            doc_stats[doc_id]['title_length'] = title_length
        
        # Extract and analyze document text
        doc_text = doc.get('document', doc.get('text', doc.get('content', '')))
        if doc_text and isinstance(doc_text, str):
            doc_length = len(doc_text)
            doc_lengths.append(doc_length)
            doc_stats[doc_id]['char_length'] = doc_length
            
            words = word_pattern.findall(doc_text.lower())
            word_count = len(words)
            doc_word_counts.append(word_count)
            doc_stats[doc_id]['word_count'] = word_count
            
            # Generate an excerpt
            excerpt = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            doc_stats[doc_id]['excerpt'] = excerpt
            
            # Store the document info
            doc_contents.append((doc_length, word_count, doc_text, doc_id))
            
            # Count sentences
            sentences = sentence_pattern.split(doc_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences)
            sentences_per_doc.append(sentence_count)
            doc_stats[doc_id]['sentence_count'] = sentence_count
        
        # Add entity and relation counts for this document
        if doc_id in entity_per_doc:
            entity_count = sum(entity_per_doc[doc_id].values())
            doc_stats[doc_id]['entity_count'] = entity_count
            doc_stats[doc_id]['entity_types'] = entity_per_doc[doc_id]
        
        if doc_id in relation_per_doc:
            relation_count = sum(relation_per_doc[doc_id].values())
            doc_stats[doc_id]['relation_count'] = relation_count
            doc_stats[doc_id]['relation_types'] = relation_per_doc[doc_id]

    # Check for specific document lookup
    if target_doc_id:
        return find_document_info(target_doc_id, doc_stats)

    # Save document statistics to CSV
    csv_path = save_doc_stats_to_csv(doc_stats, output_dir)

    # --- Print Reports --- 
    print("\n=== Test Data Analysis Report ===")

    # Basic Stats
    print("\n--- Basic Statistics ---")
    print(f"Total Test Documents: {num_documents}") 
    
    # Document Content Stats
    if doc_lengths and doc_contents:  # Only process if we have document content
        print("\n--- Document Content Statistics ---")
        print(f"Documents with Text Content: {len(doc_lengths)} ({len(doc_lengths)/num_documents*100:.1f}%)")
        
        avg_doc_length = mean(doc_lengths)
        print(f"Average Document Length: {avg_doc_length:.1f} characters")
        print(f"Median Document Length: {median(doc_lengths)} characters")
        if len(doc_lengths) > 1:
            print(f"Standard Deviation: {stdev(doc_lengths):.1f} characters")
        
        # Get shortest and longest documents by character length
        shortest_doc_info = min(doc_contents, key=lambda x: x[0])
        longest_doc_info = max(doc_contents, key=lambda x: x[0])
        
        # Prepare excerpts (first 100 chars for shortest, first 200 for longest)
        shortest_excerpt = shortest_doc_info[2][:100] + "..." if len(shortest_doc_info[2]) > 100 else shortest_doc_info[2]
        longest_excerpt = longest_doc_info[2][:200] + "..." if len(longest_doc_info[2]) > 200 else longest_doc_info[2]
        
        # Display shortest document
        print(f"Shortest Document: {shortest_doc_info[0]} characters, {shortest_doc_info[1]} words")
        print(f"  - Doc ID: {shortest_doc_info[3]}")
        print(f"  - Excerpt: \"{shortest_excerpt}\"")
        
        # Display longest document
        print(f"Longest Document: {longest_doc_info[0]} characters, {longest_doc_info[1]} words")
        print(f"  - Doc ID: {longest_doc_info[3]}")
        print(f"  - Excerpt: \"{longest_excerpt}\"")
        
        # Get most and least wordy documents (by word count)
        wordiest_doc_info = max(doc_contents, key=lambda x: x[1])
        least_wordy_doc_info = min(doc_contents, key=lambda x: x[1])
        
        # Display wordiest document
        print(f"\nDocument with Most Words: {wordiest_doc_info[1]} words, {wordiest_doc_info[0]} characters")
        print(f"  - Doc ID: {wordiest_doc_info[3]}")
        wordiest_excerpt = wordiest_doc_info[2][:200] + "..." if len(wordiest_doc_info[2]) > 200 else wordiest_doc_info[2]
        print(f"  - Excerpt: \"{wordiest_excerpt}\"")
        
        # Only show least wordy if different from shortest
        if least_wordy_doc_info[3] != shortest_doc_info[3]:
            least_wordy_excerpt = least_wordy_doc_info[2][:100] + "..." if len(least_wordy_doc_info[2]) > 100 else least_wordy_doc_info[2]
            print(f"\nDocument with Fewest Words: {least_wordy_doc_info[1]} words, {least_wordy_doc_info[0]} characters")
            print(f"  - Doc ID: {least_wordy_doc_info[3]}")
            print(f"  - Excerpt: \"{least_wordy_excerpt}\"")
        
        if doc_word_counts:
            avg_word_count = mean(doc_word_counts)
            print(f"\nAverage Word Count: {avg_word_count:.1f} words")
            print(f"Median Word Count: {median(doc_word_counts)} words")
            if len(doc_word_counts) > 1:
                print(f"Standard Deviation: {stdev(doc_word_counts):.1f} words")
        
        if sentences_per_doc:
            print(f"\nAverage Sentences per Document: {mean(sentences_per_doc):.1f}")
            print(f"Median Sentences per Document: {median(sentences_per_doc)}")
            if len(sentences_per_doc) > 1:
                print(f"Standard Deviation: {stdev(sentences_per_doc):.1f} sentences")
            print(f"Min Sentences: {min(sentences_per_doc)}")
            print(f"Max Sentences: {max(sentences_per_doc)}")
    else:
        print("\nNo document content found for analysis.")

    # Domain Analysis
    print("\n--- Domain Analysis ---")
    print(f"Total Unique Domains: {len(domains)}")
    print("Domain Distribution:")
    for domain, count in domains.most_common():
        print(f"  - {domain}: {count} documents ({count/num_documents*100:.1f}%)")
    
    # Title Analysis
    print("\n--- Title Analysis ---")
    if title_lengths:
        print(f"Documents with Titles: {len(title_lengths)} ({len(title_lengths)/num_documents*100:.1f}%)")
        print(f"Average Title Length: {mean(title_lengths):.1f} characters")
        print(f"Median Title Length: {median(title_lengths)} characters")
        if len(title_lengths) > 1:
            print(f"Standard Deviation: {stdev(title_lengths):.1f} characters")
        
        # Get shortest and longest titles
        shortest_title_info = min(titles, key=lambda x: x[0])
        longest_title_info = max(titles, key=lambda x: x[0])
        
        # Display shortest title
        print(f"Shortest Title: {shortest_title_info[0]} characters")
        print(f"  - Doc ID: {shortest_title_info[2]}")
        print(f"  - Title: \"{shortest_title_info[1]}\"")
        
        # Display longest title
        print(f"Longest Title: {longest_title_info[0]} characters")
        print(f"  - Doc ID: {longest_title_info[2]}")
        print(f"  - Title: \"{longest_title_info[1]}\"")
        
        print("\nMost Common Words in Titles (excluding short words):")
        for word, count in title_words.most_common(15):
            print(f"  - '{word}': {count}")
    else:
        print("No title information found in documents")
    
    # Entity Type Analysis
    print("\n--- Entity Type Analysis ---")
    if entity_types:
        print(f"Total Entity Types: {len(entity_types)}")
        print(f"Total Entity Mentions: {sum(entity_types.values())}")
        #print("\nEntity Type Distribution:")
        # for entity_type, count in entity_types.most_common()[:10]:
        #     print(f"  - {entity_type}: {count} ({count/sum(entity_types.values())*100:.1f}%)")
    else:
        print("No entity type information found in documents")
        
    # Relation Type Analysis
    print("\n--- Relation Type Analysis ---")
    if relation_types:
        print(f"Total Relation Types: {len(relation_types)}")
        print(f"Total Relations: {sum(relation_types.values())}")
        # print("\nRelation Type Distribution:")
        # for rel_type, count in relation_types.most_common():
        #     print(f"  - {rel_type}: {count} ({count/sum(relation_types.values())*100:.1f}%)")
    else:
        print("No relation type information found in documents")
    
    # Label Set Analysis from Metadata
    print("\n--- Label Set Analysis from Metadata ---")
    if ner_labels:
        print(f"NER Labels in Metadata ({len(ner_labels)}):")
        for i, label in enumerate(ner_labels):
            print(f"  - {label}")
    else:
        print("No NER labels found in metadata")
        
    if re_labels:
        print(f"\nRE Labels in Metadata ({len(re_labels)}):")
        for i, label in enumerate(re_labels):
            print(f"  - {label}")
    else:
        print("No RE labels found in metadata")
        
    print("==================")
    
    # Print CSV path info
    if csv_path:
        print(f"\nDetailed document statistics saved to: {csv_path}")
        if wordiest_doc_info:  # Only print if we found a wordiest document
            print(f"Most Words: Doc ID {wordiest_doc_info[3]} with {wordiest_doc_info[1]} words")
        if doc_lengths:  # Only print if we have document lengths
            print(f"Average Document Length: {avg_doc_length:.1f} characters")
        if doc_word_counts:  # Only print if we have word counts
            print(f"Average Word Count: {avg_word_count:.1f} words")

    # --- Generate Plots --- 
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Document length plots
    if doc_lengths:
        plot_histogram(doc_lengths, 
                      title="Document Length Distribution", 
                      xlabel="Document Length (Characters)", 
                      ylabel="Frequency", 
                      bins=min(30, max(10, len(set(doc_lengths)) // 10)),
                      filename=os.path.join(output_dir, "document_length_histogram.png"))
        
        if doc_word_counts:
            plot_histogram(doc_word_counts, 
                          title="Document Word Count Distribution", 
                          xlabel="Word Count", 
                          ylabel="Frequency", 
                          bins=min(30, max(10, len(set(doc_word_counts)) // 10)),
                          filename=os.path.join(output_dir, "document_word_count_histogram.png"))
        
        if sentences_per_doc:
            plot_histogram(sentences_per_doc, 
                          title="Sentences per Document Distribution", 
                          xlabel="Sentence Count", 
                          ylabel="Frequency", 
                          bins=min(30, max(10, len(set(sentences_per_doc)) // 2)),
                          filename=os.path.join(output_dir, "sentence_count_histogram.png"))
    
    # Domain plot
    if domains:
        plot_distribution(domains, 
                          title="Test Document Domain Distribution", 
                          xlabel="Domain", 
                          ylabel="Number of Documents", 
                          top_n=len(domains) if len(domains) <= 30 else 30,
                          filename=os.path.join(output_dir, "domain_distribution.png"))
    
    # Title length histogram
    if title_lengths:
        plot_histogram(title_lengths, 
                       title="Title Length Distribution", 
                       xlabel="Title Length (Characters)", 
                       ylabel="Frequency", 
                       bins=min(30, max(10, len(set(title_lengths)))),
                       filename=os.path.join(output_dir, "title_length_histogram.png"))
        
        # Common words in titles
        plot_distribution(title_words, 
                          title="Common Words in Document Titles", 
                          xlabel="Word", 
                          ylabel="Frequency", 
                          top_n=30,
                          filename=os.path.join(output_dir, "title_words_distribution.png"))
    
    # Entity and relation type plots
    if entity_types:
        plot_distribution(entity_types,
                          title="Entity Type Distribution",
                          xlabel="Entity Type",
                          ylabel="Count",
                          top_n=min(30, len(entity_types)),
                          filename=os.path.join(output_dir, "entity_type_distribution.png"))
    
    if relation_types:
        plot_distribution(relation_types,
                          title="Relation Type Distribution",
                          xlabel="Relation Type",
                          ylabel="Count",
                          top_n=min(30, len(relation_types)),
                          filename=os.path.join(output_dir, "relation_type_distribution.png"))

def main():
    parser = argparse.ArgumentParser(description="Analyze test data JSON files focusing on domains, titles, document content, and NER/RE labels.")
    parser.add_argument('input_file', nargs='?', default="attempts/deepseek/results.json", help='Path to the test JSON file.')
    parser.add_argument('-o', '--output_dir', default='attempts/deepseek/analysis', help='Directory to save generated plots.')
    parser.add_argument('-d', '--doc_id', help='Look up information for a specific document ID.')
    
    args = parser.parse_args()
    
    data = load_data(args.input_file)
    if data is None:
        return
        
    doc_data, metadata = extract_document_data(data)
    
    if doc_data is not None:
        analyze_test_data(doc_data, metadata, args.output_dir, args.doc_id)
    else:
        print("\nAborting analysis due to data extraction errors.")

if __name__ == "__main__":
    main() 
