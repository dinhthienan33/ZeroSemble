import json
from typing import Dict, List, Any, Set, Tuple
from collections import Counter

def normalize_entity(entity: str) -> str:
    """Normalize entity text for comparison by lowercasing and stripping whitespace."""
    return entity.lower().strip()

def extract_entities_from_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract entities from model results.
    
    Args:
        results: Model outputs containing entity information
        
    Returns:
        List of normalized entity dictionaries
    """
    # This is a placeholder function that would extract entities from the model's outputs
    # The actual implementation depends on the exact structure of the model outputs
    entities = []
    try:
        # Try to parse the model output if it's a string
        if isinstance(results, str):
            results = json.loads(results)
        
        # Extract entities from parsed results
        for doc_id, doc_data in results.items():
            if 'entities' in doc_data:
                for entity in doc_data['entities']:
                    if isinstance(entity, dict) and 'mentions' in entity and 'type' in entity:
                        entities.append(entity)
    except:
        # Return empty list if parsing fails
        pass
    
    return entities

def extract_triples_from_results(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract triples from model results.
    
    Args:
        results: Model outputs containing triple information
        
    Returns:
        List of normalized triple dictionaries
    """
    # This is a placeholder function that would extract triples from the model's outputs
    triples = []
    try:
        # Try to parse the model output if it's a string
        if isinstance(results, str):
            results = json.loads(results)
        
        # Extract triples from parsed results
        for doc_id, doc_data in results.items():
            if 'triples' in doc_data:
                for triple in doc_data['triples']:
                    if isinstance(triple, dict) and 'head' in triple and 'relation' in triple and 'tail' in triple:
                        # Normalize the head and tail for consistent comparison
                        triple_copy = triple.copy()
                        triple_copy['head'] = normalize_entity(triple['head'])
                        triple_copy['tail'] = normalize_entity(triple['tail'])
                        triples.append(triple_copy)
    except:
        # Return empty list if parsing fails
        pass
    
    return triples

def ensemble_entities(model_outputs: Dict[str, Any], min_models: int = 2) -> List[Dict[str, Any]]:
    """
    Combine entity predictions from multiple models using a voting mechanism.
    
    Args:
        model_outputs: Dictionary mapping model names to their prediction outputs
        min_models: Minimum number of models that must agree on an entity
        
    Returns:
        List of entity dictionaries that meet the ensemble criteria
    """
    # Extract entities from each model's output
    model_entities = {}
    for model_name, output in model_outputs.items():
        model_entities[model_name] = extract_entities_from_results(output)
    
    # Count entity occurrences across models
    entity_counter = Counter()
    entity_type_map = {}
    entity_mentions_map = {}
    
    for model_name, entities in model_entities.items():
        for entity in entities:
            # Use the first mention as the representative for this entity
            if not entity.get('mentions'):
                continue
                
            entity_text = normalize_entity(entity['mentions'][0])
            entity_counter[entity_text] += 1
            
            # Store the most common entity type and all mentions
            if entity_text not in entity_type_map:
                entity_type_map[entity_text] = []
                entity_mentions_map[entity_text] = set()
                
            entity_type_map[entity_text].append(entity['type'])
            for mention in entity['mentions']:
                entity_mentions_map[entity_text].add(mention)
    
    # Select entities that meet the minimum vote threshold
    selected_entities = []
    for entity_text, count in entity_counter.items():
        if count >= min_models:
            # Determine the most common entity type
            type_counter = Counter(entity_type_map[entity_text])
            entity_type = type_counter.most_common(1)[0][0]
            
            # Create the selected entity entry
            selected_entities.append({
                "mentions": list(entity_mentions_map[entity_text]),
                "type": entity_type
            })
    
    return selected_entities

def ensemble_triples(model_outputs: Dict[str, Any], min_models: int = 2) -> List[Dict[str, str]]:
    """
    Combine triple predictions from multiple models using a voting mechanism.
    
    Args:
        model_outputs: Dictionary mapping model names to their prediction outputs
        min_models: Minimum number of models that must agree on a triple
        
    Returns:
        List of triple dictionaries that meet the ensemble criteria
    """
    # Extract triples from each model's output
    model_triples = {}
    for model_name, output in model_outputs.items():
        model_triples[model_name] = extract_triples_from_results(output)
    
    # Count triple occurrences across models
    triple_counter = Counter()
    relation_map = {}
    
    for model_name, triples in model_triples.items():
        for triple in triples:
            # Create a triple key by combining head and tail
            triple_key = (triple['head'], triple['tail'])
            triple_counter[triple_key] += 1
            
            # Store the relation for this head-tail pair
            if triple_key not in relation_map:
                relation_map[triple_key] = []
            relation_map[triple_key].append(triple['relation'])
    
    # Select triples that meet the minimum vote threshold
    selected_triples = []
    for triple_key, count in triple_counter.items():
        if count >= min_models:
            # Determine the most common relation for this head-tail pair
            relation_counter = Counter(relation_map[triple_key])
            relation = relation_counter.most_common(1)[0][0]
            
            # Create the selected triple entry
            selected_triples.append({
                "head": triple_key[0],
                "relation": relation,
                "tail": triple_key[1]
            })
    
    return selected_triples

def combine_models_output(model_outputs: Dict[str, Any], 
                         doc_id: str, 
                         title: str,
                         entity_min_votes: int = 2, 
                         triple_min_votes: int = 2) -> Dict[str, Any]:
    """
    Combine outputs from multiple models into a single result.
    
    Args:
        model_outputs: Dictionary mapping model names to their outputs
        doc_id: Document ID for the result
        title: Document title
        entity_min_votes: Minimum number of models that must agree on an entity
        triple_min_votes: Minimum number of models that must agree on a triple
        
    Returns:
        Combined extraction result
    """
    # Ensemble entities and triples
    combined_entities = ensemble_entities(model_outputs, min_models=entity_min_votes)
    combined_triples = ensemble_triples(model_outputs, min_models=triple_min_votes)
    
    # Create the final output structure
    result = {
        doc_id: {
            "title": title,
            "entities": combined_entities,
            "triples": combined_triples
        }
    }
    
    return result

def refine_triples_with_entities(triples: List[Dict[str, str]], entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Filter triples to only include those where both head and tail are in the entity list.
    
    Args:
        triples: List of triple dictionaries
        entities: List of entity dictionaries
        
    Returns:
        Filtered list of triples
    """
    # Extract all entity mentions
    entity_mentions = set()
    for entity in entities:
        for mention in entity.get('mentions', []):
            entity_mentions.add(normalize_entity(mention))
    
    # Filter triples to only include those with head and tail in entity_mentions
    filtered_triples = []
    for triple in triples:
        head = normalize_entity(triple['head'])
        tail = normalize_entity(triple['tail'])
        
        if head in entity_mentions and tail in entity_mentions:
            filtered_triples.append(triple)
    
    return filtered_triples 