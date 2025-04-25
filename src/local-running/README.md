# Local Running - Document-Level Information Extraction Pipeline

This directory contains code for running document-level information extraction locally using three powerful LLMs from Hugging Face. The pipeline implements a two-stage approach for robust extraction across multiple domains without domain-specific training.

## Quick Start

```bash
# Install dependencies
pip install torch transformers accelerate

# Run on a single document
python run_extraction.py --input sample_document.json --output ./results --verbose

# Run with specific models (faster, less VRAM required)
python run_extraction.py --input sample_document.json --output ./results --models qwen deepseek --verbose
```

## Installation

1. **System Requirements**:
   - Python 3.8+
   - CUDA-capable GPU with at least 24GB VRAM (for single model)
   - ~40GB+ VRAM for all three models (can use model offloading)
   - ~100GB disk space for model weights

2. **Dependencies**:

```bash
# Basic dependencies
pip install torch transformers accelerate tqdm

# For CUDA support (recommended)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

3. **Model Availability**:
   - Ensure you have access to the Hugging Face models:
     - `Qwen/Qwen2.5-14B-Instruct`
     - `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
     - `meta-llama/Llama-3.3-70B-Instruct`
   - Some models may require authentication with Hugging Face

## Usage Examples

### Process a Single Document

```bash
python run_extraction.py --input path/to/document.json --output ./results --verbose
```

### Process Multiple Documents in a Directory

```bash
python run_extraction.py --input path/to/documents_dir/ --output ./results --verbose
```

### Use Only Specific Models (Faster/Less Memory)

```bash
# Use only Qwen model (fastest)
python run_extraction.py --input document.json --output ./results --models qwen --verbose

# Use Qwen and DeepSeek models
python run_extraction.py --input document.json --output ./results --models qwen deepseek --verbose
```

### Adjust Voting Thresholds

```bash
# Require only one model to agree on entities but two for triples
python run_extraction.py --input document.json --output ./results --entity-votes 1 --triple-votes 2 --verbose
```

### Use Single-Stage Pipeline (Faster but Less Accurate)

```bash
python run_extraction.py --input document.json --output ./results --single-stage --verbose
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input JSON file or directory | Required |
| `--output` | Path to output directory for results | Required |
| `--models` | Models to use (qwen, deepseek, llama) | All three |
| `--single-stage` | Use single-stage extraction | False |
| `--entity-votes` | Min models that must agree on entity | 2 |
| `--triple-votes` | Min models that must agree on triple | 2 |
| `--verbose` | Print detailed information | False |

## Input Format

The input JSON should have this structure:

```json
{
  "id": "doc_1",
  "title": "Document Title",
  "text": "Document content...",
  "domain": "medical"
}
```

Multiple documents can be provided as an array of objects.

## Output Format

```json
{
  "doc_1": {
    "title": "Document Title",
    "entities": [
      {
        "mentions": ["Entity Text", "Alternative Mention"],
        "type": "NER Label"
      }
    ],
    "triples": [
      {
        "head": "Entity 1",
        "relation": "Relationship",
        "tail": "Entity 2"
      }
    ]
  }
}
```

## Troubleshooting

- **Out of Memory Errors**: Try using fewer models with `--models qwen`
- **Slow Processing**: The first run will download models (~100GB). Subsequent runs will be faster.
- **Model Loading Issues**: Ensure you've logged in with `huggingface-cli login` if models require authentication

## File Structure

- `run_extraction.py` - Main script for running the pipeline
- `models/llm.py` - Model management for Hugging Face models
- `utils/ensemble.py` - Utilities for combining model outputs
- `inference/run_inference.py` - Inference logic and prompt formatting 