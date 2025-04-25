import transformers
import torch
from typing import List, Dict, Optional, Union, Any

class HuggingFaceModel:
    """
    A class to handle individual Hugging Face model inference.
    """
    def __init__(self, model_id: str, max_new_tokens: int = 2048, device_map: str = "auto"):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        
        # Initialize model with appropriate settings
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device_map,
        )

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> Any:
        """
        Generate text based on input messages.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Optional parameter to override the default max_new_tokens
            
        Returns:
            The model's generated output
        """
        tokens = max_new_tokens or self.max_new_tokens
        outputs = self.pipeline(
            messages,
            max_new_tokens=tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return outputs
    
    def __str__(self) -> str:
        return f"HuggingFaceModel({self.model_id})"


class ModelManager:
    """
    A class to manage multiple LLM models and provide a unified interface.
    """
    AVAILABLE_MODELS = {
        "qwen": "Qwen/Qwen2.5-14B-Instruct",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "llama": "meta-llama/Llama-3.3-70B-Instruct"
    }
    
    def __init__(self, use_models: List[str] = None, load_all: bool = False):
        """
        Initialize the ModelManager.
        
        Args:
            use_models: List of model identifiers to load ("qwen", "deepseek", "llama")
            load_all: If True, load all available models
        """
        self.models = {}
        models_to_load = list(self.AVAILABLE_MODELS.keys()) if load_all else (use_models or ["qwen"])
        
        for model_name in models_to_load:
            if model_name in self.AVAILABLE_MODELS:
                print(f"Loading {model_name} model ({self.AVAILABLE_MODELS[model_name]})...")
                self.models[model_name] = HuggingFaceModel(self.AVAILABLE_MODELS[model_name])
            else:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.AVAILABLE_MODELS.keys())}")

    def get_model(self, model_name: str) -> HuggingFaceModel:
        """Get a specific model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def generate(self, messages: List[Dict[str, str]], model_name: str = None) -> Any:
        """
        Generate text using the specified or default model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Name of the model to use (if None, uses the first loaded model)
            
        Returns:
            The generated output from the model
        """
        if not self.models:
            raise RuntimeError("No models loaded. Please load at least one model.")
            
        if model_name is None:
            model_name = next(iter(self.models))
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
            
        return self.models[model_name].generate(messages)
    
    def generate_from_all(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate text from all loaded models.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}
        for model_name, model in self.models.items():
            print(f"Generating with {model_name}...")
            results[model_name] = model.generate(messages)
        return results
    
    def __str__(self) -> str:
        return f"ModelManager(loaded_models={list(self.models.keys())})"


# Example usage
if __name__ == "__main__":
    # Initialize with all models
    manager = ModelManager(load_all=True)
    
    # Create a sample prompt
    messages = [
        {"role": "user", "content": "What is document-level information extraction?"},
    ]
    
    # Generate from specific model
    result = manager.generate(messages, model_name="qwen")
    print(result)
    
    # Generate from all models
    all_results = manager.generate_from_all(messages)
    for model_name, result in all_results.items():
        print(f"\n--- {model_name.upper()} RESULT ---")
        print(result)