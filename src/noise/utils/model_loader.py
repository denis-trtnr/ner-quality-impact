from typing import Dict, Any
from transformers import pipeline
import torch
import gensim.downloader as api

# model cache
LOADED_MODELS: Dict[str, Any] = {}

def load_static_embedding_model(model_path: str):
    """Loads a static embedding model (e.g., GloVe) and caches it."""
    if model_path not in LOADED_MODELS:
        print(f"Loading static embedding model: {model_path}")
        LOADED_MODELS[model_path] = api.load(model_path)
    return LOADED_MODELS[model_path]

def load_contextual_embedding_model(model_name: str):
    """Loads a Masked-Language-Model from Hugging Face and caches it."""
    if model_name not in LOADED_MODELS:
        print(f"Loading contextual model: {model_name}")
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        LOADED_MODELS[model_name] = pipeline('fill-mask', model=model_name, device=device, top_k=30)
    return LOADED_MODELS[model_name]