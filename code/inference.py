import io
import json
import logging
import sys
from typing import Dict, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accurate_token_count(text: str, processor: CLIPProcessor) -> int:
    return len(processor.tokenizer(text).input_ids)

def chunk_text(text: str, processor: CLIPProcessor, max_tokens: int = 77) -> list:
    words = text.split()
    chunks, chunk, tokens = [], [], 0
    
    for word in words:
        new_token_count = accurate_token_count(word, processor)
        if tokens + new_token_count > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [word]
            tokens = new_token_count
        else:
            chunk.append(word)
            tokens += new_token_count
            
    if chunk:
        chunks.append(" ".join(chunk))
        
    return chunks

def model_fn(model_dir: str) -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_dir)
    return model, processor

def input_fn(request_body, request_content_type):
    assert request_content_type in ("application/json", "application/x-image"), f"{request_content_type} is an unknown type."
    
    content_type_map = {
        "application/json": lambda x: json.loads(x)["inputs"],
        "application/x-image": lambda x: Image.open(io.BytesIO(x))
    }
    
    return request_content_type, content_type_map[request_content_type](request_body)

def predict_fn(request_content_type_and_input_object, model_and_processor):
    model, processor = model_and_processor
    request_content_type, input_object = request_content_type_and_input_object
    
    def chunk_and_process_text(text):
        text_chunks = chunk_text(text, processor)
        features_sum, count = None, 0
        
        for text_chunk in text_chunks:
            inputs = processor(text=text_chunk, padding=True, return_tensors="pt").to(device)
            try:
                features = model.get_text_features(**inputs)
                if features_sum is None:
                    features_sum = torch.zeros_like(features).to(device)
                features_sum += features
                count += 1
            except RuntimeError as e:
                logger.error(f"Text chunk '{text_chunk}' of length {len(text_chunk)} caused an error: {e}")

        return (features_sum / count).detach()
        
    input_processors = {
        "application/x-image": lambda x: processor(images=x, padding=True, return_tensors="pt").to(device),
        "application/json": lambda x: chunk_and_process_text(x)
    }
    
    feature_getters = {
        "application/x-image": lambda x: model.get_image_features(**x),
        "application/json": lambda x: x 
    }
    
    inputs = input_processors[request_content_type](input_object)
    return feature_getters[request_content_type](inputs)
