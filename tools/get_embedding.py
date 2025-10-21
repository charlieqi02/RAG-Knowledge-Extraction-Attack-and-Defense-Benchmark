import logging, os, yaml

from ._embedding_models import *


def get_embedding(model_name, device="cuda"):
    # load model configs from yaml
    with open(os.environ["KEYS_PATH"], "r") as f:
        keys_cfg = yaml.safe_load(f)['embedding']
    
    
    if model_name == "MiniLM":
        embedding_model = MiniLMEmbeddings(device=device)
    else:
        raise NotImplementedError

    logging.info(f"Embedding model {model_name}, dim = {embedding_model.dim}")
    
    return embedding_model
    
    
    