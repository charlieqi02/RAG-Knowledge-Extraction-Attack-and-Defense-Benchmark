import os, yaml

from ._llm_engines import *
from ._llama_engines import *


def get_llm(model_name):
    # load model configs from yaml
    with open(os.environ["KEYS_PATH"], "r") as f:
        keys_cfg = yaml.safe_load(f)['llm']
    
    if model_name == "gpt4o-mini":
        model_kwargs = keys_cfg['azure']['gpt4o-mini']
        llm = AzureOpenAIEngine(model_name=model_kwargs['model_name'],
                                api_key=model_kwargs['api_key'],
                                azure_endpoint=model_kwargs['azure_endpoint'],
                                api_version=model_kwargs['api_version'])
    elif model_name == "llama3-8B-I":
        llm = LlamaEngine(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda:0")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return llm 
