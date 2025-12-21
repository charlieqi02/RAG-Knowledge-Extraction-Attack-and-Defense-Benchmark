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
    elif model_name == "gpt4o":
        model_kwargs = keys_cfg['azure']['gpt4o']
        llm = AzureOpenAIEngine(model_name=model_kwargs['model_name'],
                                api_key=model_kwargs['api_key'],
                                azure_endpoint=model_kwargs['azure_endpoint'],
                                api_version=model_kwargs['api_version'])
    elif model_name == "llama3-8B-I":
        llm = LlamaEngine(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda:0")
    elif model_name == "qwen2.5-7B-I":
        llm = LlamaEngine(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:0")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return llm 





if __name__ == "__main__":
    import os
    import sys
    pwd = "/home/charq/Lab-UO/Projects/Extraction-AD-Pipeline"
    os.environ["PYTHONPATH"] = pwd + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["KEYS_PATH"] = pwd + "/keys.yaml"
    sys.path.append(pwd)
    
    # Example usage gpt4o
    engine = get_llm("qwen2.5-7B-I")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = engine(messages)
    print(response)
    