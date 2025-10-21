import logging 
import os

import numpy as np 
import pandas as pd 
import json
import re 

from ._utils import split_text, index_qas



def load_from_local(data_path, dataset):
    data_dir = os.path.join(data_path, dataset)
    
    if dataset == "HealthCareMagic":
        data_file = os.path.join(data_dir, "HealthCareMagic-100k.json")
        qa_pairs = load_HealthCareMagic(data_file)
        unique_queries, unique_contents, id2q, id2c, qa_id = index_qas(qa_pairs)
    elif dataset == "Enron":
        ...
    
    return {
        "uni_q": unique_queries, "uni_c": unique_contents, 
        "id2q": id2q, "id2c": id2c, "qa_id": qa_id
    }
        

def load_HealthCareMagic(data_file):
    # ["<human>:...\n<bot>:...", ...]
    with open(data_file, "r") as f:
        data = json.load(f)
    
    # process the data into a {"query": , "content": } style
    pattern = re.compile(r"<human>:\s*(.*?)\s*<bot>:\s*(.*)\s*$", re.DOTALL)
    transformed = []
    for raw in data:
        human, bot = split_text(raw, pattern)
        transformed.append({"query": human, "content": bot})
    return transformed


