"""Implementation for Dynamic Greedy Embedding Attack (DGEA)."""
import logging
from queue import Queue
import random
import re
import os
from urllib import response

import torch
import torch.nn as nn
import numpy as np
from langchain.prompts import ChatPromptTemplate
import pandas as pd

from .base import KnowExAttack
from tools.attacks import extract_indexes
from tools.get_embedding import get_embedding
from tools.get_llm import get_llm
from tools.parse_response import parse_repeat_response

copybreak_attack = ["CopyBreak"]


class CopyBreak(KnowExAttack):
    """
    Feedback-Guided Agent Attack (CopyBreak).
    Reference: https://arxiv.org/pdf/2411.14110
    """

    def __init__(self, args):
        """
        parameters:
        -----------
        """
        super().__init__(args)

        # Overall
        # if [0, 1], then possibility of exploitation; 
        # if > 1,    then # of exploitation and follow a exploration.
        self.attack_template = args.attack_template
        
        self.exchange_rate = args.exchange_rate # default 11
        self.attack_llm = get_llm(args.llm_model)   # explore: generate random text; exploit: generate head and tail
        
        # Exploration 
        self.extracted_chunks = set()   # extracted (chunk, emb) pairs
        self.extracted_embs = list()
        
        self.embedding_model = get_embedding(args.emb_model)    # for calculate extracted chunk & generated text embedding 
        self.embedding_dim = self.embedding_model.dim
        
        self.sim_thresh = args.sim_thresh  # sim threshold, sim(gen_v, extract_chunks) < threshold 
        self.iterations = args.iterations # limit number of tries for each exploration 
        self.explore_template = args.explore_template # for generate text far from extracted chunks
        self.explore_temp = args.explore_temperature
        
        # Exploitation
        self.num_of_reason = args.num_of_reason  # number of head/tail reasoning queries to generate based on an anchor chunk
        self.unexploited_chunks = set()      # extracted and unexploited (chunk)
        self.reasoning_queries = Queue()  # store head/tail reasoning queries
        
        self.exploit_template = args.exploit_template # for generate head and tail based on anchor chunk
        self.exploit_temp = args.exploit_temperature
        
        
        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.attack_template), "r") as f:
            self.attack_template = f.read()
        with open(os.path.join(prompt_dir, self.explore_template), "r") as f:
            self.explore_template = f.read()
        with open(os.path.join(prompt_dir, self.exploit_template), "r") as f:
            self.exploit_template = f.read()


    def get_query(self, query_id):
        """Generate query based on exploration-exploitation strategy."""
        if query_id == 0:
            info = self._exploration(first_time=True)
        else:
            if self.exchange_rate < 1:
                info = self._exploration() if random.random() < self.exchange_rate else self._exploitation() 
            else:
                info = self._exploration() if query_id % int(self.exchange_rate) == 0 \
                    or len(self.unexploited_chunks) == 0 else self._exploitation()

        logging.info(f"Query info: {info}")
        query = self.attack_template.replace("<info>", info)
        return query


    def _exploration(self, first_time=False):
        """Curiosity-driven explore: far from current extracted chunk as possible."""       
        logging.info("Exploration: generating diverse information.")
        logging.info(f"Exploration: current extracted chunks num={len(self.extracted_chunks)}")
        most_dissim_info = ""
        min_sim_score = np.inf
        
        prompt = ChatPromptTemplate.from_template(self.explore_template).format(context="\n".join(self.extracted_chunks))
        message = [{"role": "user", "content": prompt}]
        
        if first_time or len(self.unexploited_chunks) == 0:
            info = self.attack_llm(message, temperature=self.explore_temp)
            return info

        for i in range(self.iterations):
            info = self.attack_llm(message, temperature=self.explore_temp)
            
            emb = self.embedding_model._embed(info)
            embedding_space = torch.tensor(self.extracted_embs)  # (N, D)
            emb_tensor = torch.tensor(emb).unsqueeze(0)  # (1, D)
            # calculate most similar cosin score
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_scores = cos(emb_tensor.repeat(embedding_space.size(0), 1), embedding_space)  # (N,)
            max_score = torch.max(cos_scores).item()
            
            if max_score < self.sim_thresh:
                most_dissim_info = info
                min_sim_score = max_score
                break

            if max_score < min_sim_score:
                most_dissim_info = info
                min_sim_score = max_score

        logging.info(f"Exploration: selected info with sim score {min_sim_score}")
        return most_dissim_info
        
    
    def _exploitation(self):
        """Reasoning-based exploitation: close to current extracted chunk, backward-forward reasoning."""
        if self.reasoning_queries.qsize() > 0:
            logging.info(f"Exploitation: (rest={self.reasoning_queries.qsize()}) using stored reasoning query.")
            return self.reasoning_queries.get()

        logging.info(f"Exploitation: (rest=0) generating new reasoning queries based on an anchor chunk.")
        logging.info(f"Exploitation: current unexploited chunks num={len(self.unexploited_chunks)}")
        anchor_chunk = random.sample(list(self.unexploited_chunks), 1)[0]
        self.unexploited_chunks.remove(anchor_chunk)

        prompt = ChatPromptTemplate.from_template(self.exploit_template).format(\
            num=self.num_of_reason, num_total=2*self.num_of_reason, num_total=self.num_of_reason*50, chunk=anchor_chunk)
        
        message = [{"role": "user", "content": prompt}]
        response = self.attack_llm(message, temperature=self.exploit_temp)
        reasonings_queries = self._parse_exploit_response(response)
        logging.info(f"Exploitation: generated {len(reasonings_queries)} reasoning queries based on anchor chunk.")
        
        for rq in reasonings_queries:
            self.reasoning_queries.put(rq)
        return self._exploitation() 

    def _parse_exploit_response(self, response):
        """Output format 1. ...\n2. ...\n3. ..."""
        parts = re.split(r'\n(?=\d+\.\s)', response.strip())
        
        results = []
        for part in parts:
            cleaned = re.sub(r'^\d+\.\s*', '', part).strip()
            if cleaned:
                results.append(cleaned)
        return results


    def parse_response(self, response):
        results = parse_repeat_response(response)
        self._embed_unique_contents(results)
        return results
    

    def _embed_unique_contents(self, contents):
        """
        Embed and store unique contents in the embedding space.
        """
        for content in contents:
            content_embedding = self.embedding_model._embed(content)
            is_unique = all(np.linalg.norm(np.array(content_embedding) - np.array(vec)) > 1e-6 for vec in self.extracted_embs)
            if is_unique:
                self.extracted_chunks.add(content)
                self.extracted_embs.append(content_embedding)
                self.unexploited_chunks.add(content)

