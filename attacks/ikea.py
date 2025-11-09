"""Implementation for Dynamic Greedy Embedding Attack (DGEA)."""
import logging
from queue import Queue
import random
import re
import os
from typing import List, Optional, Tuple
from urllib import response

import torch
import torch.nn as nn
import numpy as np
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json
import re

from .base import KnowExAttack
from tools.attacks import extract_indexes
from tools.get_embedding import get_embedding
from tools.get_llm import get_llm
from tools.attacks import cos_sim, parse_anchor_words, detect_refusal


ikea_attack = ["IKEA"]


class IKEA(KnowExAttack):
    """
    Implicit Knowledge Extraction Attack (IKEA).
    Reference: https://arxiv.org/pdf/2505.15420
    """

    def __init__(self, args):
        """
        parameters:
        -----------
        """
        super().__init__(args)

        # Overall
        self.attack_llm = get_llm(args.attack_llm)
        self.attack_emb = get_embedding(args.attack_emb_model)
        
        ## Anchor concepts
        self.topic_word = args.topic_word  # topic word to guide anchor concept generation (prior)
        self.num_anchors = args.num_anchors  # initial number of anchor concepts to be generated
        self.anchors = {}  # generated anchor concepts <'aid': anchor_word>
        self.anchor_gen_template = args.anchor_gen_template  # prompt template for anchor concept generation
        self.query_gen_iteration = args.query_gen_iterations  # number of query generation iterations per anchor concept
        
        self.thresh_sim_topic = args.thresh_sim_topic   # generated anchors should be relevant to topic word (>= thresh)
        self.thresh_dissim_anchor = args.thresh_dissim_anchor   # generated anchors should be dissimilar to each other (<= thresh)
        self.thresh_q_anchor = args.thresh_q_anchor  # anchor-based query should be relevant to anchor (>= thresh)
        
        ## Experience reflection       
        self.query_response_pool = [] 
        self.query_history_unrelated = []
        self.query_history_outlier = []
        
        self.anchor_sample_scores = []  # anchor sampling scores <'aid': score>
        self.softmax = nn.Softmax(dim=0)        
        self.sample_temperature = args.sample_temperature
        
        self.anchor_query_gen_template = args.anchor_query_gen_template  # prompt template for anchor-based query generation
        self.thresh_irrelevant = args.thresh_irrelevant   # genertated response should be relevant to the query (>= thresh)
        self.thresh_outlier = args.thresh_outlier   # Outlier: refused 
                
        self.penalty_irrelevant = args.penalty_irrelevant
        self.penalty_refusal = args.penalty_refusal
        
        self.thresh_qy_sim = args.thresh_qy_sim  # successful (q,y) should have sim(q,y) >= thresh
        
        ## Trust-region directed mutation
        self.current_query = None
        self.current_response = None
        self.mut_flag = 0 # 0 means no mutation this round, 1 means do mutation
        
        self.gamma = args.gamma
        self.anchor_mutate_gen_template = args.anchor_mutate_gen_template  # prompt template for anchor mutation
        
        self.thresh_stop_q = args.thresh_stop_q
        self.thresh_stop_y = args.thresh_stop_y


        # load templates
        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.anchor_gen_template), 'r') as f:
            self.anchor_gen_template = f.read()
        with open(os.path.join(prompt_dir, self.anchor_query_gen_template), 'r') as f:
            self.anchor_query_gen_template = f.read()
        with open(os.path.join(prompt_dir, self.anchor_mutate_gen_template), 'r') as f:
            self.anchor_mutate_gen_template = f.read()
        
        self.selected = []
        self.generate_anchors()
        logging.info("Anchors: %s", str([c[0] for c in self.anchors.values()]))
    
    

    def get_query(self, query_id):
        """
        Two modes for getting a query:
        1) Sample anchor and get query.
        2) Mutate anchor based on previous successful (q,y) and get query.
        """
        if self.mut_flag == 0:
            # sampling mode: experience reflection based sampling
            anchor = self.sample_anchor()
            query = self.generate_queries_for_anchor(anchor)
        elif self.mut_flag == 1:
            # mutation mode: trust-region directed mutation
            anchor = self.mutate_anchor()
            if anchor is None:
                # cannot find valid mutated anchor, fallback to sampling
                anchor = self.sample_anchor()
                self.mut_flag = 0
            query = self.generate_queries_for_anchor(anchor)
        self.current_query = query
        
        logging.info(f"Query {query_id}: {query}")
        return query 
    
    
    # ---------------------
    # Anchor generation
    # ---------------------
    def generate_anchors(self):
        """
        Use the LLM to propose candidate anchors, filter by similarity to topic and by diversity.
        Populates self.anchors with num_anchors many items.
        """
        regen_candidates = self.num_anchors * 3
        
        if not self.topic_word:
            raise ValueError("topic_word required for anchor generation")

        prompt = self.anchor_gen_template.format(topic=self.topic_word, n=regen_candidates)
        raw = self.attack_llm([{"role": "user", "content": prompt}], temperature=0.5)
        # parse in a json format: {{ “anchor words”: [ “word1”, “word2”, “word3”, “...” ] }} 
        candidates = parse_anchor_words(raw)
        # logging.info("Raw anchor generation output: %s", raw)
        logging.info("Generated %d candidate anchors", len(candidates))

        # compute embeddings and filter by sim to topic
        topic_vec = self.attack_emb._embed(self.topic_word)
        cand_filtered: List[Tuple[str, float, np.ndarray]] = []
        for c in candidates:
            vec = self.attack_emb._embed(c)
            sim = cos_sim(topic_vec, vec)
            if sim >= self.thresh_sim_topic:
                cand_filtered.append((c, sim, vec))
        logging.info("Generated %d candidate anchors after topic relevance filtering", len(cand_filtered))

        # diversity selection: greedy farthest-first
        selected: List[Tuple[str, float, np.ndarray]] = self.selected
        while len(selected) < self.num_anchors and cand_filtered:
            if not selected:
                # pick highest sim to topic first
                cand_filtered.sort(key=lambda x: -x[1])
                selected.append(cand_filtered.pop(0))
            else:
                # pick candidate maximizing min distance to selected
                best_idx, best_score = None, -1.0
                for idx, (c, sim, vec) in enumerate(cand_filtered):
                    min_sim = min(cos_sim(vec, svec) for (_, _, svec) in selected)
                    # we want dissimilar anchors, so minimize similarity -> maximize (1 - sim)
                    # use min_sim directly to prefer lower similarity
                    if (1.0 - min_sim) > best_score:
                        best_score = (1.0 - min_sim)
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(cand_filtered.pop(best_idx))

            # enforce pairwise dissimilarity softly: if next candidate violates thresh_dissim_anchor
            # we'll still allow soft violations but log
            if len(selected) > 1:
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        s = cos_sim(selected[i][2], selected[j][2])
                        if s > self.thresh_dissim_anchor:
                            logging.info("Selected anchors have high pairwise sim=%.3f", s)

        self.selected = selected
        logging.info("Selected %d anchors after diversity filtering", len(selected))

        # if len(selected) < self.num_anchors:
        #     logging.info("Could not generate enough diverse anchors, generate another round ...")
        #     self.generate_anchors()
        #     return

        # populate anchors map
        for idx, (word, sim, vec) in enumerate(selected[: self.num_anchors]):
            aid = f"{idx}"
            self.anchors[aid] = (word, vec)
            self.anchor_sample_scores.append(0.0)

        logging.info("Generated %d anchors for topic '%s'", len(self.anchors), self.topic_word)
        
        return list(self.anchors.items())
    
    # ---------------------
    # Query generation for anchor
    # ---------------------
    def generate_queries_for_anchor(self, anchor: str) -> List[str]:
        """
        Generate candidate natural user questions from an anchor word.
        for loop, util q_per_anchor limit or one is pass thresh_q_anchor.
        """
        anchor_word = anchor
        anchor_vec = self.attack_emb._embed(anchor_word)
        prompt = self.anchor_query_gen_template.format(topic=self.topic_word, keyword=anchor_word)
        
        best_query, best_sim = None, -1.0
        for i in range(self.query_gen_iteration):        
            raw = self.attack_llm([{"role": "user", "content": prompt}], temperature=0.5)
            qvec = self.attack_emb._embed(raw)
            qw_sim = cos_sim(qvec, anchor_vec)
            if best_sim < qw_sim:
                best_sim = qw_sim
                best_query = raw
            if qw_sim >= self.thresh_q_anchor:
                break

        return best_query
    
    
    def sample_anchor(self) -> str:
        """
        Compute sample distribution over anchors using softmax over penalty scores.
        Anchors with more negative penalties will have lower probability.
        """
        scores = torch.tensor(self.anchor_sample_scores, dtype=torch.float32)
        # We use softmax( beta * scores )
        beta = self.sample_temperature
        self.softmax = nn.Softmax(dim=0)   
        probs = self.softmax(beta * scores)
        idx = torch.multinomial(probs, num_samples=1).item()

        return self.anchors[str(idx)][0]

    # ---------------------
    # TRDM (mutation)
    # ---------------------
    def mutate_anchor(self) -> Optional[str]:
        """
        Given a successful (q, y) pair for anchor 'aid', attempt to generate mutation anchors
        inside the trust region W* = { w | sim(w, y) >= gamma * sim(q, y) }.
        Select candidate that has minimal sim to q (i.e., farthest from q) to explore new areas.
        If a valid new anchor is found, add to self.anchors and return its id; else None.
        """
        # create prompt to generate candidates
        prompt = self.anchor_mutate_gen_template.format(q=self.current_query, y=self.current_response, n=50)
        raw = self.attack_llm([{"role": "user", "content": prompt}], temperature=0.5)
        candidates = parse_anchor_words(raw)
        logging.info("Generated %d candidate mutated anchors", len(candidates))
        
        # filter candidates by trust region
        query_vec = self.attack_emb._embed(self.current_query)
        response_vec = self.attack_emb._embed(self.current_response)
        sim_qy = cos_sim(query_vec, response_vec)
        trust_region_thresh = self.gamma * sim_qy
        
        trust_region_candidates = []
        for c in candidates:
            c_vec = self.attack_emb._embed(c)
            sim_cy = cos_sim(c_vec, response_vec)
            if sim_cy >= trust_region_thresh:
                # valid mutated anchor found
                trust_region_candidates.append((c, c_vec))
        logging.info("Found %d candidates inside trust region", len(trust_region_candidates))
        
        if not trust_region_candidates:
            return None
        
        # select candidate farthest from query
        best_cand, best_vec, best_sim = None, None, float('inf')
        for c, c_vec in trust_region_candidates:
            sim_cq = cos_sim(c_vec, query_vec)
            if sim_cq < best_sim:
                best_sim = sim_cq
                best_cand = c
                best_vec = c_vec
        
        return best_cand


    def parse_response(self, response):
        self.current_response = response
        
        query_vec = self.attack_emb._embed(self.current_query)
        response_vec = self.attack_emb._embed(self.current_response)
        
        
        if self.mut_flag == 1:
            # additional check for mutated (q,y)
            for his_q, his_y in self.query_response_pool:
                his_q_vec = self.attack_emb._embed(his_q)
                his_y_vec = self.attack_emb._embed(his_y)
                sim_y = cos_sim(his_y_vec, response_vec)
                if sim_y >= self.thresh_stop_y:
                    self.mut_flag = 0
                    break
                sim_q = cos_sim(his_q_vec, query_vec)
                if sim_q >= self.thresh_stop_q:
                    self.mut_flag = 0
                    break
        elif self.mut_flag == 0:
            self.mut_flag = 1  # assume mutation unless proven otherwise
            
            # detect irrelevant
            sim_qy = cos_sim(query_vec, response_vec)
            if sim_qy < self.thresh_qy_sim:
                self.query_history_unrelated.append(self.current_query)
                # update anchor sample score
                for idx, (anchor, anchor_vec) in self.anchors.items():
                    if cos_sim(anchor_vec, query_vec) >= self.thresh_irrelevant:
                        self.anchor_sample_scores[int(idx)] -= self.penalty_irrelevant
                self.mut_flag = 0
            
            # detect refusal
            if detect_refusal(self.attack_llm, self.current_response) == 0:
                self.query_history_outlier.append(self.current_query)
                # update anchor sample score
                for idx, (anchor, anchor_vec) in self.anchors.items():
                    if cos_sim(anchor_vec, query_vec) >= self.thresh_outlier:
                        self.anchor_sample_scores[int(idx)] -= self.penalty_refusal
                self.mut_flag = 0


        self.query_response_pool.append((self.current_query, self.current_response))
            
        return [response]
    


