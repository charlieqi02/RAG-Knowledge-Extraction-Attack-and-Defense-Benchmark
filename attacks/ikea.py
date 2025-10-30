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

from .base import KnowExAttack
from tools.attacks import extract_indexes
from tools.get_embedding import get_embedding
from tools.get_llm import get_llm


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
        
        # TODO: How to banlance diversity and relevance in anchor generation? (constrained optimization)
        self.thresh_sim_topic = args.thresh_sim_topic   # generated anchors should be relevant to topic word (>= thresh)
        self.thresh_dissim_anchor = args.thresh_dissim_anchor   # generated anchors should be dissimilar to each other (<= thresh)
        self.thresh_q_anchor = args.thresh_q_anchor  # anchor-based query should be relevant to anchor (>= thresh)
        
        ## Experience reflection
        self.current_query = None  # store current query info for experience reflection
        self.current_anchor = None
        
        self.anchor_query_response_pool = [] # store (anchor, query, response) triplets for experience reflection
        self.initial_anchor_sample_scores = []  # anchor sampling scores <'aid': score>
        self.softmax = nn.Softmax(dim=0)        
        self.sample_temperature = args.sample_temperature
        
        self.anchor_query_gen_template = args.anchor_query_gen_template  # prompt template for anchor-based query generation
        self.thresh_irrelevant = args.thresh_irrelevant   # genertated response should be relevant to the query (>= thresh)
        self.thresh_outlier = args.thresh_outlier   # Outlier: refused by the system, or system don't know this information say "I don't know" something
        
        self.penalty_irrelevant = args.penalty_irrelevant
        self.penalty_refusal = args.penalty_refusal
        
        ## Trust-region directed mutation
        self.trust_region_scale = args.trust_region_scale
        self.anchor_mutate_gen_template = args.anchor_mutate_gen_template  # prompt template for anchor mutation
        
        self.thresh_stop_q = args.thresh_stop_q
        self.thresh_stop_y = args.thresh_stop_y
        self.thresh_sim = args.thresh_sim
        

    def get_query(self, query_id):
        """
        Sample an anchor via ER and produce a natural user-like query for that anchor.
        Returns a dict: { 'query_id': int, 'aid': str, 'query': str }
        """
        # sample anchor
        aid = self.sample_anchor()
        # generate queries
        queries = self.generate_queries_for_anchor(aid, q_per_anchor=5)
        if not queries:
            # fallback: use anchor itself phrased as question
            q = f"What can you tell me about {self.anchors[aid]}?"
        else:
            # choose randomly among generated queries
            q = random.choice(queries)

        # store a "pending" placeholder in history? We'll store after parse_response to keep (q,y)
        meta = {"query_id": query_id, "aid": aid}
        return {"query_id": query_id, "aid": aid, "query": q, "meta": meta}


    # ---------------------
    # Anchor generation
    # ---------------------
    def generate_anchors(self, regen_candidates: int = 150):
        """
        Use the LLM to propose candidate anchors, filter by similarity to topic and by diversity.
        Populates self.anchors with num_anchors many items.
        """
        if not self.topic_word:
            raise ValueError("topic_word required for anchor generation")

        prompt = self.anchor_gen_template.format(topic=self.topic_word, n=regen_candidates)
        raw = self.llm_generate(prompt)
        # simple split: lines, commas
        candidates = re.split(r"[\n,;]+", raw)
        candidates = [c.strip() for c in candidates if c and len(c.strip()) <= 80]

        # compute embeddings and filter by sim to topic
        topic_vec = self.embed(self.topic_word)
        cand_filtered: List[Tuple[str, float, np.ndarray]] = []
        for c in candidates:
            try:
                vec = self.embed(c)
                sim = self.cos_sim(topic_vec, vec)
                if sim >= self.thresh_sim_topic:
                    cand_filtered.append((c, sim, vec))
            except Exception:
                continue

        # diversity selection: greedy farthest-first
        selected: List[Tuple[str, float, np.ndarray]] = []
        while len(selected) < self.num_anchors and cand_filtered:
            if not selected:
                # pick highest sim to topic first
                cand_filtered.sort(key=lambda x: -x[1])
                selected.append(cand_filtered.pop(0))
            else:
                # pick candidate maximizing min distance to selected
                best_idx, best_score = None, -1.0
                for idx, (c, sim, vec) in enumerate(cand_filtered):
                    min_sim = min(self.cos_sim(vec, svec) for (_, _, svec) in selected)
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
                        s = self.cos_sim(selected[i][2], selected[j][2])
                        if s > self.thresh_dissim_anchor:
                            logging.debug("Selected anchors have high pairwise sim=%.3f", s)

        # populate anchors map
        for word, sim, vec in selected[: self.num_anchors]:
            aid = f"A{self.next_anchor_idx}"
            self.next_anchor_idx += 1
            self.anchors[aid] = word
            self.initial_anchor_sample_scores[aid] = 0.0

        logging.info("Generated %d anchors for topic '%s'", len(self.anchors), self.topic_word)
        return list(self.anchors.items())

    # ---------------------
    # Query generation for anchor
    # ---------------------
    def generate_queries_for_anchor(self, aid: str, q_per_anchor: int = 5) -> List[str]:
        """
        Generate candidate natural user questions from an anchor word.
        Returns list of queries that pass thresh_q_anchor.
        """
        if aid not in self.anchors:
            return []
        anchor_word = self.anchors[aid]
        prompt = self.anchor_query_gen_template.format(anchor=anchor_word, k=q_per_anchor)
        raw = self.llm_generate(prompt)
        qs = re.split(r"[\n]+", raw)
        qs = [q.strip() for q in qs if q.strip()]

        queries = []
        anchor_vec = self.embed(anchor_word)
        for q in qs:
            if len(queries) >= q_per_anchor:
                break
            try:
                qvec = self.embed(q)
                if self.cos_sim(qvec, anchor_vec) >= self.thresh_q_anchor:
                    # simple question check
                    if q.endswith("?") or re.match(r"^(what|how|why|when|where|who)\b", q.lower()):
                        queries.append(q)
            except Exception:
                continue

        # fallback: relax threshold once
        if not queries:
            logging.debug("No queries passed thresh for anchor %s, relaxing threshold", aid)
            for q in qs:
                try:
                    qvec = self.embed(q)
                    if self.cos_sim(qvec, anchor_vec) >= (self.thresh_q_anchor - 0.1):
                        queries.append(q)
                except Exception:
                    continue

        return queries


    # ---------------------
    # ER sampling
    # ---------------------
    def _anchor_penalty(self, aid: str) -> float:
        """
        Sum of psi(w,h) for anchor aid over history.
        psi(w,h) defined:
          - if h in H_o (refusal) and sim(w,q_h) > delta_o -> -p
          - if h in H_u (irrelevant) and sim(w,q_h) > delta_u -> -kappa
          - else 0
        We'll use delta_o and delta_u as thresholds derived from attributes (thresh_stop_q etc).
        """
        total = 0.0
        # get anchor vector
        try:
            anchor_vec = self.embed(self.anchors[aid])
        except Exception:
            return 0.0

        # define similarity radii
        delta_o = getattr(self, "thresh_stop_q", 0.3)
        delta_u = getattr(self, "thresh_stop_q", 0.45)

        for (h_aid, q_h, y_h, rd_h, sim_qy_h, sim_y_rd_h, label) in self.anchor_query_response_pool:
            # compute sim between anchor and historical query
            try:
                qh_vec = self.embed(q_h)
                sim_wa = self.cos_sim(anchor_vec, qh_vec)
            except Exception:
                sim_wa = 0.0

            if label == "refusal" and sim_wa > delta_o:
                total -= self.penalty_refusal
            elif label == "irrelevant" and sim_wa > delta_u:
                total -= self.penalty_irrelevant
            # useful -> no negative penalty

        return total
    
    
    def sample_anchor(self) -> str:
        """
        Compute sample distribution over anchors using softmax over penalty scores.
        Anchors with more negative penalties will have lower probability.
        """
        aids = list(self.anchors.keys())
        if not aids:
            raise RuntimeError("No anchors to sample")

        scores = np.array([self._anchor_penalty(aid) for aid in aids], dtype=float)
        # convert to probabilities: higher score -> more likely; but penalties are negative so we invert
        # We use softmax( beta * scores )
        beta = float(getattr(self, "sample_temperature", 1.0))
        # To improve numeric stability, subtract max
        max_score = np.max(scores)
        exps = np.exp(beta * (scores - max_score))
        probs = exps / (np.sum(exps) + 1e-12)

        choice = np.random.choice(len(aids), p=probs)
        chosen_aid = aids[choice]
        return chosen_aid

    # ---------------------
    # TRDM (mutation)
    # ---------------------
    def mutate_anchor(self, aid: str, q: str, y: str, sim_qy: float) -> Optional[str]:
        """
        Given a successful (q, y) pair for anchor 'aid', attempt to generate mutation anchors
        inside the trust region W* = { w | sim(w, y) >= gamma * sim(q, y) }.
        Select candidate that has minimal sim to q (i.e., farthest from q) to explore new areas.
        If a valid new anchor is found, add to self.anchors and return its id; else None.
        """
        # create prompt to generate candidates
        prompt = self.anchor_mutate_gen_template.format(anchor=self.anchors[aid], query=q, response=y, n=20)
        raw = self.llm_generate(prompt)
        candidates = re.split(r"[\n,;]+", raw)
        candidates = [c.strip() for c in candidates if c.strip()]

        if not candidates:
            return None

        y_vec = self.embed(y)
        q_vec = self.embed(q)
        gamma = float(self.trust_region_scale)

        # build candidate list in W*
        cand_valid = []
        for c in candidates:
            try:
                cvec = self.embed(c)
                sim_c_y = self.cos_sim(cvec, y_vec)
                if sim_c_y >= gamma * sim_qy:
                    sim_c_q = self.cos_sim(cvec, q_vec)
                    cand_valid.append((c, sim_c_y, sim_c_q))
            except Exception:
                continue

        if not cand_valid:
            logging.debug("No mutation candidates inside trust region for anchor %s", aid)
            return None

        # choose candidate with minimal sim to q (farthest)
        cand_valid.sort(key=lambda x: x[2])  # by sim_c_q ascending
        chosen_word = cand_valid[0][0]

        # check similarity to existing anchors to avoid near-duplicates
        chosen_vec = self.embed(chosen_word)
        for existing in self.anchors.values():
            if self.cos_sim(chosen_vec, self.embed(existing)) > self.thresh_dissim_anchor:
                logging.debug("Mutation candidate too similar to existing anchor, skipping")
                # still allow but log; in practice we may skip or add depending on policy
                # for now continue to next candidate
                # try to find another candidate
                for cand in cand_valid[1:]:
                    cand_word = cand[0]
                    cand_vec = self.embed(cand_word)
                    if self.cos_sim(cand_vec, self.embed(existing)) <= self.thresh_dissim_anchor:
                        chosen_word = cand_word
                        chosen_vec = cand_vec
                        break

        # add to anchors
        new_aid = f"A{self.next_anchor_idx}"
        self.next_anchor_idx += 1
        self.anchors[new_aid] = chosen_word
        self.initial_anchor_sample_scores[new_aid] = 0.0
        logging.info("Mutated new anchor %s -> %s from parent %s", new_aid, chosen_word, aid)
        return new_aid


    def parse_response(self, response):
        self.anchor_query_response_pool.append(
            self.current_anchor,
            self.current_query,
            response
            )
        return [response]
    




