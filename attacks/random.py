"""Implementation for Random attack."""
import logging
import random
import re
import os

import torch
import torch.nn as nn
import numpy as np
from langchain.prompts import ChatPromptTemplate
import pandas as pd

from .base import KnowExAttack
from tools.attacks import extract_indexes, detect_refusal
from tools.get_embedding import get_embedding
from tools.get_llm import get_llm
from tools.parse_response import parse_repeat_response


random_attack = ["RandomEmb", "RandomToken", "RandomText"]


class RandomEmb(KnowExAttack):
    """
    Random query attack: embedding-random.
    Reference: 
    """

    def __init__(self, args):
        """
        embedding_model, info_prompt, iterations=3, pool_size=512, allow_non_ascii=True,
        parser_llm="gpt4o-mini", llm_kwargs={}
        """
        super().__init__(args)
        self.embedding_model = get_embedding(args.emb_model)
        self.embedding_dim = self.embedding_model.dim
        
        # for greedy query optimization
        self.attack_template = args.attack_template    # Instraction for reproducing
        self.info_prompt = args.info_prompt
        
        self.iterations = args.iterations  # Number of iterations for query optimization.
        self.pool_size = args.pool_size  # tokens to sample from the total pool.
        self.allow_non_ascii = args.allow_non_ascii  # Whether to allow non-ASCII characters in the token pool.
        
        self.tokenizer = self.embedding_model.tokenizer
        self.device = self.embedding_model.device
        self.embed_sentence = self.embedding_model._embed
        self.control_toks = self.tokenizer.encode(self.info_prompt, add_special_tokens=False, return_tensors='pt')[0].to(self.device)
        self.all_tokens = list(range(len(self.tokenizer)))
        if not self.allow_non_ascii:
            self.all_tokens = [tok for tok in self.all_tokens if self.tokenizer.decode([tok]).isascii()]
        logging.info(f"Total tokens in tokenizer: {len(self.all_tokens)}")        
        
        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.attack_template), 'r') as f:
            self.attack_template = f.read()
        with open(os.path.join(prompt_dir, self.info_prompt), 'r') as f:
            self.info_prompt = f.read()
        
        extra_dir = os.environ.get("EXTRA_PATH")
        args.random_vec = args.random_vec.split(".")[0] + f"_wiki_{args.emb_model}.csv"
        vec_df = pd.read_csv(os.path.join(extra_dir, args.random_vec))
        self.random_vec = self._get_distribution_of_embeddings(vec_df['mean'].values, vec_df['variance'].values, vectors_num=args.max_query)
        

    def get_query(self, query_id):
        """
        Get the adversarial query for a given query round.
        Two processes:
            1) Choose target embedding
            2) Optimize the query info_prompt to approach the target embedding
        """
        # Choose target embedding
        target_embedding = self.random_vec[query_id]    # TODO: to device
        
        # Optimize the query info_prompt to approach the target embedding
        info_prompt, best_loss, best_embedding = self._optimize_query(target_embedding)
        query = self.attack_template.replace("<info>", info_prompt)
        logging.info(f"Query {query_id}: {query}")
        logging.info(f"Cosine to target_embedding: {1 - best_loss}")

        # TODO: consider the logging system
        return query

    def _get_distribution_of_embeddings(self, mean_vector, variance_vector, vectors_num=100):
        """
        Generate a set of vectors based on a normal distribution of mean and variance vectors.

        Args:
            mean_vector (list or np.array): Mean vector for generating embeddings.
            variance_vector (list or np.array): Variance vector for generating embeddings.
            vectors_num (int): Number of vectors to generate.

        Returns:
            np.array: Generated vectors based on the distribution.
        """
        mean_vector = np.array(mean_vector)
        variance_vector = np.array(variance_vector)
        generated_vectors = []
        for _ in range(vectors_num):
            sampled_vector = np.random.normal(loc=mean_vector, scale=np.sqrt(variance_vector))
            generated_vectors.append(sampled_vector)
        return np.array(generated_vectors)


    def _optimize_query(self, target_embedding):
        """
        Perform Greedy Cosine Quantization (GCQ) attack to perturb a suffix and create adversarial embeddings.

        Returns:
            tuple: Best suffix, best loss, and best embedding found during the attack.
            """
        # start optimizing
        best_suffix = self.info_prompt
        best_loss = float('inf')
        best_embedding = None
        for iteration in range(self.iterations):
            indices = list(range(len(self.control_toks)))
            random.shuffle(indices)
            for i in indices:
                current_best_toks = self.tokenizer.encode(best_suffix, add_special_tokens=False, return_tensors='pt')[0].to(
                    self.device)
                candidate_tokens = random.sample(self.all_tokens, self.pool_size)
                for token in candidate_tokens:
                    new_control_toks = current_best_toks.clone()
                    new_control_toks[i] = token
                    new_control_text = self.tokenizer.decode(new_control_toks)
                    perturbed_sentence = self.attack_template.replace("<info>", new_control_text)
                    sentence_embedding = self.embed_sentence(perturbed_sentence)
                    loss = self.__calculate_loss(sentence_embedding, target_embedding, self.device)
                    if loss < best_loss:
                        best_loss = loss
                        best_suffix = new_control_text
                        best_embedding = sentence_embedding
                        """if best_loss < 0.3: # add a Threshold 
                            return best_suffix, best_loss, best_embedding"""
            logging.info(f"Iteration {iteration + 1}/{self.iterations}, Loss: {best_loss}")
        return best_suffix, best_loss, best_embedding

    def __calculate_loss(self, opt_emb, target_emb, device):
        """
        Calculate cosine similarity loss between two embeddings.

        Returns:
            float: Cosine similarity loss.
        """
        cosine_similarity = nn.CosineSimilarity(dim=1)
        sentence_embedding = torch.tensor(opt_emb).to(device)
        target_embedding = torch.tensor(target_emb).to(device)
        if sentence_embedding.dim() == 1:
            sentence_embedding = sentence_embedding.unsqueeze(0)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        loss = 1 - cosine_similarity(sentence_embedding, target_embedding).mean().item()
        return loss

    def parse_response(self, response):
        results = parse_repeat_response(response)
        if results == []:
            if detect_refusal(response) == 1:
                results = [response]
        return results
                

class RandomToken(KnowExAttack):
    """
    Random query attack: Token-random. Info is a combination of random tokens.
    Reference: 
    """

    def __init__(self, args):
        """
        embedding_model, command_prompt, info_prompt, iterations=3, pool_size=512, allow_non_ascii=True,
        parser_llm="gpt4o-mini", llm_kwargs={}
        """
        super().__init__(args)
        embedding_model = get_embedding(args.emb_model)
        tokenizer = embedding_model.tokenizer
        all_tokens = list(range(len(tokenizer)))
        self.all_words = [tokenizer.decode([tok]) for tok in all_tokens]
        if not args.allow_non_ascii:
            self.all_words = [word for word in self.all_words if word.isascii()]
        logging.info(f"Total tokens in tokenizer: {len(self.all_words)}")
        
        # for random query
        self.attack_template = args.attack_template    # Instruction for reproducing
        self.pool_size = args.pool_size  # tokens to sample from the total pool.
        
        prompt_dir = os.environ.get("PROMPT_PATH")        
        with open(os.path.join(prompt_dir, self.attack_template), 'r') as f:
            self.attack_template = f.read()
        

    def get_query(self, query_id):
        """
        Get the adversarial query for a given query round.
        Get a random combination of tokens as the info_prompt.
        """
        # Optimize the query info_prompt to approach the target embedding
        info_prompt = self._random_query()
        query = self.attack_template.replace("<info>", info_prompt)
        logging.info(f"Query {query_id}: {query}")
        return query

    def _random_query(self):
        """
        Perform random token combination.

        Returns:
            tuple: Best suffix, best loss, and best embedding found during the attack.
        """
        # sample pool size words from all words
        sampled_words = random.sample(self.all_words, self.pool_size)
        info_prompt = ' '.join(sampled_words)
        return info_prompt

    def parse_response(self, response):
        results = parse_repeat_response(response)
        if results == []:
            if detect_refusal(response) == 1:
                results = [response]
        return results



class RandomText(KnowExAttack):
    """
    Random query attack: embedding-random.
    Reference: 
    """

    def __init__(self, args):
        """
        embedding_model, command_prompt, info_prompt, iterations=3, pool_size=512, allow_non_ascii=True,
        parser_llm="gpt4o-mini", llm_kwargs={}
        """
        super().__init__(args)
        
        # for greedy query optimization
        self.attack_template = args.attack_template    # Instraction for reproducing

        # for content generation
        self.attack_llm = get_llm(args.llm_model)
        
        self.random_template = args.template
        self.random_system_prompt = args.system_prompt
        self.random_temperature = args.temperature

        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.attack_template), 'r') as f:
            self.attack_template = f.read()
        with open(os.path.join(prompt_dir, self.random_template), 'r') as f:
            self.random_template = f.read()
        with open(os.path.join(prompt_dir, self.random_system_prompt), 'r') as f:
            self.random_system_prompt = f.read()


    def get_query(self, query_id):
        """
        Get the adversarial query for a given query round.
        Two processes:
            1) Choose target embedding
            2) Optimize the query info_prompt to approach the target embedding
        """
        messages = [
            {"role": "system", "content": self.random_system_prompt},
            {"role": "user", "content": self.random_template},
        ]

        info_prompt = self.attack_llm(messages, temperature=self.random_temperature)
        query = self.attack_template.replace("<info>", info_prompt)
        logging.info(f"Query {query_id}: {query}")

        return query


    def parse_response(self, response):
        results = parse_repeat_response(response)
        if results == []:
            if detect_refusal(response) == 1:
                results = [response]
        return results