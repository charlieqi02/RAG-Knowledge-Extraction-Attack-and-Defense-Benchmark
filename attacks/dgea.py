"""Implementation for Dynamic Greedy Embedding Attack (DGEA)."""
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
from tools.attacks import extract_indexes
from tools.get_embedding import get_embedding
from tools.get_llm import get_llm


dgea_attack = ["DGEA"]


class DGEA(KnowExAttack):
    """
    Dynamic Greedy Embedding Attack (DGEA).
    Reference: https://arxiv.org/pdf/2409.08045
    """

    def __init__(self, args):
        """
        embedding_model, command_prompt, info_prompt, iterations=3, pool_size=512, allow_non_ascii=True,
        parser_llm="gpt4o-mini", llm_kwargs={}
        """
        super().__init__(args)
        self.queried = []
        self.current_query = None
        
        self.retrieved_idx = set()
        self.replied_idx = set()
        
        self.embedding_model = get_embedding(args.emb_model)
        self.embedding_dim = self.embedding_model.dim
        self.embedding_space = []  # A collection of existing vectors (embeddings) that form the embedding space.
        
        # for greedy query optimization
        self.command_prompt = args.command_prompt    # Instraction for reproducing
        self.info_prompt = args.info_prompt    # Init info_prompt, for locating the query to some target document in database.

        self.iterations = args.iterations  # Number of iterations for query optimization.
        self.pool_size = args.pool_size  # tokens to sample from the total pool.
        self.allow_non_ascii = args.allow_non_ascii  # Whether to allow non-ASCII characters in the token pool.

        # for content extraction when updating embedding space
        self.parser_llm = get_llm(args.parser_llm)
        self.temperature = args.llm_kwargs.temperature
        self.example = args.llm_kwargs.example 
        self.template = args.llm_kwargs.template
        self.system_prompt = args.llm_kwargs.system_prompt
        
        
        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.template), 'r') as f:
            self.template = f.read()
        with open(os.path.join(prompt_dir, self.system_prompt), 'r') as f:
            self.system_prompt = f.read()
        with open(os.path.join(prompt_dir, self.example), 'r') as f:
            self.example_text = f.read()
        with open(os.path.join(prompt_dir, self.command_prompt), 'r') as f:
            self.command_prompt = f.read()
        with open(os.path.join(prompt_dir, self.info_prompt), 'r') as f:
            self.info_prompt = f.read()
        
        extra_dir = os.environ.get("EXTRA_PATH")
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
        if query_id == 0:
            target_embedding = self.random_vec[query_id]    # TODO: to device
        else: 
            target_embedding = self._find_dissimilar_vector()
        
        # Optimize the query info_prompt to approach the target embedding
        info_prompt, best_loss, best_embedding = self._optimize_query(target_embedding)
        query = self.command_prompt + " " + info_prompt
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

        
    def _find_dissimilar_vector(self):
        """
            Find a vector that is dissimilar to the existing set of vectors in the embedding space.
            Use gradient-based optimization to find farthest vector.

            Returns:
                np.array: A vector that is dissimilar to the centroid of the embedding space.
        """
        embedding_space_tensor = torch.tensor(self.embedding_space, dtype=torch.float32)
        centroid = torch.mean(embedding_space_tensor, dim=0)
        farthest_vector = torch.randn(self.embedding_dim, requires_grad=True)
        farthest_vector = 0.6 * (farthest_vector - torch.min(farthest_vector)) / (
                torch.max(farthest_vector) - torch.min(farthest_vector)) - 0.3
        farthest_vector = farthest_vector.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([farthest_vector], lr=0.01)
        loss_fn = torch.nn.CosineEmbeddingLoss()
        for _ in range(30000):
            optimizer.zero_grad()
            target = torch.tensor([-1.0], dtype=torch.float32)
            loss = loss_fn(farthest_vector.unsqueeze(0), centroid.unsqueeze(0), target)
            loss.backward(retain_graph=False)
            optimizer.step()
            farthest_vector.data = torch.clamp(farthest_vector.data, -0.3, 0.3)
        return farthest_vector.detach().numpy()


    def _optimize_query(self, target_embedding):
        """
        Perform Greedy Cosine Quantization (GCQ) attack to perturb a suffix and create adversarial embeddings.

        Returns:
            tuple: Best suffix, best loss, and best embedding found during the attack.
            """
        tokenizer = self.embedding_model.tokenizer
        device = self.embedding_model.device
        embed_sentence = self.embedding_model._embed
        control_toks = tokenizer.encode(self.info_prompt, add_special_tokens=False, return_tensors='pt')[0].to(device)
        all_tokens = list(range(len(tokenizer)))
        if not self.allow_non_ascii:
            all_tokens = [tok for tok in all_tokens if tokenizer.decode([tok]).isascii()]

        # start optimizing
        best_suffix = self.info_prompt
        best_loss = float('inf')
        best_embedding = None
        for iteration in range(self.iterations):
            indices = list(range(len(control_toks)))
            random.shuffle(indices)
            for i in indices:
                current_best_toks = tokenizer.encode(best_suffix, add_special_tokens=False, return_tensors='pt')[0].to(
                    device)
                candidate_tokens = random.sample(all_tokens, self.pool_size)
                for token in candidate_tokens:
                    new_control_toks = current_best_toks.clone()
                    new_control_toks[i] = token
                    new_control_text = tokenizer.decode(new_control_toks)
                    perturbed_sentence = self.command_prompt + ' ' + new_control_text
                    sentence_embedding = embed_sentence(perturbed_sentence)
                    loss = self.__calculate_loss(sentence_embedding, target_embedding, device)
                    if loss < best_loss:
                        best_loss = loss
                        best_suffix = new_control_text
                        best_embedding = sentence_embedding
                        """if best_loss < 0.3: # add a Threshold 
                            return best_suffix, best_loss, best_embedding"""
            print(f"Iteration {iteration + 1}/{self.iterations}, Loss: {best_loss}")
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


    def parse_response(self, response, retrieved_docs):
        """
        Parse the response from RAG system (generator) to extract information.
        
        TODO: original implementation is not good, it just compair index generated, instead of generated text and ground truth text.
        TODO: maybe this part should be in base model, for consistent evaluation.
        """
        # What is retrieved
        NumberOfUniqueIndexesAdded = 0
        IndexesRetrieved = []
        IndexesAddedUnique = []
        IndexesAddedUniqueCosineSimilarity = []
        IndexesCosineSimilarity = []
        
        for doc in retrieved_docs:
            if doc[0].metadata["index"] not in self.retrieved_idx:
                NumberOfUniqueIndexesAdded += 1
                self.retrieved_idx.add(doc[0].metadata["index"])
                IndexesAddedUnique.append(doc[0].metadata["index"])
                IndexesAddedUniqueCosineSimilarity.append(doc[1])
            IndexesRetrieved.append(doc[0].metadata["index"])
            IndexesCosineSimilarity.append(doc[1])
            
        # What is replied
        IndexesReplied = []
        IndexesRepliedCosineSimilarity = []
        IndexesDuplicatedCount = 0
        IndexesDuplicateReplied = []
        HallucinatedIndexes = []
        
        CurrentIndexListFromReply = extract_indexes(response)
        for CurrentDocindex in CurrentIndexListFromReply:
            if CurrentDocindex not in IndexesRetrieved:
                HallucinatedIndexes.append(CurrentDocindex)
            else:
                if CurrentDocindex not in IndexesReplied:
                    IndexesReplied.append(CurrentDocindex)
                    DocCosine = IndexesCosineSimilarity[IndexesRetrieved.index(CurrentDocindex)]
                    IndexesRepliedCosineSimilarity.append(DocCosine)
                    self.replied_idx.add(CurrentDocindex)
                else:
                    IndexesDuplicateReplied.append(CurrentDocindex)
                    IndexesDuplicatedCount += 1
        
        # Update embedding space
        contents = self._extract_contents(response)
        self._embed_unique_contents(contents)
        
        return (response, IndexesRetrieved, IndexesCosineSimilarity, NumberOfUniqueIndexesAdded,
                IndexesAddedUnique, IndexesAddedUniqueCosineSimilarity, IndexesReplied,
                IndexesRepliedCosineSimilarity, IndexesDuplicateReplied, IndexesDuplicatedCount,
                HallucinatedIndexes)
        
        
    def _extract_contents(self, response):
        """
            Extract or fetch content from response, and if none are found, use an LLM to generate the content.
        """
        contents = self.__extract(response)
        if len(contents) == 0:
            logging.info("No content found in the response, using LLM to extract content.")
            prompt = ChatPromptTemplate.from_template(self.template).format(
                example=self.example, text=response)
            # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=gemini_api_key,
            #                              safety_settings=None)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            reply = self.parser_llm(messages, temperature=self.temperature)
            contents = self.__extract(reply)
        return contents
        
    
    def __extract(self, text):
        content_pattern = r'(?:\"?)Content(?:\"?)\s*:\s*\"([^\"]+)\"'
        matches = re.findall(content_pattern, text)
        return matches
        
        
    def _embed_unique_contents(self, contents):
        """
        Embed and store unique contents in the embedding space.
        """
        for content in contents:
            content_embedding = self.embedding_model._embed(content)
            is_unique = all(np.linalg.norm(np.array(content_embedding) - np.array(vec)) > 1e-6 for vec in self.embedding_space)
            if is_unique:
                self.embedding_space.append(content_embedding)