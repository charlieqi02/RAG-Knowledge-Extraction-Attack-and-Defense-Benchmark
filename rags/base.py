"""Base RAG system."""
import os

from abc import ABC, abstractmethod
from tools.get_llm import get_llm
from tools.get_embedding import get_embedding


class RAGSystem(ABC):
    """
    Base RAG system.
    """

    def __init__(self, args):  
        self.db_path = args.db_path + f"_{args.retriever}"
        self.retriever = get_embedding(args.retriever, args.device)
        self.generator = get_llm(args.generator)
        self.gen_kwargs = args.gen_kwargs      # template, system_prompt, temperature
        self.retr_kwargs = args.retr_kwargs     # top-k / threshold
        
    # ----- Public API -----

    @abstractmethod
    def index_content(self, contents):
        """
        Index raw contents into database.
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(self, query):
        """
        End-to-end QA. 
        """
        raise NotImplementedError