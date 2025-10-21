"""Base classes for Knowledge Extraction attacks."""
from abc import ABC, abstractmethod


class KnowExAttack(ABC):
    """
    Base class for Knowledge Extraction attacks. 
    TODO: Different attacks should have their own option of prompt template, etc.
    """

    def __init__(self, args):
        self.max_query = args.max_query
        self.args = args

    # ----- Public API -----

    @abstractmethod
    def get_query(self, query_id):
        """
        Get the adversarial query for a given query round.
        """
        pass

    @abstractmethod
    def parse_response(self, response, retrieved_docs):
        """
        Parse the response from RAG system (generator) to extract information.
        """
        pass