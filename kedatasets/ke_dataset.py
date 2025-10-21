"""Dataset class for loading and processing Knowledge Extraction datasets."""
import logging
import os 

from ._data_load import load_from_local


class KnowExDataset(object):
    """
    Knowledge Extraction dataset class.
    TODO: For each dataset, should also setup the scenario, e.g., template of generator, system prompt, etc.
    """
    
    def __init__(self, dataset, debug, debug_len):
        """Creates TKG dataset object for data loading.
        
        Args:
            dataset: String indicating the dataset to use
        """
        data_path = os.environ["DATA_PATH"]
        data = load_from_local(data_path, dataset)
        self.debug = debug 
        self.debug_len = debug_len 
        
        self.benign_queries = data['uni_q']  # list
        self.contents = data['uni_c']        # list
        self.id2q = data['id2q']    # dict
        self.id2c = data['id2c']    # dict
        self.qa_pairs = data['qa_id']   # dict
        if self.debug:
            self.contents = self.contents[:self.debug_len]
        
        self.num_benign_queries = len(self.benign_queries)
        self.num_contents = len(self.contents)
        self.num_qas = len(self.qa_pairs)
        logging.info(f"Load {dataset}: # contents {self.num_contents} | # benign queries {self.num_benign_queries} | # Q&A pairs {self.num_qas}")
        
    
    def get_benign_queries(self):
        return self.benign_queries, self.id2q
    
    def get_contents(self):
        return self.contents, self.id2c
    
    def get_qa_pairs(self):
        return self.qa_pairs
        
    def get_shape(self):
        """Returns KnowEx dataset shape (num_contents, num_bqueries, num_qas)."""
        return self.num_contents, self.num_benign_queries, self.qa_pairs
        
        
        
        