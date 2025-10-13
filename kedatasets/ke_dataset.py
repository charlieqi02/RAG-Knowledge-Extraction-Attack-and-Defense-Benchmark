"""Dataset class for loading and processing Knowledge Extraction datasets."""
import logging
import os 

from _data_load import load_from_local


class KnowExDataset(object):
    """Knowledge Extraction dataset class"""
    
    def __init__(self, dataset, debug, debug_len):
        """Creates TKG dataset object for data loading.
        
        Args:
            dataset: String indicating the dataset to use
        """
        data_path = os.environ["DATA_PATH"]
        data = load_from_local(data_path, dataset)
        self.debug = debug 
        self.debug_len = debug_len 
        
        self.benign_queries = data['benign_queries']
        self.contents = data['contents']
        self.qa_pairs = data['Q&As']   
        if self.debug:
            self.contents = self.contents[:self.debug_len]
        
        self.num_benign_queries = len(self.benign_queries)
        self.num_contents = len(self.contents)
        self.num_qas = len(self.qa_pairs)
        
        
    def get_shape(self):
        """Returns KnowEx dataset shape (num_contents, num_bqueries, num_qas)."""
        return self.num_contents, self.num_benign_queries, self.qa_pairs
        
        
        
        