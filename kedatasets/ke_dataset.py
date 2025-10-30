"""Dataset class for loading and processing Knowledge Extraction datasets."""
import logging
import os 

from ._data_load import load_from_local


class KnowExDataset(object):
    """
    Knowledge Extraction dataset class.
    """
    
    def __init__(self, dataset, debug, debug_len):
        """Creates TKG dataset object for data loading.
        
        Args:
            dataset: String indicating the dataset to use
        """
        data_path = os.environ["DATA_PATH"]
        data = load_from_local(data_path, dataset, debug_len)
        self.data = data
        self.debug = debug 
        self.debug_len = debug_len
        
        self.index_content = data['index_content']  # dict
        if self.debug:
            self.index_content = {str(k): self.index_content[str(k)] for k in range(self.debug_len)}
        self.num_index = len(self.index_content)
        
        logging.info(f"Load {dataset}: index content size={self.num_index}")
        
    
    def get_benign_queries(self):
        return self.data['uni_q'], self.data['id2q']
    
    def get_answers(self):
        return self.data['uni_a'], self.data['id2a']
    
    def get_qa_pairs(self):
        return self.data['qa_id']
        
        
        
        