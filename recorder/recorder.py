import json
import os
import logging


class Recorder:
    def __init__(self, save_dir, args):
        self.start_query = 0
        self.log_path = os.path.join(save_dir, "results.jsonl")


    def recording(self, query_id, query, response, retrieved_docs, extracted_info, times):
        # Save the recording information to log file
        with open(self.log_path, "a") as f:
            record = {
                "query_id": query_id,
                "query": query,
                "response": response,
                "retrieved_docs": [
                    {"content": doc[0].page_content, "index": doc[0].metadata.get("index", None), "score": doc[1]} 
                    for doc in retrieved_docs
                ],
                "extracted_info": extracted_info,
                "attack_query_time": times[1] - times[0],
                "rag_response_time": times[2] - times[1],
                "parsing_time": times[3] - times[2],
                "total_time": times[3] - times[0]
            }
            f.write(json.dumps(record, indent=2) + "\n")
        
        logging.info(f"Recorded query <{query_id}> results.")
        logging.info(f"Saved to {self.log_path} ...")
        self.start_query = query_id + 1
        

    def some_metrics_calculation(self):
        pass