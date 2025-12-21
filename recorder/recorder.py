import json
import os
import logging


class Recorder:
    def __init__(self, save_dir, args):
        self.start_query = 0
        self.log_path = os.path.join(save_dir, "results.jsonl")
        
        self.acm_rag_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.acm_attack_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        

    def recording(self, query_id, query, response, retrieved_docs, extracted_info, args, rag, attack, times):
        rag_step_cost, attack_step_cost = self.calculate_cost(args, rag, attack)
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
                "total_time": times[3] - times[0],
                'rag_step_cost': rag_step_cost,
                'attack_step_cost': attack_step_cost,
            }
            f.write(json.dumps(record, indent=2) + "\n")
        
        logging.info(f"Recorded query <{query_id}> results.")
        logging.info(f"Saved to {self.log_path} ...")
        self.start_query = query_id + 1
        

    def print_cost(self, args, rag, attack):    
        # petty print
        logging.info("=" * 40)
        logging.info("             TOKEN USAGE SUMMARY         ")
        logging.info("=" * 40)

        logging.info(">> RAG System")
        logging.info(f"   Prompt Tokens     : {self.acm_rag_cost['prompt_tokens']:,}")
        logging.info(f"   Completion Tokens : {self.acm_rag_cost['completion_tokens']:,}")
        logging.info(f"   Total Tokens      : {self.acm_rag_cost['total_tokens']:,}\n")

        logging.info(">> Attack LLM")
        logging.info(f"   Prompt Tokens     : {self.acm_attack_cost['prompt_tokens']:,}")
        logging.info(f"   Completion Tokens : {self.acm_attack_cost['completion_tokens']:,}")
        logging.info(f"   Total Tokens      : {self.acm_attack_cost['total_tokens']:,}\n")

        total_prompt = self.acm_rag_cost['prompt_tokens'] + self.acm_attack_cost['prompt_tokens']
        total_completion = self.acm_rag_cost['completion_tokens'] + self.acm_attack_cost['completion_tokens']
        total_tokens = self.acm_rag_cost['total_tokens'] + self.acm_attack_cost['total_tokens']

        logging.info(">> Overall Total")
        logging.info(f"   Prompt Tokens     : {total_prompt:,}")
        logging.info(f"   Completion Tokens : {total_completion:,}")
        logging.info(f"   Total Tokens      : {total_tokens:,}")
        logging.info("=" * 40 + "\n")
            
    
    def calculate_cost(self, args, rag, attack):
        rag_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        cost = rag.generator.metrics
        rag_cost["prompt_tokens"] += cost["prompt_tokens"]
        rag_cost["completion_tokens"] += cost["completion_tokens"]
        rag_cost["total_tokens"] += cost["prompt_tokens"] + cost["completion_tokens"]
        
        step_rag_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        step_rag_cost['prompt_tokens'] = rag_cost['prompt_tokens'] - self.acm_rag_cost['prompt_tokens']
        step_rag_cost['completion_tokens'] = rag_cost['completion_tokens'] - self.acm_rag_cost['completion_tokens']
        step_rag_cost['total_tokens'] = rag_cost['total_tokens'] - self.acm_rag_cost['total_tokens']
        self.acm_rag_cost = rag_cost
        
        
        attack_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if getattr(args.ak, "llm_model", None) or getattr(args.ak, "attack_llm", None):
            cost = attack.attack_llm.metrics
            attack_cost["prompt_tokens"] += cost["prompt_tokens"]
            attack_cost["completion_tokens"] += cost["completion_tokens"]
            attack_cost["total_tokens"] += cost["prompt_tokens"] + cost["completion_tokens"]
        step_attack_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        step_attack_cost['prompt_tokens'] = attack_cost['prompt_tokens'] - self.acm_attack_cost['prompt_tokens']
        step_attack_cost['completion_tokens'] = attack_cost['completion_tokens'] - self.acm_attack_cost['completion_tokens']
        step_attack_cost['total_tokens'] = attack_cost['total_tokens'] - self.acm_attack_cost['total_tokens']
        self.acm_attack_cost = attack_cost
            
        return step_rag_cost, step_attack_cost