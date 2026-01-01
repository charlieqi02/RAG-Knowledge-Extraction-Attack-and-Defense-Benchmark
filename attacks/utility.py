"""Implementation for Utility measure."""
import logging
import random
import re
import os
import json  # Added json import
import sys

import torch
import torch.nn as nn
import numpy as np
from langchain.prompts import ChatPromptTemplate
import pandas as pd

from .base import KnowExAttack
from tools.attacks import detect_refusal
from tools.parse_response import parse_repeat_response


utility = ["Utility"]


class Utility(KnowExAttack):
    """
    Utility: utility-based query.
    Loads QA pairs from utility_questions.jsonl and queries the model.
    """

    def __init__(self, args):
        """
        Args:
            args: Namespace containing arguments. 
                  Expected args.data_path to point to the dataset folder containing utility_questions.jsonl
                  or args.utility_questions_file to point directly to the file.
        """
        super().__init__(args)

        # 1. 确定文件路径
        # 优先使用 args 中指定的具体文件路径，否则尝试在 data_path 下寻找
        if hasattr(args, 'data_path') and args.data_path:
            jsonl_path = os.path.join(args.data_path, "utility_questions.jsonl")
        self.utility_queue = []
        
        # 2. 加载 JSONL 数据
        if os.path.exists(jsonl_path):
            print(f"Loading utility questions from {jsonl_path}...")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.utility_queue.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            # 3. 限制数量为 1000 (如果文件中有更多)
            if len(self.utility_queue) > 1000:
                self.utility_queue = self.utility_queue[:1000]
            
            print(f"Loaded {len(self.utility_queue)} utility questions.")
        else:
            logging.error(f"Utility file not found at {jsonl_path}")
            # 防止崩溃，初始化为空列表
            self.utility_queue = []
        self.current_sample = None

    def get_query(self, query_id):
        """
        Get the utility query for the current round.
        Removes the question from the queue after fetching.
        """
        self.current_sample = self.utility_queue.pop(0)
        question = self.current_sample.get("question", "")
        query = question

        logging.info(f"Query {query_id}: {query}")
        return query

    def parse_response(self, response):
        """
        Parse the model's response.
        """
        results = parse_repeat_response(response)
        if results == []:
            if detect_refusal(response) == 1:
                results = [response]
        return results