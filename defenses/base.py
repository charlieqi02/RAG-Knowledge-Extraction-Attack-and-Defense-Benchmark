import os 
import numpy as np
from defenses.queryblock import QueryBlockerDefense



class DefenseBase:
    def __init__(self, args, df_args):
        self.args = args
        self.df_args = df_args
        if args.defense == "None":
            pass
        if args.defense == "Summary":
            args.rg.gen_kwargs.template = df_args.summary_prompt
        if args.defense == "SystemBlock":
            args.rg.gen_kwargs.system_prompt = df_args.system_block
        if args.defense == "Threshold":
            args.rg.retr_kwargs.threshold = df_args.threshold
            
        if args.defense == "QueryBlock":
            args.rg.query_blocker = QueryBlockerDefense(
                df_args.query_block_system,
                df_args.query_block_template
            )