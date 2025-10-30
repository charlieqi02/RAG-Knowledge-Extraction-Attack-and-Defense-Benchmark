"""Knowledge Extraction Attack & Defense Pipeline."""
import json
import logging
import sys
import os
from tqdm import tqdm
import time

import rags
import attacks
import defenses

from kedatasets.ke_dataset import KnowExDataset
from recorder.recorder import Recorder

from rags import all_rags
from attacks import all_attacks
from defenses import all_defenses

from tools.train import set_seed, get_savedir
from tools.args import parse_all_args, ns_to_dict

from args.pipeline_args import get_pipeline_args
from args.rag_args import get_rag_args
from args.dataset_args import get_dataset_args
from args.attack_args import get_attack_args
from args.defense_args import get_defense_args


def pipeline(args):
    set_seed(args.seed)
    save_dir = get_savedir(args) 
    
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s    %(message)s",
        level=logging.INFO,
        datefmt="%m-%d %H:%M",
        filename=os.path.join(save_dir, "pipeline.log"))
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s    %(message)s", datefmt="%m-%d %H:%M")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))
    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(ns_to_dict(args), fjson, indent=2)
    logging.info("Config saved as config.json")
        
    # create dataset
    dataset = KnowExDataset(args.dataset, args.debug, args.debug_len)
    # create RAG system 
    rag = getattr(rags, args.rag)(args.rg)
    rag.index_content(dataset)
    
    # get attack method
    attack = getattr(attacks, args.attack)(args.ak)
    # get defense method
    # TODO: defense = getattr(defenses, args.defense)(args.df)
    # get recorder
    recorder = Recorder(save_dir, args)
    

    logging.info("Start knowledge extraction pipeline")
    logging.info(f"Starting from query_id={recorder.start_query}, ending at {args.ak.max_query - 1}, total={args.ak.max_query - recorder.start_query} queries")
    pbar = tqdm(total=args.ak.max_query - recorder.start_query, desc="Queries", unit="q", dynamic_ncols=True)
    for query_id in range(recorder.start_query, args.ak.max_query):
        logging.info(f"===== Query {query_id} =====")
        t0 = time.perf_counter()
        
        # use attack method to craft a query
        query = attack.get_query(query_id)
        t1 = time.perf_counter()
        
        # inject the query into rag system to get response
        response, retrieved_docs = rag.get_response(query)
        t2 = time.perf_counter()
        
        # parse response into extracted information
        extracted_info = attack.parse_response(response)
        t3 = time.perf_counter()
        
        # record information
        recorder.recording(query_id, query, 
                           response, 
                           retrieved_docs, 
                           extracted_info, 
                           times=(t0, t1, t2, t3))
        
        pbar.update(1)
        pbar.set_postfix({"last_q": query_id, "t_s": f"{t3-t0:.2f}"})
        
    # recorder.close_writer()
    # recorder.print_performance()


if __name__ == "__main__":
    try:
        p = parse_all_args(get_pipeline_args(all_rags, all_attacks, all_defenses),
                           get_rag_args,
                           get_dataset_args,
                           get_attack_args,
                           get_defense_args)
        pipeline(p)
    except Exception as e:
        logging.error(e, exc_info=True)

