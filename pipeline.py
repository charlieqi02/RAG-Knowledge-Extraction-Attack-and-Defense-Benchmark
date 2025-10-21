"""Knowledge Extraction Attack & Defense Pipeline."""
import json
import logging
import os

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
        format="%(asctime)s %(levelname)-8s \t %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "pipeline.log"))
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s \t %(message)s")
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
    rag.index_content(dataset, args.debug_len)
    
    # get attack method
    attack = getattr(attacks, args.attack)(args.ak)
    # get defense method
    # TODO: defense = getattr(defenses, args.defense)(args.df)
    # get recorder
    recorder = Recorder(save_dir, args)
    

    logging.info("Start knowledge extraction pipeline")
    for query_id in range(recorder.start_query, args.ak.max_query):
        
        # use attack method to craft a query
        query = attack.get_query(query_id)
        
        # inject the query into rag system to get response
        response, retrieved_docs = rag.get_response(query)
        
        # parse response into extracted information
        extracted_info = attack.parse_response(response, retrieved_docs)
        
        # record information
        recorder.recording(query, response, retrieved_docs, extracted_info)
        
    recorder.close_writer()
    recorder.print_performance()


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

