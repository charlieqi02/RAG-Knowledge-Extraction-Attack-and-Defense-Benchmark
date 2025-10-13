"""Knowledge Extraction Attack & Defense Pipeline."""
import argparse
import json
import logging
import os

import rags
import attacks
import defenses

from kedatasets.ke_dataset import KnowExDataset
from recorder import Recorder
from rags import all_rags
from attacks import all_attacks
from defenses import all_defenses

from tools.train import set_seed, get_savedir


parser = argparse.ArgumentParser(
    description="Knowledge Extraction Attack & Defense Pipeline"
)
# General arguments
parser.add_argument(
    "--des", default="", type=str, help="Description of the run, will be saved in config.json")
parser.add_argument(
    "--dataset", default="HealthCareMagic", choices=["Enron", "HealthCareMagic"],
    help="Knowledge Extraction dataset")

parser.add_argument(
    "--rag", choices=all_rags, default="TextRAG", help="RAG systems for knowledge extraction attack & defense")
parser.add_argument(
    "--rag_args", type=json.loads, default=r"{}", help="JSON arguments for the sepcific RAG system")
parser.add_argument(
    "--attack", choices=all_attacks, default="DGEA", help="Knowledge extraction attack methods; run RAG benchmarking if None")
parser.add_argument(
    "--attack_args", type=json.loads, default=r"{}", help="JSON arguments for the sepcific attack method")
parser.add_argument(
    "--defense", choices=all_defenses, default="None", help="Knowledge extraction defense methods")
parser.add_argument(
    "--defense_args", type=json.loads, default=r"{}", help="JSON arguments for the sepcific defense method")

parser.add_argument(
    "--max_query", default=50, type=int, help="Query budget got attack")
parser.add_argument(
    "--continue_pipe", default=False, action='store_true', help="If use checkpoint to continue")
parser.add_argument(
    "--continue_dir", default="", type=str, help="Dir to use checkpoint to continue")
parser.add_argument(
    "--seed", default=3407, type=int, help="Random seed")
parser.add_argument(
    "--gpu", type=int, default=-1, help="gpu")

parser.add_argument(
    "--debug", default=False, action='store_true', help="Decrease docs in database for debugging")
parser.add_argument(
    "--debug_len", default=100, type=int, help="Decreased docs in database for debugging")


## Advanced record option
# parser.add_argument(
#     "--record_atth", default=False, action='store_true', help="Entity embedding l2 norm and multi-c record option")
# parser.add_argument(
#     "--record_model", default=0, type=int, help="Record model params every X epoch (0 -> no record)")
# parser.add_argument(
#     "--record_scale", default=0, type=int, help="Record scalors of certain train snap")


# model specific arguments
## ATTH
# parser.add_argument(
#     "--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
# parser.add_argument(
#     "--multi_c", action="store_true", help="Multiple curvatures per relation")
# parser.add_argument(
#     "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision")



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
        json.dump(vars(args), fjson, indent=2)
    logging.info("Config saved as config.json")
        
    # create dataset
    dataset = KnowExDataset(args.dataset, args.debug, args.debug_len)
    # create RAG system 
    rag = getattr(rags, args.rag)(args)
    # get attack method
    attack = getattr(attacks, args.attack)(args)
    # get defense method
    defense = getattr(defenses, args.defense)(args)
    # get recorder
    recorder = Recorder(save_dir, args, ...)
    

    logging.info("Start knowledge extraction pipeline")
    for query_id in range(recorder.start_query, args.max_query):
        
        # use attack method to craft a query
        query = attack.get_query(query_id)
        
        # inject the query into rag system to get response
        response = rag.get_response(query)
        
        # parse response into extracted information
        extracted_info = attack.parse_response(response)
        
        # record information
        recorder.recording(query, response, extracted_info)
        
        
    recorder.close_writer()
    recorder.print_performance()


if __name__ == "__main__":
    try:
        pipeline(parser.parse_args())
    except Exception as e:
        logging.error(e, exc_info=True)

