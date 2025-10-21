import argparse
from types import SimpleNamespace


def parse_all_args(add_pipeline_args, add_rag_args, add_dataset_args, add_attack_args, add_defense_args):
    # 1) parse general first
    p1 = build_base_parser()
    add_pipeline_args(p1)
    args1, _ = p1.parse_known_args()

    # 2) dynamic parse
    p2 = build_base_parser()
    add_pipeline_args(p2)                     
    add_rag_args(p2, args1.rag)
    add_dataset_args(p2, args1.dataset)
    add_attack_args(p2, args1.attack)
    add_defense_args(p2, args1.defense)
    args = p2.parse_args()

    # 3) flat -> nested
    cfg = to_namespace(nestify(vars(args)))

    return cfg


def build_base_parser():
    return argparse.ArgumentParser(
            description="Knowledge Extraction Attack & Defense Pipeline")
    
def nestify(flat: dict, sep="."):
    out = {}
    for k, v in flat.items():
        if v is None: 
            continue
        parts = k.split(sep) if sep in k else [k]
        cur = out
        for s in parts[:-1]:
            cur = cur.setdefault(s, {})
        cur[parts[-1]] = v
    return out

def to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, to_namespace(v) if isinstance(v, dict) else v)
    return ns



def ns_to_dict(obj):
    """Recursively convert SimpleNamespace to dict"""
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [ns_to_dict(v) for v in obj]
    else:
        return obj
