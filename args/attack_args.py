


def get_attack_args(p, attack):

    p.add_argument(
        "--ak_max_query", dest="ak.max_query", default=50, type=int, help="Query budget got attack")

    if attack == "DGEA":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for DGEA")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=3, type=int, help="Number of iterations for query optimization")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for DGEA")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query")
        p.add_argument(
            "--ak_command_prompt", dest="ak.command_prompt", default="default", type=str, help="Command prompt path for DGEA")
        p.add_argument(
            "--ak_info_prompt", dest="ak.info_prompt", default="default", type=str, help="Info prompt path for DGEA")
        p.add_argument(
            "--ak_random_vec", dest="ak.random_vec", default="", type=str, help="Random vector distribution file path for DGEA")
    
    elif attack == "CopyBreak":
        p.add_argument(
            "--ak_llm_model", dest="ak.llm_model", default="gpt4o-mini", type=str, help="LLM model for CopyBreak")
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for CopyBreak")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="default", type=str, help="Attack template prompt path for CopyBreak")
        p.add_argument(
            "--ak_sim_thresh", dest="ak.sim_thresh", default=0.7, type=float, help="Similarity threshold for exploration in CopyBreak")
        p.add_argument(
            "--ak_explore_template", dest="ak.explore_template", default="default", type=str, help="Exploration template prompt path for CopyBreak")
        p.add_argument(
            "--ak_exploit_template", dest="ak.exploit_template", default="default", type=str, help="Exploitation template prompt path for CopyBreak")
        p.add_argument(
            "--ak_exchange_rate", dest="ak.exchange_rate", default=0.5, type=float, help="Rate of change between exploration and exploitation in CopyBreak")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=10, type=int, help="Number of iterations for each exploration/exploitation in CopyBreak")
        p.add_argument(
            "--ak_explore_temperature", dest="ak.explore_temperature", default=0.7, type=float, help="Temperature for exploration generation in CopyBreak")
        p.add_argument(
            "--ak_exploit_temperature", dest="ak.exploit_temperature", default=0.7, type=float, help="Temperature for exploitation generation in CopyBreak")
        p.add_argument(
            "--ak_num_of_each_reason", dest="ak.num_of_reason", default=3, type=int, help="Number of each back/forward reasoning queries to generate per anchor chunk in CopyBreak")
        
    elif attack == "IKEA":
        p.add_argument(
            "--ak_attack_llm", dest="ak.attack_llm", default="gpt4o-mini", type=str, help="LLM model for IKEA attack")
        p.add_argument(
            "--ak_attack_emb_model", dest="ak.attack_emb_model", default="MiniLM", type=str, help="Embedding model for IKEA attack")
        p.add_argument(
            "--ak_topic_word", dest="ak.topic_word", default="default", type=str, help="Topic word for IKEA attack")
        p.add_argument(
            "--ak_num_anchors", dest="ak.num_anchors", default=5, type=int, help="Number of anchor points for IKEA attack")
        p.add_argument(
            "--ak_anchor_gen_template", dest="ak.anchor_gen_template", default="default", type=str, help="Anchor generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_query_gen_iterations", dest="ak.query_gen_iterations", default=5, type=int, help="Number of query generation iterations per anchor for IKEA attack")
        p.add_argument(
            "--ak_thresh_sim_topic", dest="ak.thresh_sim_topic", default=0.5, type=float, help="Similarity threshold between anchor and topic word for IKEA attack")
        p.add_argument(
            "--ak_thresh_dissim_anchor", dest="ak.thresh_dissim_anchor", default=0.5, type=float, help="Dissimilarity threshold among anchors for IKEA attack")
        p.add_argument(
            "--ak_thresh_q_anchor", dest="ak.thresh_q_anchor", default=0.5, type=float, help="Similarity threshold between query and anchor for IKEA attack")
        p.add_argument(
            "--ak_sample_temperature", dest="ak.sample_temperature", default=1, type=float, help="Sampling temperature for IKEA attack")
        p.add_argument(
            "--ak_anchor_query_gen_template", dest="ak.anchor_query_gen_template", default="default", type=str, help="Anchor to query generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_thresh_irrelevant", dest="ak.thresh_irrelevant", default=0.5, type=float, help="Threshold to filter irrelevant queries for IKEA attack")
        p.add_argument(
            "--ak_thresh_outlier", dest="ak.thresh_outlier", default=0.5, type=float, help="Threshold to filter outlier queries for IKEA attack")
        p.add_argument(
            "--ak_penalty_irrelevant", dest="ak.penalty_irrelevant", default=0.5, type=float, help="Penalty weight for irrelevant queries for IKEA attack")
        p.add_argument(
            "--ak_penalty_refusal", dest="ak.penalty_refusal", default=0.5, type=float, help="Penalty weight for refusal queries for IKEA attack")
        p.add_argument(
            "--ak_thresh_qy_sim", dest="ak.thresh_qy_sim", default=0.5, type=float, help="Threshold for query-response similarity for IKEA attack")
        p.add_argument(
            "--ak_gamma", dest="ak.gamma", default=0.5, type=float, help="Trust region scale for IKEA attack")
        p.add_argument(
            "--ak_anchor_mutate_gen_template", dest="ak.anchor_mutate_gen_template", default="default", type=str, help="Anchor mutation generation template prompt path for IKEA attack")
        p.add_argument(
            "--ak_thresh_stop_q", dest="ak.thresh_stop_q", default=0.9, type=float, help="Threshold to stop query generation for IKEA attack")
        p.add_argument(
            "--ak_thresh_stop_y", dest="ak.thresh_stop_y", default=0.9, type=float, help="Threshold to stop query generation for IKEA attack")

    elif attack == "RandomText":
        p.add_argument(
            "--ak_llm_model", dest="ak.llm_model", default="gpt4o-mini", type=str, help="LLM model for RandomText attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="random/attack_template.txt", type=str, help="Attack instruction template path for RandomText attack")
        p.add_argument(
            "--ak_system_prompt", dest="ak.system_prompt", default="random/gen_system.txt", type=str, help="System prompt path for RandomText attack")
        p.add_argument(
            "--ak_template", dest="ak.template", default="random/gen_template.txt", type=str, help="Prompt template path for RandomText attack")
        p.add_argument(
            "--ak_temperature", dest="ak.temperature", default=0.9, type=float, help="Generation temperature for RandomText attack")
        
    elif attack == "RandomToken":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for RandomToken attack")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for RandomToken attack")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query for RandomToken attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="random/attack_template.txt", type=str, help="Attack instruction template path for RandomToken attack")
        
    elif attack == "RandomEmb":
        p.add_argument(
            "--ak_emb_model", dest="ak.emb_model", default="MiniLM", type=str, help="Embedding model for RandomEmb attack")
        p.add_argument(
            "--ak_random_vec", dest="ak.random_vec", default="", type=str, help="Random vector distribution file path for RandomEmb attack")
        p.add_argument(
            "--ak_attack_template", dest="ak.attack_template", default="random/attack_template.txt", type=str, help="Attack instruction template path for RandomEmb attack")
        p.add_argument(
            "--ak_iterations", dest="ak.iterations", default=3, type=int, help="Number of iterations for RandomEmb attack")
        p.add_argument(
            "--ak_pool_size", dest="ak.pool_size", default=512, type=int, help="Token pool size for RandomEmb attack")
        p.add_argument(
            "--ak_allow_non_ascii", dest="ak.allow_non_ascii", default=False, action='store_true', help="Allow non-ascii characters in the query for RandomEmb attack")
        p.add_argument(
            "--ak_info_prompt", dest="ak.info_prompt", default="default", type=str, help="Info prompt path for RandomEmb attack")

    elif attack == "Utility":
        p.add_argument(
            "--ak_data_path", dest="ak.data_path", default="", type=str, help="Path to utility questions JSONL file for Utility attack")
        

        

    return p