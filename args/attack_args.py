


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
        ...
        
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
        
        

    return p