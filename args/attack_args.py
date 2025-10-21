


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
            "--ak_parser_llm", dest="ak.parser_llm", default="gpt4o-mini", type=str, help="LLM for parsing the response to extract info")
        p.add_argument(
            "--ak_llm_kwargs_temperature", dest="ak.llm_kwargs.temperature", default=0.0, type=float, help="LLM temperature for parsing the response to extract info")

        p.add_argument(
            "--ak_llm_kwargs_system_prompt", dest="ak.llm_kwargs.system_prompt", default="default", type=str, help="System prompt path for DGEA")
        p.add_argument(
            "--ak_llm_kwargs_template", dest="ak.llm_kwargs.template", default="default", type=str, help="Template prompt path for DGEA")
        p.add_argument(
            "--ak_llm_kwargs_example", dest="ak.llm_kwargs.example", default="default", type=str, help="Example prompt path for DGEA")

        p.add_argument(
            "--ak_command_prompt", dest="ak.command_prompt", default="default", type=str, help="Command prompt path for DGEA")
        p.add_argument(
            "--ak_info_prompt", dest="ak.info_prompt", default="default", type=str, help="Info prompt path for DGEA")

        p.add_argument(
            "--ak_random_vec", dest="ak.random_vec", default="", type=str, help="Random vector distribution file path for DGEA")
            

    return p