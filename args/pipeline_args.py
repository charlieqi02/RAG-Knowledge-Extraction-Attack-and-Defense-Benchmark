

def get_pipeline_args(all_rags, all_attacks, all_defenses):
    def pipe_args(p):
        # General arguments
        p.add_argument(
            "--des", default="", type=str, help="Description of the run, will be saved in config.json")
        p.add_argument(
            "--dataset", default="HealthCareMagic", choices=["Enron", "HealthCareMagic"], help="Knowledge Extraction dataset")
        p.add_argument(
            "--rag", choices=all_rags, default="TextRAG", help="RAG systems for knowledge extraction attack & defense")
        p.add_argument(
            "--attack", choices=all_attacks, default="DGEA", help="Knowledge extraction attack methods; run RAG benchmarking if None")
        p.add_argument(
            "--defense", choices=all_defenses, default="None", help="Knowledge extraction defense methods")

        p.add_argument(
            "--continue_pipe", default=False, action='store_true', help="If use checkpoint to continue")
        p.add_argument(
            "--continue_dir", default="", type=str, help="Dir to use checkpoint to continue")
        p.add_argument(
            "--seed", default=3407, type=int, help="Random seed")
        p.add_argument(
            "--gpu", type=int, default=-1, help="gpu")

        p.add_argument(
            "--debug", default=False, action='store_true', help="Decrease docs in database for debugging")
        p.add_argument(
            "--debug_len", default=100, type=int, help="Decreased docs in database for debugging")

        return p

    return pipe_args