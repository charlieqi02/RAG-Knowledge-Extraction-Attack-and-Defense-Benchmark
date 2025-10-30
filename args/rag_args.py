


def get_rag_args(p, rag):
    
    if rag == "TextRAG":
        p.add_argument(
            "--rg_db_path", dest="rg.db_path", default="txt_rag_db", type=str, help="Database path for RAG")
        p.add_argument(
            "--rg_retriever", dest="rg.retriever", default="MiniLM", type=str, help="Retriever model for RAG")
        p.add_argument(
            "--rg_generator", dest="rg.generator", default="gpt4o-mini", type=str, help="Generator model for RAG")
        p.add_argument(
            "--rg_device", dest="rg.device", default="cuda", type=str, help="Device for RAG")
        p.add_argument(
            "--rg_gen_kwargs_temperature", dest="rg.gen_kwargs.temperature", default=0.1, type=float, help="Generation temperature.")
        p.add_argument(
            "--rg_retr_kwargs_topk", dest="rg.retr_kwargs.topk", default=5, type=int, help="Retriever top-k documents.")
        
        p.add_argument(
            "--rg_gen_kwargs_template", dest="rg.gen_kwargs.template", default="default_template.txt", type=str, help="Gneration prompt template path.")
        p.add_argument(
            "--rg_gen_kwargs_system_prompt", dest="rg.gen_kwargs.system_prompt", default="default_system_prompt.txt", type=str, help="Generation system prompt path.")
        p.add_argument(
            "--rg_role", dest="rg.role", default="medical assistant", type=str, help="Role to be filled in the system prompt.")
        