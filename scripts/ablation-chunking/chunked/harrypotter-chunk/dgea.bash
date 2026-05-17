source set_env.sh
# Defaults to the legacy "hp" db; pass --dataset HarryPotterQAChunk --rg_db_path harrypotter_qa_chunk
# (or HarryPotterBookChunk / harrypotter_book_chunk) to switch chunked variants.
python pipeline.py \
    --des "DGEA Attack (chunked HarryPotter)" \
    --dataset "HarryPotter" \
    --rag "TextRAG" \
    --attack "DGEA" \
    --defense "None" \
    --seed 42 \
    --gpu 0 \
    \
    --rg_db_path "hp" \
    --rg_retriever "MiniLM" \
    --rg_generator "gpt4o-mini" \
    --rg_device "cuda:0" \
    --rg_retr_kwargs_topk 3 \
    --rg_role "book assistant" \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_gen_kwargs_temperature 0.1 \
    \
    --ak_max_query 200 \
    --ak_command_prompt "copybreak/attack_template.txt" \
    --ak_emb_model "MiniLM" \
    --ak_iterations 3 \
    --ak_info_prompt "dgea/ak_suffix.txt" \
    --ak_pool_size 512 \
    --ak_random_vec "embedding_statistics.csv" \
    "$@"
