source set_env.sh
python pipeline.py \
    --des "for debugging" \
    --dataset "HealthCareMagic" \
    --rag "TextRAG" \
    --attack "DGEA" \
    --defense "None" \
    --seed 42 \
    --debug \
    --debug_len 100 \
    --gpu 0 \
    \
    --ak_max_query 5 \
    --ak_emb_model "MiniLM" \
    --ak_iterations 3 \
    \
    --rg_db_path "debug" \
    \
    --rg_gen_kwargs_system_prompt "dgea/rag_system.txt" \
    --rg_gen_kwargs_template "dgea/rag_template.txt" \
    --ak_llm_kwargs_system_prompt "dgea/extract_system.txt" \
    --ak_llm_kwargs_template "dgea/extract_template.txt" \
    --ak_llm_kwargs_example "dgea/extract_example.txt" \
    --ak_command_prompt "dgea/ak_prefix.txt" \
    --ak_info_prompt "dgea/ak_suffix.txt" \
    \
    --ak_random_vec "dgea/embedding_statistics.csv" \
