source set_env.sh
python pipeline.py \
    --des "Main experiment: DGEA Attack" \
    --dataset "Pokemon" \
    --rag "TextRAG" \
    --attack "DGEA" \
    --defense "Threshold" \
    --seed 42 \
    --gpu 0 \
    \
    --rg_db_path "pokemon_1k" \
    --rg_retriever "MiniLM" \
    --rg_generator "gpt4o-mini" \
    --rg_device "cuda:0" \
    --rg_retr_kwargs_topk 3 \
    --rg_role "pokemon assistant" \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_gen_kwargs_temperature 0.1 \
    \
    --df_threshold 0.5 \
    \
    --ak_max_query 200 \
    --ak_command_prompt "copybreak/attack_template.txt" \
    --ak_emb_model "MiniLM" \
    --ak_iterations 3 \
    --ak_info_prompt "dgea/ak_suffix.txt" \
    --ak_pool_size 512 \
    --ak_random_vec "embedding_statistics.csv"