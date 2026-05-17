source set_env.sh
python pipeline.py \
    --des "Utility Measure" \
    --dataset "Pokemon" \
    --rag "TextRAG" \
    --attack "Utility" \
    --defense "None" \
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
    --ak_max_query 1000 \
    --ak_data_path "./data/Pokemon" \
    "$@"
