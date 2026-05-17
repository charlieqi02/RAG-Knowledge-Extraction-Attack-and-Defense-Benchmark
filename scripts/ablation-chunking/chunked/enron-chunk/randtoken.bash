source set_env.sh
python pipeline.py \
    --des "RandomToken Attack (chunked Enron)" \
    --dataset "EnronChunk" \
    --rag "TextRAG" \
    --attack "RandomToken" \
    --defense "None" \
    --seed 42 \
    --gpu 0 \
    \
    --rg_db_path "enron-chunk" \
    --rg_retriever "BGE-large" \
    --rg_generator "gpt4o-mini-openai" \
    --rg_device "cuda:0" \
    --rg_retr_kwargs_topk 10 \
    --rg_role "email assistant" \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_gen_kwargs_temperature 0.1 \
    \
    --ak_max_query 200 \
    --ak_attack_template "copybreak/attack_template.txt" \
    --ak_emb_model "MiniLM" \
    --ak_pool_size 25 \
    "$@"
