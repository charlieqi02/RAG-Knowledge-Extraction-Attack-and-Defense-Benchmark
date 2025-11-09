source set_env.sh
python pipeline.py \
    --des "Main experiment: RandomText Attack" \
    --dataset "HarryPotter" \
    --rag "TextRAG" \
    --attack "RandomText" \
    --defense "None" \
    --seed 42 \
    --gpu 0 \
    \
    --rg_db_path "harrypotter_26k" \
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
    --ak_attack_template "copybreak/attack_template.txt" \
    --ak_llm_model "gpt4o-mini" \
    --ak_system_prompt "random/gen_system.txt" \
    --ak_template "random/gen_template.txt" \
    --ak_temperature 0.9