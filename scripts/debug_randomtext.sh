source set_env.sh
rm -rf ./data/databases/debug
python pipeline.py \
    --des "for debugging" \
    --dataset "HarryPotter" \
    --rag "TextRAG" \
    --attack "RandomText" \
    --defense "None" \
    --seed 42 \
    --gpu 0 \
    \
    --ak_max_query 5 \
    \
    --rg_generator "gpt4o-mini" \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_db_path "debug" \
    --rg_retr_kwargs_topk 3 \
    \
    --ak_llm_model "gpt4o-mini" \
    --ak_attack_template "random/attack_template.txt" \
    --ak_temperature 0.7 \
    --ak_template "random/gen_template.txt" \
    --ak_system_prompt "random/gen_system.txt" \