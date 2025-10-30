source set_env.sh
rm -rf ./data/databases/debug_cb
python pipeline.py \
    --des "for debugging" \
    --dataset "HealthCareMagic" \
    --rag "TextRAG" \
    --attack "CopyBreak" \
    --defense "None" \
    --seed 42 \
    --debug \
    --debug_len 100 \
    --gpu 0 \
    \
    --ak_max_query 20 \
    --ak_emb_model "MiniLM" \
    --ak_iterations 3 \
    \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_db_path "debug_cb" \
    --rg_retr_kwargs_topk 5 \
    \
    --ak_attack_template "copybreak/attack_template.txt" \
    --ak_explore_template "copybreak/explore_template.txt" \
    --ak_exploit_template "copybreak/exploit_template.txt" \
    --ak_num_of_each_reason 1 \
    --ak_exchange_rate 3 \
