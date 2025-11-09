source set_env.sh
python pipeline.py \
    --des "Main experiment: CopyBreak Attack" \
    --dataset "Enron" \
    --rag "TextRAG" \
    --attack "CopyBreak" \
    --defense "Threshold" \
    --seed 42 \
    --gpu 0 \
    \
    --rg_db_path "enron_500k" \
    --rg_retriever "MiniLM" \
    --rg_generator "gpt4o-mini" \
    --rg_device "cuda:0" \
    --rg_retr_kwargs_topk 3 \
    --rg_role "email assistant" \
    --rg_gen_kwargs_system_prompt "textrag/system.txt" \
    --rg_gen_kwargs_template "textrag/template.txt" \
    --rg_gen_kwargs_temperature 0.1 \
    \
    --df_threshold 0.5 \
    \
    --ak_max_query 200 \
    --ak_llm_model "gpt4o-mini" \
    --ak_emb_model "MiniLM" \
    --ak_attack_template "copybreak/attack_template.txt" \
    --ak_sim_thresh 0.6 \
    --ak_iterations 10 \
    --ak_explore_template "copybreak/explore_template.txt" \
    --ak_exploit_template "copybreak/exploit_template.txt" \
    --ak_exchange_rate 5 \
    --ak_num_of_each_reason 2 \
    --ak_explore_temperature 0.7 \
    --ak_exploit_temperature 0.3


