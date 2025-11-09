source set_env.sh
python pipeline.py \
    --des "Main experiment for IKEA" \
    --dataset "Pokemon" \
    --rag "TextRAG" \
    --attack "IKEA" \
    --defense "Summary" \
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
    --df_summary_prompt "defense/summary_abstract.txt" \
    \
    --ak_max_query 200 \
    --ak_attack_llm "gpt4o-mini" \
    --ak_attack_emb_model "MiniLM" \
    --ak_topic_word "pokemon" \
    --ak_num_anchors 50 \
    --ak_anchor_gen_template "ikea/anchor_gen_template.txt" \
    --ak_query_gen_iterations 5 \
    --ak_thresh_sim_topic 0.3 \
    --ak_thresh_dissim_anchor 0.5 \
    --ak_thresh_q_anchor 0.7 \
    --ak_sample_temperature 1.0 \
    --ak_anchor_query_gen_template "ikea/anchor_query_gen_template.txt" \
    --ak_thresh_irrelevant 0.7 \
    --ak_thresh_outlier 0.7 \
    --ak_penalty_refusal 10.0 \
    --ak_penalty_irrelevant 7.0 \
    --ak_thresh_qy_sim 0.5 \
    --ak_gamma 0.5 \
    --ak_anchor_mutate_gen_template "ikea/anchor_mutate.txt" \
    --ak_thresh_stop_q 0.6 \
    --ak_thresh_stop_y 0.6
