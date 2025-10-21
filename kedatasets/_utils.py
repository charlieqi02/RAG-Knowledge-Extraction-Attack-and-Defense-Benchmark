import re 
import logging



def split_text(text, pattern):
    if text is None:
        return "", ""
    
    s = str(text).replace("\r\n", "\n").strip()
    m = pattern.search(s)
    if m:
        human = m.group(1).strip()
        bot = m.group(2).strip()
        return human, bot
    else:
        # 兜底逻辑：尝试按首次出现的 "<bot>:" 
        idx = s.find("<bot>:")
        if idx != -1:
            human = s[:idx].replace("<human>:", "").strip()
            bot = s[idx + len("<bot>:"):].strip()
            return human, bot
        # 如果仍失败，保留原文到 output，input 设为空
        
        logging.warning(f"dataset parse wrong: {text}")
        return "", s
    
    
def index_qas(qa_pairs):
    unique_queries = []
    unique_contents = []
    q2id = {}
    c2id = {}
    id2q = {}
    id2c = {}
    qa_id = []

    for qa in qa_pairs:
        q = qa["query"]
        c = qa["content"]

        # 处理 query id
        if q not in q2id:
            q_id = str(len(unique_queries))
            q2id[q] = q_id
            id2q[q_id] = q
            unique_queries.append(q)
        else:
            q_id = q2id[q]

        # 处理 content id
        if c not in c2id:
            c_id = str(len(unique_contents))
            c2id[c] = c_id
            id2c[c_id] = c
            unique_contents.append(c)
        else:
            c_id = c2id[c]

        qa_id.append({"query": q_id, "content": c_id})

    return unique_queries, unique_contents, id2q, id2c, qa_id