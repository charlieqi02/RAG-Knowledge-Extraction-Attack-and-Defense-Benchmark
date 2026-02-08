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
        # Fallback: try splitting at first occurrence of "<bot>:"
        idx = s.find("<bot>:")
        if idx != -1:
            human = s[:idx].replace("<human>:", "").strip()
            bot = s[idx + len("<bot>:"):].strip()
            return human, bot
        # If still fails, keep original text as output, set input to empty
        
        logging.warning(f"dataset parse wrong: {text}")
        return "", s
    
    
def index_qas(qa_pairs):
    unique_queries = []
    unique_answers = []
    q2id = {}
    a2id = {}
    id2q = {}
    id2a = {}
    qa_id = []

    for qa in qa_pairs:
        q = qa["query"]
        c = qa["answer"]

        # Process query id
        if q not in q2id:
            q_id = str(len(unique_queries))
            q2id[q] = q_id
            id2q[q_id] = q
            unique_queries.append(q)
        else:
            q_id = q2id[q]

        # Process answer id
        if c not in a2id:
            c_id = str(len(unique_answers))
            a2id[c] = c_id
            id2a[c_id] = c
            unique_answers.append(c)
        else:
            c_id = a2id[c]

        qa_id.append({"query": q_id, "answer": c_id})

    return unique_queries, unique_answers, id2q, id2a, qa_id