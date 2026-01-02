import os
import logging

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
import openai
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .base import RAGSystem


TXT_RAGS = ["TextRAG"]

class TextRAG(RAGSystem):
    """
    Normal text RAG system.
    """
    def __init__(self, args):
        super().__init__(args)
        
        prompt_dir = os.environ.get("PROMPT_PATH")
        with open(os.path.join(prompt_dir, self.gen_kwargs.template), "r") as f:
            self.gen_kwargs.template = f.read()
        with open(os.path.join(prompt_dir, self.gen_kwargs.system_prompt), "r") as f:
            self.gen_kwargs.system_prompt = f.read()
        self.system_prompt = self.gen_kwargs.system_prompt.replace("{role}", args.role)

        # for query-block defense
        self.query_blocker = getattr(args, "query_blocker", None)


    def index_content(self, dataset):
        """
        Index raw contents into database.
        If a database already exists, load it and compare document count.
        """
        index_content = dataset.index_content  # dict: id -> content
        persist_dir = os.path.join(os.environ["DB_PATH"], self.db_path)
        os.makedirs(persist_dir, exist_ok=True)

        expected_count = len(index_content)
        logging.info(f"Preparing to index {expected_count} documents into {persist_dir}")

        # 判断是否已有数据库
        db_exists = any(name.endswith(".sqlite3") for name in os.listdir(persist_dir))

        if db_exists:
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.retriever,
                collection_name="v_db"
            )
            current_count = len(db.get()["ids"])
            if current_count == expected_count:
                logging.info(f"✅ Database already exists, document count matches ({current_count}).")
            else:
                logging.warning(f"⚠️ Database exists but count mismatch: existing {current_count}, expected {expected_count}.")
            self.db = db
            return

        # --- 如果数据库还不存在，创建并分批插入 ---
        # 先把所有文本和元数据准备好
        texts = []
        metadatas = []
        for cid, content in index_content.items():
            texts.append(content)
            metadatas.append({"index": cid})

        logging.info("Creating new Chroma collection (empty)")
        db = Chroma(
            embedding_function=self.retriever,
            persist_directory=persist_dir,
            collection_name="v_db"
        )

        # 分批 upsert，避免一次塞太多
        batch_size = 5000  # 安全批大小，小于 5461 的限制
        n = len(texts)
        logging.info(f"Indexing documents in batches of {batch_size}...")
        for start in tqdm(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            batch_metas = metadatas[start:end]
            # logging.info(f"Upserting batch {start} ~ {end-1} ({len(batch_texts)} docs)")
            db.add_texts(texts=batch_texts, metadatas=batch_metas)

        # 持久化到磁盘
        db.persist()
        logging.info(f"✅ New database created and persisted at {persist_dir} with {n} documents.")

        self.db = db
    
    
    
    def get_response(self, query):
        """
        End-to-end QA. 
        """
        if self.query_blocker is not None:
            block_signal = self.query_blocker.detect(query)
            if not block_signal:    # False == blocked
                return "ACCESS DENIED: Potential extraction attack detected.", []
        
        retrieved_docs = self.retrieve(query)
        context_list = [f"context {i}: \n\"{doc[0].page_content}\"" for i, doc in enumerate(retrieved_docs)]
        context = "\n\n".join(context_list)
        logging.info(f"Retrieved {len(retrieved_docs)} documents for the query.")
        prompt = ChatPromptTemplate.from_template(self.gen_kwargs.template).format(context=context, query=query)
        
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        try: 
            response = self.generator(messages, temperature=self.gen_kwargs.temperature)
        except:
            logging.error(f"Too long error")
            logging.info("prompt length: %d", len(prompt))
            logging.info("prompt content: %s", prompt[:100])
            response = "Error: The generated prompt is too long for the model to process." 
        return response, retrieved_docs


    def retrieve(self, query):
        """
        Retrieve relevant documents for a given query. Support topk or threshold based retrieval.
        """
        if not "threshold" in vars(self.retr_kwargs):
            topk = self.retr_kwargs.topk
            retrieved_docs = self.db.similarity_search_with_score(query, k=topk)
            docs = retrieved_docs

        elif "threshold" in vars(self.retr_kwargs):
            threshold = self.retr_kwargs.threshold
            topk = self.retr_kwargs.topk
            retrieved_docs = self.db.similarity_search_with_score(query, k=topk)
            docs = []
            
            if retrieved_docs:
                # 假设 retrieved_docs = [(Document, score), ...]
                # 计算 query 向量
                query_emb = self.retriever._embed(query)
                query_emb = torch.tensor(query_emb, dtype=torch.float32)
                query_emb = F.normalize(query_emb, p=2, dim=0)

                for doc, _ in retrieved_docs:
                    doc_emb = self.retriever._embed(doc.page_content)
                    doc_emb = torch.tensor(doc_emb, dtype=torch.float32)
                    doc_emb = F.normalize(doc_emb, p=2, dim=0)

                    cos_sim = torch.dot(query_emb, doc_emb).item()
                    if cos_sim >= threshold:
                        docs.append((doc, cos_sim))
        else:
            raise NotImplementedError("Only support topk or threshold based retrieval.")
            
        return docs