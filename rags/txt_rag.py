import os
import logging

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate

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
    
    
    
    def index_content(self, dataset, debug_len=None):
        """
        Index raw contents into database.
        """
        persist_dir = os.path.join(os.environ["DB_PATH"], self.db_path)
        os.makedirs(persist_dir, exist_ok=True)

        # Embed and store each document
        documents = []
        seen_docs = set()
        for qa in dataset.qa_pairs:
            qid, cid = qa['query'], qa['content']
            q, c = dataset.id2q[str(qid)], dataset.id2c[str(cid)]
            if qid in seen_docs:
                continue
            seen_docs.add(qid)
            metadata = {'content': c, "index":qid}  # based on query embeddings to retrieve
            documents.append(Document(page_content=q, metadata=metadata))

        if debug_len is not None:
            documents = documents[:debug_len]

        # Create ChromaDB client
        self.db = Chroma.from_documents(documents, self.retriever, 
                                        persist_directory=persist_dir, collection_name="v_db")
        logging.info("Documents added and vector store persisted at " + persist_dir)


    def get_response(self, query):
        """
        End-to-end QA. 
        """
        retrieved_docs = self.retrieve(query)
        prompt = ChatPromptTemplate.from_template(self.gen_kwargs.template).format(Context=retrieved_docs, query=query)
        
        messages = [
            {'role': 'system', 'content': self.gen_kwargs.system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.generator(messages, temperature=self.gen_kwargs.temperature)
        return response, retrieved_docs


    def retrieve(self, query):
        """
        Retrieve relevant documents for a given query. Support topk or threshold based retrieval.
        """
        if "topk" in vars(self.retr_kwargs):
            topk = self.retr_kwargs.topk
            retrieved_docs = self.db.similarity_search_with_score(query, k=topk)
            docs = retrieved_docs

        elif "threshold" in vars(self.retr_kwargs):
            threshold = self.retr_kwargs.threshold
            retrieved_docs = self.db.similarity_search_with_score(query, k=50)
            docs = [doc for doc in retrieved_docs if doc[1] >= threshold] if retrieved_docs else []

        else:
            raise NotImplementedError("Only support topk or threshold based retrieval.")
            
        return docs