import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiniLMEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.dim = 384

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_output = self.model(**inputs)
        sentence_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().detach().numpy().flatten().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class nomicEmbeddings:
    def __init__(self, device="cuda", matryoshka_dim=384):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True,
                                               safe_serialization=True).to(device)
        self.model.eval()
        self.device = device
        self.matryoshka_dim = matryoshka_dim

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :self.matryoshka_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Gte_smallEmbeddings:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        self.model = AutoModel.from_pretrained("thenlper/gte-small").to(device)
        self.dim = 384
        self.model.eval()
        self.device = device

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        encoded_input = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Gte_baseEmbeddings:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
        self.model = AutoModel.from_pretrained("thenlper/gte-base").to(device)
        self.dim = 768
        self.model.eval()
        self.device = device

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        encoded_input = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Gte_largeEmbeddings:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large").to(device)
        self.dim = 1024
        self.model.eval()
        self.device = device

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        encoded_input = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MpnetEmbeddings:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        self.device = device
        self.dim = 768
        self.model.eval()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class Bge_largeEmbeddings:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device)
        self.dim = 1024
        self.model.eval()
        self.device = device

    def embed_documents(self, texts):
        return [self._embed("passage: " + text) for text in texts]

    def embed_query(self, text):
        return self._embed("query: " + text)

    def _embed(self, text):
        encoded_input = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # BGE officially recommends [CLS] pooling
            sentence_embeddings = model_output.last_hidden_state[:, 0, :]
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().numpy().flatten().tolist()

    def _mean_pooling(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    

# if __name__ == "__main__":
#     model = Bge_largeEmbeddings(device="cuda")
#     texts = ["This is a test sentence.", "Another sentence for embedding."]
#     embeddings = model.embed_documents(texts)
#     for i, emb in enumerate(embeddings):
#         print(f"Text {i}: {texts[i]}")
#         print(f"Embedding {i} dimension: {len(emb)}")
#         print(f"Embedding {i}: {emb[:5]}...")  # Print first 5 dimensions