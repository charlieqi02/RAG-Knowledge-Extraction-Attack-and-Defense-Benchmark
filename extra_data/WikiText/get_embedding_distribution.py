import os
import pandas as pd
import numpy as np
import torch

from tools.get_embedding import get_embedding


wikitext_path = "./extra_data/WikiText/wikitext_sample_5000.csv"
embedding_models = ["MiniLM", "GTE-base", "BGE-large"]


def get_statistics(embeddings, embed_model):
    """
    Calculate and save statistics of the given embeddings.
    
    Args:
        embeddings (torch.Tensor): The sentence embeddings.
        savePath (str): The directory path to save the statistics CSV file.
    """
    mean = embeddings.mean(dim=0)
    variance = embeddings.var(dim=0)
    
    stats_df = pd.DataFrame({
        'mean': mean.tolist(),
        'variance': variance.tolist()
    })
    save_path = f'./extra_data/embedding_statistics_wiki_{embed_model}.csv'
    stats_df.to_csv(save_path, index=False)
    print(f'Statistics saved successfully to {save_path}')



def compute_embeddings_for_model(texts, embed_model):
    """
    Use `get_embedding` to compute embeddings for all texts with batching.
    
    Args:
        texts (List[str]): List of sentences.
        embed_model (str): Embedding model name.
        batch_size (int): Batch size for embedding computation.

    Returns:
        torch.Tensor: All sentence embeddings concatenated. Shape [N, D]
    """
    embed_model = get_embedding(embed_model)
    all_embeddings = []
    n = len(texts)

    for i in range(0, n):
        # 假设 get_embedding 接口：get_embedding(list_of_texts, embed_model)
        embedding = embed_model._embed(texts[i])  # 获取单个文本的 embedding
        all_embeddings.append(torch.tensor(embedding).unsqueeze(0))  # [1, D]

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    return embeddings


def main():
    # 1. 读取 wikitext 样本
    if not os.path.exists(wikitext_path):
        raise FileNotFoundError(f"{wikitext_path} not found")

    df = pd.read_csv(wikitext_path)
    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")

    texts = df["text"].dropna().astype(str).tolist()
    print(f"Loaded {len(texts)} texts from {wikitext_path}")

    # 2. 依次对每个 embedding model 计算 & 保存统计信息
    for embed_model in embedding_models:
        print(f"\n=== Processing model: {embed_model} ===")
        embeddings = compute_embeddings_for_model(texts, embed_model)
        print(f"[{embed_model}] Embeddings shape: {embeddings.shape}")
        get_statistics(embeddings, embed_model)


if __name__ == "__main__":
    main()