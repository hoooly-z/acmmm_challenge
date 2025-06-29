import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import numpy as np
from tqdm import tqdm


GT_CSV = "data/train/gt_train.csv"  # 
FAISS_INDEX_PATH = "data/database/faiss_content_bgem3.index" 
ID_MAP_PATH = "data/database/faiss_content_bgem3.index.id_map.json"  
NEWS_JSON_PATH = "data/database/database.json"  # 

topk = 100       # 
max_count = 2000  # 
use_reranker = True  #


df = pd.read_csv(GT_CSV)
if max_count is not None:
    df = df.head(max_count)
queries = df["caption"].astype(str).tolist()
gt_ids = df["retrieved_article_id"].astype(str).tolist()


print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
print("Encoding queries...")
query_vecs = model.encode(queries, show_progress_bar=True, normalize_embeddings=True)
print("Query embeddings shape:", query_vecs.shape)


print("Loading FAISS index and id map...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    news_ids = json.load(f)


with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    if isinstance(raw_data, dict):
        news_map = {k: v["content"] for k, v in raw_data.items()}
    else:
      
        news_map = {item["news_id"]: item["content"] for item in raw_data}


print(f"Searching Top-{topk}...")
D, I = faiss_index.search(query_vecs, topk)


if use_reranker:
    print("Loading BGE reranker...")
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
    rerank_topk = topk  #

    print("Applying reranker to Top-K results...")
    reranked_indices = []
    for idx, (q, topk_idx) in tqdm(enumerate(zip(queries, I)), total=len(queries)):
        candidate_ids = [str(news_ids[i]) for i in topk_idx]
        candidate_texts = [news_map.get(cid, "") for cid in candidate_ids]

        pairs = [[q, doc] for doc in candidate_texts]
        scores = reranker.predict(pairs)
        order = np.argsort(scores)[::-1]  

   
        reranked_indices.append([topk_idx[o] for o in order])
    I = np.array(reranked_indices)


hits_at_k = 0
hits_at_1 = 0

for qidx, indices in enumerate(I):
    pred_ids = [str(news_ids[i]) for i in indices]
    gt = str(gt_ids[qidx])
    if gt in pred_ids:
        hits_at_k += 1
    if gt == pred_ids[0]:
        hits_at_1 += 1

print(f"Total queries: {len(gt_ids)}")
print(f"Top-{topk} Recall: {hits_at_k / len(gt_ids):.4f}")
print(f"Top-1 Recall: {hits_at_1 / len(gt_ids):.4f}")

print("\n--- Some retrieval failures (ground truth missed in Top-K) ---")
cnt = 0
for qidx, indices in enumerate(I):
    pred_ids = [str(news_ids[i]) for i in indices]
    gt = str(gt_ids[qidx])
    if gt not in pred_ids:
        print(f"\nQuery: {queries[qidx][:120]}")
        print(f"Ground Truth ID: {gt}")
        print(f"Top-{topk} Predicted: {pred_ids}")
        cnt += 1
        if cnt >= 5:
            break

print("\nDone.")