import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import numpy as np
from tqdm import tqdm


QUERY_CSV = "data/track2/private_set/query.csv"
FAISS_INDEX_PATH = "data/database/faiss_content_bgem3.index"
ID_MAP_PATH = "data/database/faiss_content_bgem3.index.id_map.json"
NEWS_JSON_PATH = "data/database/database.json"
TOPK = 3
OUT_JSON = "data/result/query_topk_news_images_v3.json"

df = pd.read_csv(QUERY_CSV)
queries = df["query_text"].astype(str).tolist()

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
print("Encoding queries...")
query_vecs = model.encode(queries, show_progress_bar=True, normalize_embeddings=True)

print("Loading FAISS index and id map...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    news_ids = json.load(f)


with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    if isinstance(raw_data, dict):
        news_map = {k: v for k, v in raw_data.items()}
    else:
        news_map = {item["news_id"]: item for item in raw_data}


print("Loading reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3",cache_dir="./qwen", device="cuda")

print(f"Searching Top-{TOPK} and reranking...")
results = []
for qidx, indices in tqdm(enumerate(I := faiss_index.search(query_vecs, TOPK)[1]), total=len(queries)):
    pred_news_ids = [str(news_ids[i]) for i in indices]
    candidate_texts = [news_map[nid].get("content", "") for nid in pred_news_ids]


    pairs = [[queries[qidx], t] for t in candidate_texts]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1]
    reranked_news_ids = [pred_news_ids[i] for i in order]


    retrieved_imgs = []
    for news_id in reranked_news_ids:
        images = news_map.get(news_id, {}).get("images", [])
        for img in images:
            retrieved_imgs.append({"news_id": news_id, "image_id": img})

    seen_imgs = set()
    filtered_imgs = []
    for pair in retrieved_imgs:
        if pair["image_id"] not in seen_imgs:
            filtered_imgs.append(pair)
            seen_imgs.add(pair["image_id"])

    results.append({
        "query_id": qidx,
        "query_text": queries[qidx],
        "topk_news_ids": reranked_news_ids,
        "retrieved_images": filtered_imgs,
    })


with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved recall results to {OUT_JSON}")
