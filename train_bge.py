import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader
import faiss
import random


GT_CSV = "data/train/gt_train.csv"
NEWS_JSON_PATH = "data/database/database.json"
ID_MAP_PATH = "data/database/faiss_content_bgem3.index.id_map.json"
FAISS_INDEX_PATH = "data/database/faiss_content_bgem3.index"


TOPK = 10
HARD_NEG_K = 5    
BATCH_SIZE = 8
EPOCHS = 2
MAX_DATA = None     


def ensure_numpy2d_float32(vec):

    import numpy as np
    import torch
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return vec


df = pd.read_csv(GT_CSV)
if MAX_DATA:
    df = df.head(MAX_DATA)

with open(NEWS_JSON_PATH, "r", encoding="utf-8") as f:
    news_raw = json.load(f)
if isinstance(news_raw, dict):
    news_map = {k: v["content"] for k, v in news_raw.items()}
else:
    news_map = {item["news_id"]: item["content"] for item in news_raw}

with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    news_ids = json.load(f)
id2idx = {str(news_ids[i]): i for i in range(len(news_ids))}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"[INFO] Train: {len(train_df)}, Test: {len(test_df)}")


embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)


train_examples = []
print("[INFO] Building training pairs...")
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    q = str(row["caption"])
    gt_id = str(row["retrieved_article_id"])
    gt_doc = news_map.get(gt_id, "")
    if not gt_doc:
        continue


    train_examples.append(InputExample(texts=[q, gt_doc], label=1.0))

    q_vec = embedder.encode(q, normalize_embeddings=True)
    q_vec = ensure_numpy2d_float32(q_vec)
    D, I = faiss_index.search(q_vec, HARD_NEG_K + 1)

    top_indices = I[0]
    for idx in top_indices:
        cand_id = str(news_ids[idx])
        if cand_id == gt_id:
            continue 
        neg_doc = news_map.get(cand_id, "")
        if neg_doc:
            train_examples.append(InputExample(texts=[q, neg_doc], label=0.0))

print(f"[INFO] Total train samples: {len(train_examples)}")


test_queries = test_df["caption"].astype(str).tolist()
test_gt_ids = test_df["retrieved_article_id"].astype(str).tolist()


print("[INFO] Start CrossEncoder fine-tuning...")
model = CrossEncoder("BAAI/bge-reranker-base", num_labels=1, device="cuda")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
model.fit(train_dataloader=train_dataloader, epochs=EPOCHS, show_progress_bar=True)
model.save("output/my-finetuned-reranker")


hits_at_k = 0
hits_at_1 = 0

for qidx, query in tqdm(enumerate(test_queries), total=len(test_queries)):
    gt_id = test_gt_ids[qidx]

 
    q_vec = embedder.encode(query, normalize_embeddings=True)
    q_vec = ensure_numpy2d_float32(q_vec)
    D, I = faiss_index.search(q_vec, TOPK)

    candidate_ids = [str(news_ids[i]) for i in I[0]]
    candidate_texts = [news_map.get(cid, "") for cid in candidate_ids]

   
    pairs = [[query, doc] for doc in candidate_texts]
    scores = model.predict(pairs)
    order = np.argsort(scores)[::-1]
    pred_ids = [candidate_ids[i] for i in order]

    if gt_id in pred_ids:
        hits_at_k += 1
    if gt_id == pred_ids[0]:
        hits_at_1 += 1

print(f"Test samples: {len(test_queries)}")
print(f"Top-{TOPK} Recall: {hits_at_k / len(test_queries):.4f}")
print(f"Top-1 Recall: {hits_at_1 / len(test_queries):.4f}")