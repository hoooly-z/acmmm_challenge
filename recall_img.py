import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
import gc
import multiprocessing as mp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


JSON_PATH = "data/result/query_topk_news_images_with_id_v3.json"
IMG_DIR = "data/database/database_images_compressed90"
MODEL_PATH = "BAAI/BGE-VL-MLLM-S1"
SUBMISSION_CSV = "data/result/submission_3.csv"
TOPK = 10
N_GPU = torch.cuda.device_count()  

if N_GPU == 0:
    raise RuntimeError("No GPU detected! This script requires at least one CUDA device.")

print(f"[INFO] Detected {N_GPU} GPUs.")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    all_results = json.load(f)

def debug_image_loading(img_ids, img_dir):
    valid_imgs, valid_ids = [], []
    for imgid in img_ids:
        img_path = os.path.join(img_dir, imgid + ".jpg")
        if not os.path.exists(img_path):
            print(f"[WARN] Image NOT FOUND: {img_path}")
            continue
        valid_imgs.append(img_path)
        valid_ids.append(imgid)
    return valid_imgs, valid_ids

def get_embedding(output):

    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
        tensor = output.hidden_states[-1]
    elif hasattr(output, "last_hidden_state"):
        tensor = output.last_hidden_state
    elif isinstance(output, tuple) and hasattr(output[0], "ndim"):
        tensor = output[0]
    else:
        raise RuntimeError(f"Cannot extract embedding from output type {type(output)}")
    if tensor.ndim == 3:
        tensor = tensor[:, -1, :]
    return tensor

def worker(gpu_id, query_chunk, out_csv, out_score_csv):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_PATH, cache_dir="./qwen", trust_remote_code=True
    ).to(device).eval()
    model.set_processor(MODEL_PATH)
    if getattr(model.processor, "patch_size", None) is None:
        model.processor.patch_size = 14
    if hasattr(model.processor, "image_processor") and getattr(model.processor.image_processor, "patch_size", None) is None:
        model.processor.image_processor.patch_size = 14
    if not hasattr(model.processor, "vision_feature_select_strategy") or model.processor.vision_feature_select_strategy is None:
        model.processor.vision_feature_select_strategy = "default"

    rows = []
    top1_scores = []
    for item in tqdm(query_chunk, desc=f"[GPU {gpu_id}] Query retrieval"):
        query_id = item.get("query_id", -1)
        query_text = item.get("query_text", "")
        img_entries = item.get("retrieved_images", [])
        img_ids = [img['image_id'] for img in img_entries if img.get('image_id') not in [None, "#", ""]]

        valid_imgs, valid_img_ids = debug_image_loading(img_ids, IMG_DIR)
        if len(valid_imgs) == 0:
            rows.append([query_id] + ["#"] * TOPK)
            top1_scores.append({"query_id": query_id, "top1_score": -1e9, "top1_img_id": "#"})
            continue

        try:
            with torch.no_grad():
                query_inputs = model.data_process(
                    text=query_text,
                    q_or_c="q",
                    task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
                )
                query_emb_raw = model(**query_inputs, output_hidden_states=True)
                query_emb = get_embedding(query_emb_raw)
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
                del query_inputs
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FATAL][GPU {gpu_id}] Query embedding failed for {query_id}: {repr(e)}")
            rows.append([query_id] + ["#"] * TOPK)
            top1_scores.append({"query_id": query_id, "top1_score": -1e9, "top1_img_id": "#"})
            continue

        all_scores = []
        all_valid_ids = []
        for img_path, img_id in zip(valid_imgs, valid_img_ids):
            try:
                with torch.no_grad():
                    candidate_inputs = model.data_process(
                        images=[img_path],
                        q_or_c="c",
                    )
                    candi_emb_raw = model(**candidate_inputs, output_hidden_states=True)
                    candi_emb = get_embedding(candi_emb_raw)
                    candi_emb = torch.nn.functional.normalize(candi_emb, dim=-1)
                    score = torch.matmul(query_emb, candi_emb.T).squeeze().item()
                    all_scores.append(score)
                    all_valid_ids.append(img_id)
                    del candi_emb, candidate_inputs
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[FATAL][GPU {gpu_id}] Image embedding/scoring failed for img {img_id}: {repr(e)}")
                all_scores.append(-1e9)
                torch.cuda.empty_cache()

        if not all_scores:
            rows.append([query_id] + ["#"] * TOPK)
            top1_scores.append({"query_id": query_id, "top1_score": -1e9, "top1_img_id": "#"})
            continue

        scores = np.array(all_scores)
        order = np.argsort(-scores)[:TOPK]
        pred_top_img_ids = [all_valid_ids[i] for i in order]
        if len(pred_top_img_ids) < TOPK:
            pred_top_img_ids += ["#"] * (TOPK - len(pred_top_img_ids))
        row = [query_id] + pred_top_img_ids
        rows.append(row)

        top1_scores.append({
            "query_id": query_id,
            "top1_score": float(scores[order[0]]),
            "top1_img_id": pred_top_img_ids[0] if pred_top_img_ids else "#"
        })

        torch.cuda.empty_cache()

    df_rows = pd.DataFrame(rows, columns=["query_id"] + [f"image_id_{i+1}" for i in range(TOPK)])
    df_rows.to_csv(out_csv, index=False)
    df_scores = pd.DataFrame(top1_scores)
    df_scores.to_csv(out_score_csv, index=False)
    print(f"[GPU {gpu_id}] Done. Saved {out_csv} and {out_score_csv}")

if __name__ == "__main__":
    N = len(all_results)
    chunk_size = (N + N_GPU - 1) // N_GPU
    chunks = [all_results[i*chunk_size:(i+1)*chunk_size] for i in range(N_GPU)]

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        out_csv = f"data/result/submission_gpu{gpu_id}.csv"
        out_score_csv = f"data/result/top1_scores_gpu{gpu_id}.csv"
        p = mp.Process(target=worker, args=(gpu_id, chunk, out_csv, out_score_csv))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # 合并csv
    all_dfs = []
    all_score_dfs = []
    for gpu_id in range(N_GPU):
        all_dfs.append(pd.read_csv(f"data/result/submission_gpu{gpu_id}.csv"))
        all_score_dfs.append(pd.read_csv(f"data/result/top1_scores_gpu{gpu_id}.csv"))
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(SUBMISSION_CSV, index=False)
    df_score_all = pd.concat(all_score_dfs, ignore_index=True)
    df_score_all.to_csv("data/result/query_top1_scores.csv", index=False)
    print(f"[INFO] All done. Final merged csv: {SUBMISSION_CSV}")
