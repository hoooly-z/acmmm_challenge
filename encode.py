import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoProcessor


IMG_DIR = "data/database/database_images_compressed90"    
MODEL_PATH = "BAAI/BGE-VL-large"  
SAVE_EMB = "data/img/image_embeddings.npy"
SAVE_LIST = "data/img/image_id_list.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  


print("[INFO] Loading model...")
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

img_files = [os.path.join(IMG_DIR, fname)
             for fname in os.listdir(IMG_DIR)
             if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp'))]

print(f"[INFO] Found {len(img_files)} images.")

embeddings = []
id_list = []

for i in tqdm(range(0, len(img_files), BATCH_SIZE), desc="Encoding images"):
    batch_files = img_files[i:i+BATCH_SIZE]
    batch_imgs = []
    batch_ids = []
    for img_path in batch_files:
        try:
            image = Image.open(img_path).convert("RGB")
            batch_imgs.append(image)
            batch_ids.append(os.path.basename(img_path))
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")

    if not batch_imgs:
        continue

    try:

        dummy_texts = ["a picture"] * len(batch_imgs)
        inputs = processor(images=batch_imgs, text=dummy_texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

            emb = out.hidden_states[-1][:, -1, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)
        embeddings.append(emb.cpu().numpy())
        id_list.extend(batch_ids)
        del inputs, out, emb
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[WARN] Failed to process batch {batch_ids}: {e}")

if len(embeddings) == 0:
    raise RuntimeError("No images processed successfully, please check your folder and code.")


embeddings = np.concatenate(embeddings, axis=0)
np.save(SAVE_EMB, embeddings)
with open(SAVE_LIST, "w", encoding="utf-8") as f:
    for img_id in id_list:
        f.write(f"{img_id}\n")

print(f"[INFO] Saved embeddings to {SAVE_EMB}")
print(f"[INFO] Saved image id list to {SAVE_LIST}")
