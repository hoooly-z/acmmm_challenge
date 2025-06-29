import os
import json
import torch
import pandas as pd
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 参数 ===
QUERY_CSV = "data/track2/private_set/query.csv"
OUT_COMPRESSED = "data/result/query_with_entities.csv"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 77
DEVICE_LLM = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_N = None  

print("[INFO] Loading LLM for entity extraction...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, cache_dir="./qwen")
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, cache_dir="./qwen").to(DEVICE_LLM).eval()

def extract_visual_entities(query_text, max_tokens=77, retry=3):
    import re
    prompt = (
        "Extract all visible objects, people, clothing, actions, colors, and scene elements from the following sentence. "
        "Output ONLY a single English comma-separated list (no numbering, no extra text, no explanation, no commentary, no duplicates). Each entity should be a short English word or phrase. Example output: 'man, trophy, white shirt, cap, celebration, blue background'\n"
        f"Sentence: {query_text}\n"
        "Entities:"
    )
    for _ in range(retry):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE_LLM)
        outputs = llm.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        comp = text.split("Entities:")[-1].strip().replace("\n", " ")
        comp = comp.split("Explanation:")[0].split("Entities:")[0].strip()

        if not comp or len(comp) < 2:
            continue
   
        entities = [e.strip(" ,.;") for e in comp.split(",")]

        entities = [e for e in entities if re.search(r"[a-zA-Z]", e) and 1 < len(e) < 50]
   
        entities = list(dict.fromkeys(entities))

        if entities:
            return ", ".join(entities)
        else:
         
            return comp
    return ""


print("[INFO] Extracting visual entities ...")
df_query = pd.read_csv(QUERY_CSV)
if DEBUG_N is not None:
    df_query = df_query.head(DEBUG_N)

entities_list = []
for text in tqdm(df_query["query_text"], desc="Processing queries"):
    if not isinstance(text, str) or not text.strip():
        entities_list.append("")
    else:
        entities = extract_visual_entities(text, max_tokens=MAX_TOKENS)
        entities_list.append(entities)

df_query["compressed_entities"] = entities_list
df_query.to_csv(OUT_COMPRESSED, index=False)
print(f"[INFO] Saved: {OUT_COMPRESSED}")


del llm, tokenizer, entities_list, text
torch.cuda.empty_cache()
gc.collect()
