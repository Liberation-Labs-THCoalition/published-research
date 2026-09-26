#!/usr/bin/env python3
"""Embed every unique LongMemEval_S-cleaned USER turn, plus the 500 questions, on GPU1.

Model: BAAI/bge-base-en-v1.5, CLS pooling, L2-normalised, max_length 512. User turns only: that is
where the personal facts live, and the BM25 channel already reads every assistant turn in full. (A CPU
run over all 189,522 turns managed 10 turns/s on this loaded machine, >5 h: see emb/old_cpu_partial.)
Queries get bge's retrieval instruction; turns don't. Written for v10 retrieval (see DESIGN.md).

Run with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (GPU0 carries the memory backfill).
A thermal guard pauses while either card is at or above 85 C (Scrub alerts at 90).

Resumable: rows are written into a .npy memmap in a fixed order (by text length, then hash), and
emb/progress.json records the number of rows COMPLETED and flushed, never the number attempted.
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

# ~/.local's torchvision was built for another torch ("operator torchvision::nms does not exist"), and
# transformers imports it if present, which breaks the BERT import. Hide it from this process only.
sys.modules["torchvision"] = None
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

OUT = "/mnt/data1/lme_v2/v10/emb"
DATA = "/mnt/data1/datasets/longmemeval/longmemeval_s.json"
MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAXLEN, THREADS = 512, 4
ROLES = {"user"}
PAUSE_AT_C = 85

def gpu_temps():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]

def turn_key(role, content):
    return hashlib.sha1((role + "\x00" + content).encode()).hexdigest()

def main():
    torch.set_num_threads(THREADS)
    path = snapshot_download(MODEL)
    rev = os.path.basename(os.path.normpath(path))
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(path).eval().to(dev)
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    data = json.load(open(DATA))
    texts = {}
    for q in data:
        for s in q["haystack_sessions"]:
            for t in s:
                if t["role"] in ROLES:
                    texts[turn_key(t["role"], t["content"])] = t["content"]
    keys = sorted(texts, key=lambda k: (len(texts[k]), k))
    index_p = f"{OUT}/turn_index.json"
    if os.path.exists(index_p):
        assert json.load(open(index_p)) == keys, "turn order changed since the last run; refusing to resume"
    else:
        json.dump(keys, open(index_p, "w"))

    emb_p, prog_p = f"{OUT}/turns.f16.npy", f"{OUT}/progress.json"
    done = json.load(open(prog_p))["rows_done"] if os.path.exists(prog_p) else 0
    arr = (np.lib.format.open_memmap(emb_p, mode="r+") if done else
           np.lib.format.open_memmap(emb_p, mode="w+", dtype=np.float16, shape=(len(keys), 768)))
    print(f"{MODEL} @ {rev}: {len(keys)} unique turns, resuming at {done}", flush=True)

    def embed(batch):
        enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        with torch.inference_mode():
            cls = model(**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()

    t0, i, nb = time.time(), done, 0
    while i < len(keys):
        n_tok = len(tok(texts[keys[i]], truncation=True, max_length=MAXLEN)["input_ids"])
        bs = 128 if n_tok <= 64 else 48 if n_tok <= 192 else 16
        batch = [texts[k] for k in keys[i:i + bs]]
        arr[i:i + len(batch)] = embed(batch).astype(np.float16)
        i += len(batch)
        nb += 1
        if dev == "cuda" and nb % 10 == 0:
            while max(gpu_temps()) >= PAUSE_AT_C:
                print(f"  thermal pause: {gpu_temps()} C", flush=True)
                time.sleep(60)
        if nb % 25 == 0 or i == len(keys):
            arr.flush()
            json.dump({"rows_done": i, "rows_total": len(keys), "model": MODEL, "revision": rev}, open(prog_p, "w"))
            rate = (i - done) / (time.time() - t0)
            print(f"  {i}/{len(keys)}  {rate:.0f} turns/s  eta {(len(keys) - i) / max(rate, 1e-9) / 60:.0f} min  temps {gpu_temps()}", flush=True)
    arr.flush()
    json.dump({"rows_done": i, "rows_total": len(keys), "model": MODEL, "revision": rev}, open(prog_p, "w"))

    qs = embed([QUERY_PREFIX + q["question"] for q in data])
    np.save(f"{OUT}/questions.f16.npy", qs.astype(np.float16))
    json.dump({"model": MODEL, "revision": rev, "pooling": "cls+l2", "max_length": MAXLEN, "roles": sorted(ROLES),
               "query_prefix": QUERY_PREFIX, "turns": len(keys), "questions": len(data),
               "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest(),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(f"{OUT}/meta.json", "w"), indent=1)
    print("done", flush=True)

if __name__ == "__main__":
    main()
