#!/usr/bin/env python3
"""Generate mean-pooled embeddings from open protein language models for the
signal-peptide sequences, writing parquets that match the existing schema
(sequence, embedding, WA, BIN01..BIN10) so they drop into the pipeline.

Open models => no Ginkgo API key needed, so these also work for the design and
external datasets (sidestepping the key bottleneck).

Usage:  python3 gen_open_embeddings.py <model> <split>
  model: prott5 | ankh | esm2-150m
  split: train | test | both
"""
import os, sys, re, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd, torch
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
BIN_COLS = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {
    'protbert': ('Rostlab/prot_bert',                    'bert', 'protbert'),    # BERT, space-sep, ~420M
    'prott5':   ('Rostlab/prot_t5_xl_half_uniref50-enc', 't5',  'prott5'),       # T5 enc, space-sep, 3B
    'esm2-150m':('facebook/esm2_t30_150M_UR50D',         'esm',  'esm2-150M'),
}

def log(m): print(m, flush=True)

def load_t5(name):
    from transformers import T5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(name).eval().to(DEVICE)
    return tok, model

def load_esm(name):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(DEVICE)
    return tok, model

def load_bert(name):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name, do_lower_case=False)
    model = AutoModel.from_pretrained(name).eval().to(DEVICE)
    return tok, model

@torch.no_grad()
def embed_bert(seqs, tok, model, batch=16):
    out = []
    for i in range(0, len(seqs), batch):
        chunk = [' '.join(re.sub(r'[UZOB]', 'X', s)) for s in seqs[i:i+batch]]
        enc = tok(chunk, add_special_tokens=True, padding=True, return_tensors='pt').to(DEVICE)
        rep = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy())
        if i % 256 == 0: log(f"    {i}/{len(seqs)}")
    return np.concatenate(out, 0)

@torch.no_grad()
def embed_t5(seqs, tok, model, batch=8):
    out = []
    for i in range(0, len(seqs), batch):
        chunk = [' '.join(re.sub(r'[UZOB]', 'X', s)) for s in seqs[i:i+batch]]
        enc = tok(chunk, add_special_tokens=True, padding=True, return_tensors='pt').to(DEVICE)
        rep = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy())
        if i % 256 == 0: log(f"    {i}/{len(seqs)}")
    return np.concatenate(out, 0)

@torch.no_grad()
def embed_esm(seqs, tok, model, batch=16):
    out = []
    for i in range(0, len(seqs), batch):
        enc = tok(list(seqs[i:i+batch]), padding=True, return_tensors='pt').to(DEVICE)
        rep = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy())
        if i % 256 == 0: log(f"    {i}/{len(seqs)}")
    return np.concatenate(out, 0)

def run(model_key, split):
    hf, kind, outname = MODELS[model_key]
    loaders = {'t5': load_t5, 'esm': load_esm, 'bert': load_bert}
    embedders = {'t5': embed_t5, 'esm': embed_esm, 'bert': embed_bert}
    log(f"loading {hf} ...")
    tok, model = loaders[kind](hf)
    embed_fn = embedders[kind]
    splits = ['train', 'test'] if split == 'both' else [split]
    for sp in splits:
        src = pd.read_parquet(DATA / f'{sp}AA_esm2-650M.parquet')  # reuse rows/labels
        seqs = src['sequence'].tolist()
        log(f"{sp}: embedding {len(seqs)} sequences with {outname}")
        t0 = time.time()
        emb = embed_fn(seqs, tok, model)
        log(f"  done in {(time.time()-t0)/60:.1f} min, dim {emb.shape[1]}")
        out = pd.DataFrame({'sequence': seqs, 'embedding': list(emb.astype(np.float32)),
                            'WA': src['WA'].values})
        for c in BIN_COLS:
            out[c] = src[c].values
        path = DATA / f'{sp}AA_{outname}.parquet'
        out.to_parquet(path)
        log(f"  saved {path}")

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'both')
    log("GEN_DONE")
