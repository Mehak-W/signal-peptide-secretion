# Saved models

`vector_nn_seed_*.keras`, `scaler.joblib`, `config.json` are the 5-seed vector-NN
ensemble on **Ginkgo-AA-0** embeddings — the best *single-embedding* model
(~0.96), but AA-0 is API-only, proprietary, and **discontinued**, so this model
cannot embed new sequences and is **not reproducible**. It is kept only as the
historical best-case comparison.

`config.json`'s `ensemble_mse` (0.975) is one saved run; the honest value is a
distribution (see [`../docs/reproducibility_findings.md`](../docs/reproducibility_findings.md) §1).

**The reproducible headline model is the open-weight ensemble** (ProtT5 + ESM2-650M
+ ProtBERT) — regenerate it with `scripts/21_open_embedding_ensemble.py`
(open weights from HuggingFace, no API key).
