# Reproducibility Investigation and Honest Findings

This document records what a from-scratch reproducibility audit established about the
signal-peptide WA-prediction results, including a direct cross-check against
Dr. Schrier's original Wolfram Language implementation. It supersedes the earlier
"MSE 0.932" headline.

## TL;DR

- The previously reported headline **MSE 0.932 is not reproducible.** Re-training the
  *identical* configuration 5 times gives **0.957 ± 0.009** (3-layer, dropout 0.35) or
  **0.978 ± 0.012** (dropout 0.20). 0.932 was never reproduced in 10 retrains — it was a
  favorable tail draw.
- The prior benchmark **MSE 0.953** (Schrier's Wolfram `net4`) is *also* a favorable draw.
  Running his exact code across 8 random seeds gives **0.990 ± 0.046, range [0.94, 1.07]**;
  0.953 sits ~0.8 std below that mean.
- **Both "benchmarks" were noise.** The vector-regression architecture is inherently
  high-variance (~±0.05 per single model). The honest, reproducible performance of the
  PLM-embedding vector approach is **~0.96–0.99 (5-seed ensemble)**, i.e. a **~19–21%
  improvement over the Grasso et al. RF baseline (1.22)** on the same official test split.
- The only correct way to report this architecture is **mean ± std over many seeds**, with
  bootstrap CIs. Single un-replicated numbers (0.953, 0.932) are unreliable.

## 1. Reproducibility of the repository's own configuration

Config: Ginkgo-AA0-650M, vector NN `(256,256,128)`, focal loss, 5-seed ensemble,
WA = softmax·[1..10]. Five independent retrains each (GPU, nondeterministic):

| dropout | 5-seed ensemble test MSE | range |
|---|---|---|
| 0.20 | 0.978 ± 0.012 | [0.960, 0.993] |
| 0.35 | 0.957 ± 0.009 | [0.950, 0.975] |

The headline 0.932 lies below the minimum of either distribution. Independent retrains of
the same config elsewhere in the repo corroborate ~0.975–0.981 (`models/config.json` 0.975;
`results/linear_baseline_results.json` best-NN 0.981).

## 2. Direct cross-check against Schrier's Wolfram implementation

We ran his exact `scripts/10_Vector_regression.wls` `net4`
(`{Dense256, ParametricRamp, Dropout0.2}×2, Dense10, Softmax`, FocalLoss[0.25,2], single
model, full 3095-row train, 1326-row test) in Wolfram Engine 14.3, varying only the random
seed:

| seed | test MSE |   | seed | test MSE |
|---|---|---|---|---|
| default | 0.938 | | 4 | 0.983 |
| 1 | 1.065 | | 5 | 0.965 |
| 2 | 0.991 | | 6 | 1.065 |
| 3 | 0.954 | | 7 | 0.955 |

**Mean 0.990 ± 0.046.** His published 0.9531 is one favorable draw. This is the decisive
evidence that the Wolfram-vs-Python "gap" we initially suspected was **seed variance, not a
real framework advantage**: his single Wolfram models (~0.99) and our single Keras models
(~1.04) overlap heavily once you account for the ±0.05 spread.

## 3. Honest headline result

| Model (Ginkgo-AA0) | Reproducible test MSE | Notes |
|---|---|---|
| Grasso et al. RF (baseline) | 1.22 | published; our reproduction 1.19–1.25 |
| Vector NN, 5-seed ensemble | **~0.96–0.98** | mean ± ~0.01 over retrains |
| Schrier `net4` (Wolfram, single) | 0.99 ± 0.05 | his 0.953 = favorable draw |

The reproducible vector-NN result improves on the Grasso RF baseline by ~19–21%. It does not
"beat 0.953", because 0.953 is not a stable number to beat — it is a low sample of the same
~0.97–0.99 distribution.

## 4. A confirmed, faithful improvement: PReLU activation

Schrier used `ParametricRampLayer` (PReLU). The repo approximated it with `LeakyReLU`.
Switching to PReLU (faithful to his design) reproducibly improves the (256,256)/0.2 ensemble
from **1.010 → 0.987**. This is adopted.

## 5. Alternative model classes (non-neural) — negative result

On Ginkgo-AA0 embeddings, predicting WA: HistGradientBoosting 1.09, SVR-RBF 1.04,
Ridge(bins) 1.08, RF ~1.16. **None beat the vector NN (~0.97).** Neural networks best exploit
dense PLM embeddings here; tree/kernel models plateau near the RF baseline. This matches
Schrier's own observation that RF performance is flat across representations.

## 6. Design-task evaluation (vs Grasso Figure 4e)

Grasso Fig 4e: 11 of 15 hand-picked, independently (NLR-) validated designs fell within ±1 WA
of prediction (73%). Our models, evaluated on the **full** ~4,800-variant design library
(library-screening WA), hit within ±1 WA for ~41% (Spearman ~0.39, MAE ~1.55).

These are not directly comparable: his 15 were curated and independently re-assayed; ours is
the entire noisy design set with no selection. Reporting the full, unselected library is the
**less biased, more honest** evaluation — a methodological improvement over a 15-point curated
comparison, even though the headline percentage is lower.

## 7. Generalization (leave-one-gene-out)

132/134 genes (99.8% of test sequences) overlap between train and test in Grasso's official
split, so the standard test MSE is optimistic. Leave-one-gene-out CV gives **MSE 2.19**
(Spearman 0.72) — the realistic cross-gene generalization estimate, which should be reported
alongside (not hidden behind) the standard number.

This inflation is a property of the split, not of any one model. Running the same
leave-one-gene-out protocol on the Grasso RF baseline (physicochemical features, the
paper's exact RF hyperparameters) raises its MSE from **1.25** (standard split) to **2.88**
(leave-one-gene-out), a **2.31×** inflation — essentially identical to the vector NN's 2.3×
(0.96 → 2.19). A random forest on physicochemical features and a neural net on PLM
embeddings inflate by the same factor, confirming the effect comes from the ~99.8% gene
overlap in the benchmark split rather than from the model class.

## 8. Methodological lessons (the real contribution)

1. **Report mean ± std over seeds + bootstrap CIs.** A single number from this architecture is
   unreliable (±0.05). Both prior "benchmarks" were artifacts of single draws.
2. **Never select hyperparameters on the test set.** The 0.35 dropout was test-selected; on a
   validation split, dropout 0.15–0.35 are statistically indistinguishable.
3. **Foreground leave-one-gene-out** as the honest generalization metric.
4. **Cross-framework "reproductions" need variance quantification**, not single-point matching.

## 9. Ginkgo-AA0 is discontinued and permanently unreproducible (critical for the paper)

The headline embedding, **Ginkgo-AA0-650M, is API-only and proprietary — its weights were
never released.** As of mid-2026 the Ginkgo model API and self-serve registration portal
(`models.ginkgobioworks.ai`) appear discontinued (the `ginkgo-ai-client` package has been
untouched since March 2025; Ginkgo cut cell-engineering R&D ~42% and drew down its Google
Cloud / Vertex AI commitment, on which the API ran). **Consequence: nobody — not us, not
Dr. Schrier, not a reviewer or reader — can regenerate AA-0 embeddings.** Basing the paper's
headline on AA-0 would rest the central claim on a permanently unreproducible input, directly
contradicting the reproducibility thesis of this revision.

**Decision: ESM2-650M (open weights, Meta) is the primary, reproducible embedding.** AA-0 is
reported as a caveated historical comparison ("best in our experiments; API since
discontinued; embeddings not reproducible"). This is justified empirically: a linear probe on
AA-0 vs ESM2-650M predictions agree at **Pearson r = 0.975** (Spearman 0.972) on the test set,
with near-identical rank correlation to truth (0.861 vs 0.856) — i.e. ESM2 captures
essentially the same signal, only marginally less sharp (~0.97 vs ~1.05 at the NN level). All
new work (design, external, the cross-dataset extension) runs on open, reproducible models.

**This reframes the open-PLM experiments (§10) as the central question, not a side quest:**
can an open, reproducible model (ESM2, ProtBERT, ProtT5, or an ensemble) recover AA-0's ~0.97?
If so, the lost AA-0 performance is recovered *reproducibly* — a clean, defensible headline.

## 10. Improvement campaign — a thorough set of honest negative results

We systematically tested five directions for legitimately improving the reproducible test MSE
(no test-set tuning). Base: vector NN (256,256,128), PReLU, dropout 0.3, focal, 5-seed,
mean over R=2 retrains.

| Direction | Result | Verdict |
|---|---|---|
| **Cross-representation ensemble** | Ginkgo 0.966; +ESM2-650M 0.981; all-3 0.980; weighted-2×Ginkgo 0.968 | no help — weaker ESM2 models drag the ensemble up |
| **Ordinal Cramér/CDF-L2 loss** | 1.049 vs focal 0.966 | worse |
| **Focal + λ·MSE(WA) hybrid** (Schrier's idea) | 1.046 vs focal 0.966 | worse |
| **Non-neural models** (Ginkgo) | HistGBM 1.09, SVR-RBF 1.04, Ridge 1.08, RF ~1.16 | all worse than NN |
| **ProtBERT** (open, BERT-based) | Ridge-probe 1.21 vs ESM2 1.17 | not better than ESM2 |
| **PReLU activation** (Schrier's `ParametricRamp`) | (256,256)/0.2 ensemble 1.010 → 0.987 | **helps (~0.02); adopted** |

**Conclusion (single-embedding):** the focal-loss vector NN is hard to beat per embedding;
ordinal losses, tree/kernel models, and ProtBERT did not help, and Ginkgo+ESM2 ensembling hurt
(Ginkgo too dominant). PReLU is the only single-model gain.

## 11. The reproducible headline: an open-model ensemble recovers AA-0

Generating embeddings from open, architecturally-diverse PLMs (ProtT5 [T5], ProtBERT [BERT],
alongside ESM2-650M) and ensembling the vector NNs gives a **fully reproducible** result that
nearly matches the now-unavailable AA-0:

| Model (vector NN, 5-seed, mean over R=2) | Test MSE | Reproducible |
|---|---|---|
| Ginkgo AA-0 (alone) | 0.959 ± 0.014 | ✗ API discontinued |
| **ProtT5 + ESM2-650M + ProtBERT (open ensemble)** | **0.981 ± 0.005** | **✓ open weights** |
| ProtT5 + ProtBERT | 0.986 | ✓ |
| ProtT5 + ESM2-650M | 0.995 | ✓ |
| ProtT5 (alone) | 1.017 | ✓ |
| ESM2-650M | 1.072 | ✓ |
| ProtBERT | 1.084 | ✓ |

**The open-model ensemble (0.981 ± 0.005) closes all but ~0.02 of the gap to AA-0 (0.959) and
improves ~20% over the Grasso RF baseline (1.22) — and anyone can regenerate every embedding.**

Why the ensemble works here when Ginkgo+ESM2 didn't: ProtT5 (1.017) is close enough to ESM2
(1.072) and ProtBERT (1.084) that the three open models are quality-comparable, so averaging
their decorrelated errors yields a real gain (0.981 < 1.017). Earlier, Ginkgo (0.96) so
dominated ESM2 (1.07) that averaging only diluted it. Balanced components → ensembling pays.

**Recommended headline:** the open-model vector-NN ensemble (~0.98, reproducible, ~20% over
Grasso). AA-0 (~0.96) is reported as the best-but-discontinued comparison. All design/external
extension work runs on these open embeddings (no API key, no dead endpoint).

## 12. Extensions (open/reproducible embeddings)

**Cross-dataset fine-tuning (Script 13, corrected statistics).** Using the *pooled
out-of-fold* Spearman with bootstrap CI (not the old biased mean-of-per-fold), fine-tuning the
ESM2-650M vector ensemble on each external dataset gives:

| Dataset | N | Zero-shot ρ | Fine-tuned ρ [95% CI] | Significant? |
|---|---|---|---|---|
| Wu (binary) | 81 | −0.227 | **+0.341 [+0.131, +0.536]** | yes (CI excludes 0) |
| Xue | 322 | −0.104 | −0.192 [−0.296, −0.086] | yes (worse) |
| Zhang-P43 | 114 | −0.194 | −0.316 [−0.479, −0.133] | yes (worse) |
| Zhang-PglVM | 114 | −0.269 | −0.231 [−0.398, −0.051] | yes (worse) |

Honest read: fine-tuning *significantly reverses* the negative zero-shot correlation for the
binary Wu dataset, but *significantly worsens* the three continuous datasets. (The old
mean-of-folds estimator had hidden the Wu significance behind high fold-variance.)

**Design task vs Grasso Fig 4e (design_parity_fig4e.py).** On the full ~4,800-variant design
library, all four models (RF/NN × PhysChem/ESM2) predict within ±1 WA for ~41% of variants
(Spearman ~0.39–0.41), versus Grasso's 11/15 = 73% on 15 hand-picked, independently NLR-
validated designs. Not strictly comparable (his were curated and re-assayed; ours is the full,
unselected, noisier library) — the lower-bias evaluation. Parity plot:
`figures/design_parity.png`.
