# Visual Place Recognition with Adaptive Re-ranking

Visual place recognition (VPR) estimates where a photo was taken by retrieving the most similar geo-tagged images from a database. A standard pipeline has two stages:

1. **Global retrieval** ranks the database with a single descriptor per image. It takes 0.01–0.25 s per query.
2. **Geometric re-ranking** matches local features between the query and each of the top-20 candidates, runs RANSAC, and re-orders the candidates by inlier count. It is often more accurate, but it costs about 1–5 s per query.

This project **benchmarks both stages** and then adds **adaptive re-ranking**: a logistic-regression gate that pays for re-ranking **only on queries it predicts to be hard**.

> Team project (4 students), *Advanced Machine Learning*, Politecnico di Torino, 2025.
> Full report: [`s337035_s350812_s339063_s350958_project4.pdf`](s337035_s350812_s339063_s350958_project4.pdf)

## Key results: adaptive re-ranking

MixVPR retrieval + SuperPoint+LightGlue re-ranking, top-K = 20, a prediction counts as correct within 25 m (Recall@1, %):

| Test set | Retrieval only | Full re-ranking (always on) | **Adaptive re-ranking** | Share of full gain kept | Matcher calls saved |
|---|---|---|---|---|---|
| SVOX Night (823 queries) | 62.6 | 81.3 | **80.9** | 98% | 59% |
| Tokyo-XS (315 queries) | 76.5 | 89.2 | **88.3** | 93% | 71% |

- **Matcher calls saved** = 1 − (average number of matched candidates per query) / 20. An easy query costs 1 match and a hard query costs 20.
- **Where the numbers come from.** The retrieval-only and full re-ranking columns come from the benchmark runs (report, Table 1). The adaptive column comes from [`results/`](results). The adaptive runs recompute their own retrieval-only baseline, which differs slightly on SVOX Night (62.3 vs 62.6).
- **More configurations.** Results for CosPlace, LoFTR, SVOX Sun and SF-XS are in the report (Table 3) and in `results/`.

## Benchmark findings

The benchmark covers 4 retrieval models (NetVLAD, CosPlace, MixVPR, MegaLoc) × 3 local matchers (LoFTR, SuperPoint+LightGlue, SuperGlue) on SF-XS, Tokyo-XS and SVOX Sun/Night:

- **Re-ranking helps most when retrieval has headroom.** This happens when the correct place is in the top 20 but not ranked first. The largest gain is CosPlace on SVOX Night: 33.3 → 61.2 Recall@1 with LoFTR.
- **It adds little on top of strong retrieval, and can even hurt.** MegaLoc already reaches 96.5 Recall@1 on SVOX Night, and re-ranking lowers it (91.1 with SuperPoint+LightGlue) when matching fails.
- **Cost is dominated by matching.** SuperGlue took about 1–1.5 s per query, while LoFTR and SuperPoint+LightGlue were usually slower, at up to about 5 s. Timings are indicative only, because runs were spread across different machines and Colab GPUs.
- **Dot product and L2 distance give identical rankings,** as expected with L2-normalized descriptors.

## Adaptive re-ranking: method

```
query ──► global retrieval (MixVPR or CosPlace) ──► top-20 candidates
                                                     │
                 match the query with the top-1 candidate only (LoFTR or SuperPoint+LightGlue + RANSAC)
                                                     │
                   logistic regression on the top-1 inlier count ──► P(top-1 is correct)
                                                     │
                   P ≥ τ  ──►  EASY: keep the retrieval order (1 match)
                   P < τ  ──►  HARD: match all 20 candidates and re-rank by inlier count
```

- **Signal.** The only input is the number of RANSAC inliers between the query and its top-1 candidate. On the SF-XS validation set, this single number predicts whether the top-1 retrieval is correct with ROC-AUC 0.95.
- **Two domain-specific gates.**
  - LR-Sun is trained on the SVOX **train** queries under daylight (712 queries), and LR-Night on the SVOX train queries at night (702 queries).
  - Each gate pools 4 pipelines (MixVPR/CosPlace × LoFTR/SuperPoint+LightGlue).
- **Model selection on validation data only (SF-XS val, 7,993 queries).**
  - The regularization strength C maximizes validation ROC-AUC.
  - The threshold τ maximizes validation accuracy at predicting top-1 correctness, with ties broken by savings. This gives τ_Sun = 0.15 and τ_Night = 0.25.
  - The test sets (Tokyo-XS, SVOX Sun/Night test, SF-XS test) are used only for the final evaluation.
- **Equivalent inlier threshold.** Because the gate has one input, it amounts to a threshold on the inlier count. A query is treated as easy when its top-1 match has at least **14 inliers (LR-Sun)** or **16 inliers (LR-Night)**. The logistic regression provides the calibrated probability used to choose that cut-off.

## Repository structure

**Written by our team:**

| Path | Purpose |
|---|---|
| `build_csv.py` | Builds per-query tables (`inliers_top1`, `baseline_correct`, `reranked_correct`) from retrieval and matcher outputs |
| `tune_lr_and_threshold.py` | Trains the gate, selects C and τ on validation data, and writes the threshold sweep |
| `adaptive_match_and_eval.py` | Runs adaptive matching on a test set and reports retrieval-only and adaptive Recall@{1,5,10,20}, hard-query rate and savings |
| `Project_AML.ipynb` | Colab notebook with the runs from one of our environments: part of the benchmark, gate training and all adaptive evaluations. Other benchmark runs were executed by team members on their own machines |
| `csv/`, `models/`, `sweeps/`, `results/` | Gate training/validation tables, the trained gates (`lr_sun`, `lr_night`), C and τ sweeps, and per-run summaries with per-query decisions (`results/` also holds runs at τ = 0.45 and 0.85) |
| `VPR-methods-evaluation/vpr_models/mixvpr.py`, `netvlad.py` | Small changes: a ResNet-50 backbone for MixVPR, and a NetVLAD checkpoint download without `wget` |

**From the course starter code** ([FarInHeight/Visual-Place-Recognition-Project](https://github.com/FarInHeight/Visual-Place-Recognition-Project), MIT):
`VPR-methods-evaluation/`, `match_queries_preds.py`, `reranking.py`, `vpr_uncertainty/`, `util.py`, `download_datasets.py`, `start_your_project.ipynb`.
Local matchers come from [alexstoken/image-matching-models](https://github.com/alexstoken/image-matching-models).

## Reproducing

```bash
git clone https://github.com/dariush-vaghei/vpr-adaptive-reranking.git
cd vpr-adaptive-reranking

# local matchers (the version used by the course starter code)
git clone https://github.com/alexstoken/image-matching-models.git
cd image-matching-models && git checkout 0079bf7 && git submodule update --init --recursive && pip install -e .[all] && cd ..

pip install faiss-cpu pandas numpy tqdm scikit-learn==1.5.2 joblib opencv-python kornia einops
python download_datasets.py
```

1. **Retrieval.** This saves the top-20 predictions for each query:
   ```bash
   python VPR-methods-evaluation/main.py --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
     --image_size 512 512 --database_folder <db> --queries_folder <queries> \
     --num_preds_to_save 20 --recall_values 1 5 10 20 --log_dir logs/<run>
   ```
2. **Always-on re-ranking** (needed for the gate's training and validation tables):
   ```bash
   python match_queries_preds.py --preds-dir <run>/preds --matcher superpoint-lg --num-preds 20
   python reranking.py --preds-dir <run>/preds --inliers-dir <run>/preds_superpoint-lg --num-preds 20
   ```
3. **Gate tables** (one per pipeline, for SVOX train Sun/Night and SF-XS val):
   ```bash
   python build_csv.py --preds-dir <run>/preds --inliers-dir <run>/preds_superpoint-lg --out-csv csv/train/<name>.csv
   ```
4. **Train the gate and select C and τ:**
   ```bash
   python tune_lr_and_threshold.py --train-csvs csv/train/train_night_*.csv --val-csvs csv/validation/sf_val_*.csv \
     --out-model models/lr_night.joblib --out-json models/lr_night.json \
     --out-table sweeps/night_best_per_C.csv --out-curve sweeps/night_threshold_curve_bestC.csv
   ```
5. **Adaptive evaluation on a test set:**
   ```bash
   python adaptive_match_and_eval.py --preds-dir <test_run>/preds --out-dir <test_run>/adaptive_sp-lg_lr_night \
     --matcher superpoint-lg --lr-model models/lr_night.joblib --lr-json models/lr_night.json \
     --out-json results/svox_night/<name>_summary.json --log-jsonl results/svox_night/<name>_decisions.jsonl
   ```
   - `--threshold` overrides τ.
   - `--data-root` remaps image paths if the predictions were produced on another machine.
   - Matches are cached in `--out-dir` and reused on reruns.

## License

MIT. The starter code and third-party components keep their original licenses.
