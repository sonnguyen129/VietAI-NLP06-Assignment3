# Run Instructions — Exact Reproduction

System: EvoAgent (Phase 2 self-improvement loop) + Phase 3 pipeline (2-strategy ensemble, 16-candidate self-consistency voting). Orchestration runs locally; inference runs on a Cerebrium GPU app serving vLLM.

## 1. Environment

- Python 3.13 (Anaconda/venv). Install dependencies:

```powershell
pip install -r requirements.txt
```

- Key packages: `openai`, `transformers` (tokenizer-only), `datasets`, `scikit-learn`, `pydantic`, `python-dotenv`, `numpy`, `tqdm`.

## 2. Model Serving (Cerebrium)

One Cerebrium app (GPU A10 24GB) serving `QuantTrio/Qwen3.5-4B-AWQ` with vLLM `0.24.0`, exposing an OpenAI-compatible API (`/v1/completions`, supports `n > 1`). Config in `cerebrium.toml`:

- vLLM installed via pip, `VLLM_USE_FLASHINFER_SAMPLER=0`
- `--max-model-len 16384 --gpu-memory-utilization 0.90`
- scale-to-zero when idle

Deploy with the Cerebrium CLI (`cerebrium deploy`), then create a `.env` file at the project root:

```env
CEREBRIUM_BASE_URL=<your Cerebrium endpoint>
CEREBRIUM_API_KEY=<your Cerebrium API key>
CEREBRIUM_MODEL=QuantTrio/Qwen3.5-4B-AWQ
HF_TOKEN=<HuggingFace token>
```

Data files (`data/train.json`, `data/dev.json`, `data/test.json`) are the ones provided with the assignment and must sit under `data/`.

## 3. Phase 2 — EvoAgent evolution loop (evidence files)

The self-improvement loop (propose → evaluate → reflect) is orchestrated by `run_cerebrium.py` / `src/harness.py`. It produces `runs/exp_self/` with `evolution_proof.json`, `learning_curve.pdf`, `strategy_diversity.pdf`, `failure_mode_report.pdf`, and the best strategy `iter_004`. Validate the proof with:

```powershell
python graders/grade_stage4_harness.py
```

(`evolution_proof.json` must be beside the runner, unchanged — best dev accuracy 0.5792, baseline 0.42.)

## 4. Phase 3 — Final Kaggle pipeline (exact commands)

All commands are idempotent: per-sample JSONL caches auto-resume, every request has exponential-backoff retry, logs go to `runs/phase3/run.log`, parameters are snapshotted to `phase3_config.json`.

### 4.1 Dev (validation, 584 samples)

```powershell
python predict.py generate --split validation --strategies s4fix --groups greedy,sampled --k 7 --cache runs\phase3\dev_cache.jsonl --concurrency 24
python predict.py repair   --split validation --cache runs\phase3\dev_cache.jsonl
python predict.py generate --split validation --strategies retr --groups greedy,sampled --k 7 --cache runs\phase3\dev_cache.jsonl --concurrency 24
python predict.py repair   --split validation --cache runs\phase3\dev_cache.jsonl
python predict.py score --ablation A1
python predict.py score --ablation A3
python predict.py score --ablation A4
```

Expected: ablation A4 (ensemble s4fix + retr, 16 candidates, self-consistency vote) reaches **dev accuracy 0.7346**.

### 4.2 Test + submission (494 samples)

```powershell
python predict.py generate --split test --strategies s4fix,retr --groups greedy,sampled --k 7 --cache runs\phase3\test_cache.jsonl --concurrency 24
python predict.py repair   --split test --cache runs\phase3\test_cache.jsonl
python predict.py submit --ablation A4 --cache runs\phase3\test_cache.jsonl --out runs\phase3\test_predictions.json
python format_submission.py --predictions runs\phase3\test_predictions.json --output-file submission.csv
python predict.py check --submission submission.csv
```

The resulting `submission.csv` is the file submitted to Kaggle as **"submission A4"** (Public LB 0.77935), included here as `kaggle/final_submission.csv`.

## 5. Model & Sampling Settings

| Parameter | Greedy | Sampled (n=7) | Repair |
|---|---|---|---|
| temperature | 0.0 | 0.6 | 0.0 |
| top_p / top_k | 1.0 / — | 0.95 / 20 | 1.0 / — |
| repetition_penalty | 1.05 | 1.05 | 1.0 |
| max_new_tokens | 3072 | 2048 | 256 (guided JSON) |

Voting: value-cluster tolerance `max(1e-4, 0.5%)`; acceptance thresholds — confidence 0.5, members 3, margin 0.15.

## 6. Compute & Cost

- ~14–15 GPU-hours A10 on Cerebrium (scale-to-zero when idle), estimated ~$15–20.
- Tokens: generation ~55.9M in / 17.2M out; repair ~8.8M in / 0.17M out; judge ablation (rejected) ~0.12M in / 0.04M out. Total ~64.9M input / ~17.5M output.
- No external paid APIs.
