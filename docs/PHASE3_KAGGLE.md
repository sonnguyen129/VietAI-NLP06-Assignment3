# 🏆 Phase 3: The Global Kaggle Leaderboard

**FROM:** Lead AI Engineering Team  
**TO:** EvoAgent Development Teams  
**SUBJECT:** Welcome to the Big Leagues. 

If you are reading this, you have successfully rebuilt the EvoAgent loop, generated your proof files, and passed the Stage 4 grader. Take a moment to celebrate—you have secured your 6 baseline points. 

But Milestone 1 only proved that your system *works*. Milestone 2 is about proving your system can *win*. 

For the final 4 points of this project, your model will face the hidden Kaggle Test Set. The training wheels are off, the leaderboard is public, and every fraction of a percent matters. 

Here is everything you need to know to compete.

---

## ⏱️ Competition Details

- **Kaggle Link:** <https://www.kaggle.com/t/8f668980b48a473e842fc501c199d5c1>
- **Deadline:** July 9, 2026 at 23:59 (Asia/Saigon)
- **Scoring:** Final grading is based on the *Private Leaderboard* calculated after the deadline.

---

## 🚀 The Rule: Accuracy is the Product

In Phase 2, you followed our step-by-step `TODO` blocks. In Phase 3, those boundaries are gone. The starter EvoAgent is just your baseline—you are heavily encouraged to outgrow it. 

You may improve accuracy using any method you can engineer, including:
* Fine-tuning a model or adapter.
* Rewriting the prompt templates, DSL constraints, or parent-selection logic.
* Adding validation scripts to detect and fix malformed programs before submission.
* Using ensembling, self-consistency, or multiple candidate programs.
* Adding deterministic post-processing for common math or unit errors.

**Be creative. If it improves test accuracy, we want to see it.**

---

## Competition Rules

Phase 3 is intentionally open-ended, but every submission must follow these rules.

1. **Goal:** Maximize accuracy on the hidden Kaggle test set. The official score is the Kaggle Private Leaderboard score after the deadline.
2. **Allowed methods:** You may improve prompts, few-shot examples, post-processing, validation, ensembling, self-consistency, fine-tuning, retrieval, or other reproducible system changes. External APIs are allowed if your final pipeline can be explained and reproduced.
3. **Model limit:** If you run open-weight models on Modal, the largest model you may use is **9B parameters**. Larger hosted API models are allowed only as part of a documented, reproducible pipeline.
4. **Token and compute monitoring:** There is no fixed full-run token cap. `TokenBudget` is an observability tool that records Qwen input/output tokens and meta-agent tokens; it does not stop a run at a threshold. Use appropriate decode limits, prevent runaway prompts or repeated generations, and disclose major model/API usage, compute resources, and approximate cost in the final report.
5. **Synthetic and external data:** Synthetic data and general-purpose external datasets are allowed. You may generate augmented examples, paraphrases, translations, reasoning traces, or new training samples from the provided training data or general domain knowledge. Disclose every external and synthetic source in the final report.
6. **Test-data boundary:** You may not reconstruct, retrieve, or infer labels for specific test samples from external sources. This includes matching test items through videos, audio, transcripts, subtitles, source documents, URLs, filenames, IDs, timestamps, or metadata. Public availability does not make a test label permissible.
7. **No disguised test labeling:** Pseudo-labeling may use predictions from your own model pipeline, but it may not use test-specific external lookup, manually recovered answers, leaked labels, or information shared by another team.
8. **No leaderboard probing abuse:** Do not repeatedly submit tiny manual edits just to infer private labels. Kaggle submissions should come from meaningful pipeline changes.
9. **Team independence:** Teams may discuss general ideas, but may not exchange final predictions, hidden-label inferences, tuned submissions, or private evaluation results.
10. **Submission validity:** Your final `submission.csv` must match the Kaggle sample format exactly, include every required test row once, and contain no blank predictions.
11. **Reproducibility:** Keep the code, commit hash, commands, strategy file, model/API settings, and any generated artifacts needed to recreate your final submission.
12. **Auditability:** If asked, you must be able to rerun or explain the full path from data to `submission.csv`. Submissions that cannot be explained may be disqualified from the Phase 3 bonus.
13. **Enforcement:** Any hidden-label leakage, answer sharing, manual test labeling, fabricated run record, or unverifiable submission can receive 0 Phase 3 points, regardless of leaderboard rank.

The spirit of the competition is simple: build the strongest system you can, but win with engineering, not leakage.

---

## Team Formation

- You may compete individually or in a Kaggle team of up to **2 students**.
- Students may choose their own collaborator, and each student may belong to only one team.
- The team chooses one member as the designated ThinkFlic submitter.
- Submit only one final ZIP per team. It must identify every member by full name, student ID, class, Kaggle username, and contribution.
- Both members normally receive the same Phase 3 score. Individual contribution or integrity concerns may be reviewed separately.

---

## Submission Workflow

When you have found your champion system, turn its predictions into a Kaggle-ready `submission.csv`.

There are two supported paths:

1. Use `submit.py` when your final system is still based on an EvoAgent strategy JSON.
2. Use `format_submission.py` when your Phase 3 method produces predictions some other way, such as a notebook, API pipeline, fine-tuned model, ensemble, or custom post-processing script.

For the exact CSV schema and accepted input formats, see `docs/SUBMISSION_FORMAT.md`.

### Path A: EvoAgent Strategy Submission

**If running on Modal (Recommended):**
```bash
modal run --detach run_modal.py::submit --strategy-path /runs/exp_self/iter_best_strategy.json --output-file /runs/submission.csv
```

Local path, only if you have enough GPU memory:

```bash
python submit.py --strategy-path ./runs/exp_self/iter_best_strategy.json --output-file ./submission.csv
```

### Path B: Custom Phase 3 Pipeline Submission

If your method already produced predictions in a CSV or JSON file, format them with:

```bash
python format_submission.py --predictions my_predictions.csv --output-file submission.csv
```

Your prediction file must contain an `id` field and one prediction field named `predicted_value`, `prediction`, `answer`, or `value`. The formatter reads `data/test.json` to preserve the official Kaggle row order, writes every required row exactly once, and replaces missing, blank, null, NaN, or non-numeric predictions with `0.0`.

Before upload:

- [ ] `submission.csv` exists and the row count matches the Kaggle test set.
- [ ] No prediction cells are empty.
- [ ] If using a custom pipeline, you ran `format_submission.py` or otherwise matched `docs/SUBMISSION_FORMAT.md`.
- [ ] You have recorded the exact commit hash and strategy file used to generate this submission.

---

## After Kaggle: ThinkFlic Submission

After the Kaggle deadline, prepare the separate assessment ZIP, technical report, evidence files, integrity declaration, and video link by following [`docs/THINKFLIC_SUBMISSION.md`](THINKFLIC_SUBMISSION.md).

## How You Are Graded (The Final 4 Points)

Your Phase 3 score is entirely dependent on your final rank on the Kaggle Leaderboard compared to your peers.

- **If you do not submit:** 0 Points

- **Last Place Valid Submission:** 1.0 Point

- **First Place:** 4.0 Points

- **Everyone Else:** Scaled proportionally between 1 and 4 points based on your rank.

*(Example: In a class of 10 teams, Rank 2 gets 3.67 points, Rank 5 gets 2.67 points, etc.)*

**Pro Tip for the Leaderboard:** Do not submit randomly. Treat this like an engineering lab. Pick one hypothesis (e.g., "the AI struggles with percentage scaling"), change one thing, evaluate it on the dev set, and only submit to Kaggle when you see real improvement.

Good luck. We look forward to seeing your AI on the Leaderboard.
