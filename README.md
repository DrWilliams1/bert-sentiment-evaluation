# BERT Sentiment Classification
## Training trade-offs and high-confidence errors

**Joe Williams · Portfolio extension · Completed Colab pilot, November 15, 2026**

This project asks how training only BERT's final classifier compares with updating the entire model under the same fixed training settings. It combines a measured performance/time comparison with a review of incorrect predictions that receive high softmax scores.

The baseline is adapted from Chapter 11 of *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst. The portfolio extends that baseline with a measured training comparison, saved predictions, and error analysis. Sources and modifications are documented in [ATTRIBUTION.md](ATTRIBUTION.md).

### Results

Both regimes were evaluated on the same 1,066 validation examples.

| Measure | Classifier only | Full fine-tuning |
| --- | ---: | ---: |
| Accuracy | 60.69% | 85.46% |
| Macro F1 | 0.6058 | 0.8544 |
| Positive-class F1 | 0.6269 | 0.8592 |
| Training-call wall time | 36.24 s | 85.02 s |
| Trainable parameters | 1,538 | 108,311,810 |
| Correct predictions | 647 / 1,066 | 911 / 1,066 |
| Predictions with softmax score ≥ 0.90 | 0 | 718 |
| Errors in that group | 0 / 0; rate undefined | 45 / 718; 6.27% |

Full fine-tuning improved validation accuracy by **24.77 percentage points** and took **2.35 times as long**, an additional 48.78 seconds for the measured training call. The timing excludes checkpoint loading, validation, and saving. It is a single observed timing on a Tesla T4, not a hardware benchmark or a monetary cost estimate.

The selection rule chose the highest validation macro F1, with shorter training time as the tie-breaker. It selected full fine-tuning. That model achieved **85.08% accuracy (907 / 1,066)** and **0.8508 macro F1** on the official test split. Its positive-class F1 was 0.8526. Among 708 test predictions scoring at least 0.90, 48 were wrong (6.78%).

Test confusion counts, with dataset labels as rows and predictions as columns:

| Actual / predicted | Negative | Positive |
| --- | ---: | ---: |
| Negative | 447 | 86 |
| Positive | 73 | 460 |

### What the comparison shows

Under this one-epoch configuration, updating BERT's representations produced substantially better validation performance than learning only the classifier. The classifier-only run had no predictions above the 0.90 threshold; its undefined conditional error rate must not be interpreted as superior reliability.

A high softmax score did not guarantee a correct label. Validation row 748 was labeled negative but predicted positive with a score of 0.986983. The review acknowledges careful production while criticizing the film as overly dutiful and mechanical. A plausible interpretation is that mixed evaluative wording and indirect criticism were difficult for the model. This example does not identify the internal cause of the prediction or establish the frequency of that error type.

The qualitative observation is recorded in [analysis/error_analysis.csv](analysis/error_analysis.csv). The original error worksheet remains unchanged; the remaining mistakes have not been manually categorized.

### Method

- Model: `google-bert/bert-base-cased`; a pretrained BERT checkpoint with a newly initialized binary classifier.
- Dataset: `cornell-movie-review-data/rotten_tomatoes`; 8,530 training, 1,066 validation, and 1,066 test examples. Labels: 0 = negative, 1 = positive.
- Fixed settings: seed 42, one epoch, batch size 16, learning rate 0.00002, maximum input length 256, weight decay 0.01.
- Classifier-only training freezes the complete BERT backbone. Full fine-tuning updates every parameter. Both runs recorded the same initial classifier-weight hash; that check covers the weight tensor, not every model tensor.
- Validation drives comparison and model selection. Only the selected model receives final test evaluation in this run's workflow.
- The data audit found zero overlapping lowercased, whitespace-normalized exact texts between splits. It does not rule out semantic duplicates or broader data contamination.

### Evaluation limits

The official test split is held out from training and model selection within this workflow. Results are reported as a standard benchmark evaluation.

This is one seed and one epoch, with no separate hyperparameter tuning for either regime. A learning rate suitable for full fine-tuning may not be suitable for a randomly initialized classifier-only head. Results therefore compare these fixed configurations, not each approach's best achievable performance. Backbone dropout remains active during the classifier-only training pass; this is not a cached deterministic-feature experiment.

Only one selected validation mistake received qualitative review. Softmax scores were not calibrated, no calibration analysis was performed, and no independent annotation reliability study was conducted. These movie-review results do not establish suitability for other domains.

### Inspect or reproduce

1. Open [BERT_Sentiment_Evaluation.ipynb](BERT_Sentiment_Evaluation.ipynb) to read the saved run and added explanations. No training is needed to view the results.
2. To verify the included numerical evidence locally, install NumPy and run `python scripts/verify_results.py` from the repository root. This does not download data or train a model. It checks validation predictions, metrics, the complete error worksheet, selection, and final-summary arithmetic.
3. For a new experiment, make a separate notebook copy and use a GPU runtime. Before running all cells, set the training and final-test switches to `False` in their execution cells. Set `SHOW_REVIEW_TEXT = False` before sharing outputs.
4. To reuse the recorded snapshots, replace the two Hub lookup lines in the data-loading cell with the model and dataset revision strings in `results/20261115T224719Z-739463/data_record.json`. The original source resolves the latest revisions at the start of each run.
5. Run setup, deliberately enable training, inspect validation results, then enable the selected-model test. The notebook contains the saved run; a documented post-run cleanup removed an unsupported metadata field without changing training, selection, or metric calculations.

The installation ranges in `requirements.txt` are not a lockfile. Actual recorded versions and hardware are in `environment.json`; exact numerical reproduction is not guaranteed. Local model checkpoints and raw test logits are not included. Without the checkpoints, inference requires a new training run. A future run should save final test logits and labels to permit a prediction-level test audit.

### Included files and verification

The `results/20261115T224719Z-739463/` directory includes all 11 supplied result files. Ten files are preserved byte-for-byte. One unsupported provenance field was removed from the final-test JSON, its notebook display, and the source expression that constructed it; every numerical result is unchanged. The notebook contains all 11 code cells with added explanatory Markdown. The source cleanup occurred after the recorded execution and did not rerun the model. Interactive progress/download widgets and the raw review-text display were removed for the public-facing copy; the reviewed validation row remains identifiable.

`analysis/verification.json` records the completed local checks. Validation metrics were recalculated from saved predictions. Final test metrics were checked against the supplied confusion counts and denominators, not against raw test predictions, which were not saved. No model was retrained during packaging.

### Sources and license

See [ATTRIBUTION.md](ATTRIBUTION.md) and the retained upstream [LICENSE](LICENSE). Raw review texts and model checkpoints are not bundled. The dataset card lists its license as unknown; the code license does not relicense dataset content.
