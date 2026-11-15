# Attribution and changes

## Learning baseline

The original portfolio draft credits the BERT classification and parameter-freezing baseline to *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst (2024), Chapter 11.

- Original chapter code: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter11
- Original code copyright: 2024 Jay Alammar & Maarten Grootendorst.
- Upstream license: Apache License 2.0; its full license and copyright notice are retained in `LICENSE`.
- License source: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/LICENSE
- Retrieved license blob SHA: `876836a78a702053d4ca0c7d1ce0e624efdf891c`.

The code is an adaptation of that baseline. The extensions add a documented evaluation workflow and interpretation. The upstream code is Apache-2.0 licensed; the dataset has separate terms.

## Portfolio extensions

The project adds a restricted two-regime comparison, validation-based selection, one selected-model final test, timing and parameter counts, saved numerical predictions, explicit confidence-error denominators, revision/environment records, and an error-review worksheet. Joe Williams executed the supplied Colab run and supplied its outputs.

Packaging changes on November 15, 2026: added explanatory Markdown around the executed cells; added README, verification script, and interpretation and verification records; removed interactive progress/download displays and the displayed raw review excerpt. Training, evaluation, and metric logic and all numerical results were retained. A later packaging correction removed one unsupported provenance field from the final-test record, notebook display, and corresponding source expression. The notebook was not re-executed after that metadata-only source correction. The original error worksheet remains unchanged. A separate CSV contains the qualitative interpretation of the one example reviewed.

## Model and data

- Model: https://huggingface.co/google-bert/bert-base-cased
- Model snapshot: `cd5ef92a9fb2f889e972770a36d4ed042daf221e`.
- Dataset: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
- Dataset snapshot: `aa13bc287fa6fcab6daf52f0dfb9994269ffea28`.
- The model card lists Apache-2.0. The dataset card lists the dataset license as unknown (checked November 15, 2026). This package includes identifiers, labels, scores, and derived analysis rather than a copy of the review corpus.

