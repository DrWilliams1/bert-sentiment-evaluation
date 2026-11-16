"""Audit the recorded results without model downloads, training, or test inference.

Portfolio verification extension. See ../ATTRIBUTION.md.
Run from any working directory: python /path/to/scripts/verify_results.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = '20261115T224719Z-739463'
RESULTS = ROOT / 'results' / RUN_ID


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def close(actual, expected, message):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), message)


def scores_from_confusion(cm):
    n = int(cm.sum())
    require(cm.shape == (2, 2) and n > 0 and (cm >= 0).all(), 'Invalid confusion counts')
    f1 = []
    for label in (0, 1):
        denominator = cm[label, :].sum() + cm[:, label].sum()
        f1.append(float(2 * cm[label, label] / denominator) if denominator else 0.0)
    return {'accuracy': float(cm.trace() / n), 'binary_f1_positive': f1[1], 'macro_f1': sum(f1) / 2}


def main():
    records = read('validation_comparison.json')
    require(len(records) == 2, 'Expected exactly two regime records')
    require({r['regime'] for r in records} == {'head_only', 'full_finetune'}, 'Unexpected regimes')
    reference_labels = None
    checked = []
    error_rows_checked = 0
    original_observations = 0

    for record in records:
        regime = record['regime']
        require(record == read(f'{regime}_validation_metrics.json'), 'Comparison and regime JSON differ')
        with np.load(RESULTS / f'{regime}_validation_predictions.npz', allow_pickle=False) as saved:
            logits = saved['logits'].astype(np.float64)
            labels = saved['labels']
        require(logits.shape == (1066, 2), 'Unexpected validation logits shape')
        require(labels.shape == (1066,) and np.isin(labels, [0, 1]).all(), 'Invalid validation labels')
        require(np.isfinite(logits).all(), 'Nonfinite logits')
        require(np.bincount(labels.astype(int), minlength=2).tolist() == [533, 533], 'Class counts differ')
        if reference_labels is None:
            reference_labels = labels.copy()
        else:
            require(np.array_equal(labels, reference_labels), 'Regimes have different validation labels/order')

        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        cm = np.array([[int(((labels == actual) & (predicted == pred)).sum()) for pred in (0, 1)] for actual in (0, 1)])
        require(cm.tolist() == record['confusion_matrix_rows_actual_columns_predicted'], 'Confusion matrix mismatch')
        require(record['n_examples'] == len(labels), 'Example count mismatch')
        for key, value in scores_from_confusion(cm).items():
            close(value, record[key], f'{regime}: {key} mismatch')
        require(record['confidence_threshold'] == 0.9, 'Score threshold changed')
        confident = confidence >= record['confidence_threshold']
        wrong = predicted != labels
        count, errors = int(confident.sum()), int((confident & wrong).sum())
        require(count == record['confident_count'] and errors == record['confident_errors'], 'Confident counts mismatch')
        if count:
            close(errors / count, record['confident_error_rate'], 'Confident-error rate mismatch')
        else:
            require(record['confident_error_rate'] is None, 'Undefined conditional rate must be null')

        if regime == 'full_finetune':
            with (RESULTS / 'full_finetune_error_review.csv').open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            require(len(rows) == int(wrong.sum()), 'Error worksheet has missing or extra rows')
            require({int(row['validation_row_id']) for row in rows} == set(np.flatnonzero(wrong).tolist()), 'Wrong row IDs')
            for row in rows:
                i = int(row['validation_row_id'])
                require(int(row['actual_label']) == int(labels[i]), 'Worksheet actual label mismatch')
                require(int(row['predicted_label']) == int(predicted[i]), 'Worksheet prediction mismatch')
                close(row['softmax_score'], confidence[i], 'Worksheet confidence mismatch')
                require(row['correct'] == 'False', 'Worksheet contains a correct prediction')
            require(all(float(rows[i]['softmax_score']) >= float(rows[i + 1]['softmax_score']) for i in range(len(rows) - 1)), 'Error ranking mismatch')
            error_rows_checked = len(rows)
            original_observations = sum(bool(row['my_observation'].strip()) for row in rows)
        checked.append(regime)

    require(len({r['initial_head_sha256'] for r in records}) == 1, 'Recorded classifier initialization hashes differ')
    selection = read('selection.json')
    require(selection['rule'] == 'highest_validation_macro_f1_then_fastest_training', 'Unexpected selection rule')
    chosen = sorted(records, key=lambda r: (-r['macro_f1'], r['train_wall_seconds']))[0]['regime']
    require(selection['selected_regime'] == chosen, 'Selection disagrees with validation records')

    final = read('final_test_metrics.json')
    require(final['regime'] == chosen and final['split'] == 'test', 'Final record identity mismatch')
    cm = np.asarray(final['confusion_matrix_rows_actual_columns_predicted'])
    require(cm.sum() == final['n_examples'] == 1066, 'Final test denominator mismatch')
    require(cm.sum(axis=1).tolist() == [533, 533], 'Final test class counts mismatch')
    for key, value in scores_from_confusion(cm).items():
        close(value, final[key], 'Final summary arithmetic: ' + key)
    require(0 <= final['confident_errors'] <= final['confident_count'] <= final['n_examples'], 'Invalid test counts')
    close(final['confident_errors'] / final['confident_count'], final['confident_error_rate'], 'Final conditional-rate arithmetic')

    report = {
        'status': 'pass',
        'run_id': RUN_ID,
        'validation_regimes_recomputed_from_saved_logits': checked,
        'validation_examples_per_regime': 1066,
        'validation_labels_and_order_match_between_regimes': True,
        'full_finetune_error_rows_checked': error_rows_checked,
        'original_worksheet_observations_filled': original_observations,
        'qualitative_examples_reviewed': 1,
        'recorded_initial_classifier_weight_hashes_match': True,
        'selection_matches_recorded_rule': True,
        'selected_regime': chosen,
        'test_summary_arithmetic_verified': True,
        'test_metrics_recomputed_from_saved_logits': False,
        'test_recomputation_limitation': 'Final test logits and per-example labels were not included in the export.',
        'training_or_inference_performed': False,
        'verification_scope': 'Local numerical consistency checks, not independent experiment replication or independent annotation reliability.'
    }
    target = ROOT / 'analysis/verification.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
