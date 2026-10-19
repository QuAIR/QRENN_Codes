"""Predeclared score policy for the new Shoppers/Rice campaign.

Helpers only: scheduling and explicit held-out authorization live in evaluation_parallel.
The earlier four-dataset replay investigation is historical numerical precedent.
"""
from pathlib import Path
import numpy as np
from . import evaluate as original, data, train, kernels as k

ORIGINAL_ATOL=2e-10
SVM_ATOL=2e-8


def compare_scores(y, reference, replayed, *, kind, method):
    reference, replayed = np.asarray(reference), np.asarray(replayed)
    if reference.ndim != 1 or reference.shape != replayed.shape or reference.shape != np.asarray(y).shape or not np.isfinite(reference).all() or not np.isfinite(replayed).all():
        raise ValueError('Invalid numerical replay arrays')
    if train.protocol()['evaluation']['score_atol'] != ORIGINAL_ATOL or train.protocol()['evaluation']['svm_score_atol'] != SVM_ATOL:
        raise ValueError('Predeclared numerical protocol changed')
    audit = original.discrepancy(reference, replayed)
    limit = SVM_ATOL if (kind == 'kernel' and method in k.METHODS) or (kind == 'classical' and method == 'rbf_svm') else ORIGINAL_ATOL
    ref_metrics, replay_metrics = k.metrics(y, reference), k.metrics(y, replayed)
    ref_rank, replay_rank = original.rankdata(reference,method='average'), original.rankdata(replayed,method='average')
    audit.update(original_score_atol=ORIGINAL_ATOL, accepted_score_atol=limit,
                 original_policy_passed=audit['max_abs_score_error'] <= ORIGINAL_ATOL,
                 classification_agreement_required=True,
                 reference_metrics=ref_metrics, replay_metrics=replay_metrics,
                 metric_differences={name: replay_metrics[name]-ref_metrics[name] for name in ref_metrics},
                 classification_difference_row_indices=np.flatnonzero((reference>=0)!=(replayed>=0)).tolist(),
                 ranking_differences=[dict(row_index=int(i),reference_rank=float(ref_rank[i]),replay_rank=float(replay_rank[i]),reference_score=float(reference[i]),replay_score=float(replayed[i])) for i in np.flatnonzero(ref_rank!=replay_rank)],
                 accepted=audit['max_abs_score_error'] <= limit and audit['classification_disagreements'] == 0)
    return audit


def require_accepted(audit):
    if not audit['accepted']:
        raise ValueError('Numerical replay requires investigation: score tolerance exceeded or classification changed')


def policy():
    here=Path(__file__).resolve()
    return dict(schema='qrenn-a07-evaluation-numerical-policy-v1',
        declaration='predeclared_before_new_campaign_fitting',
        original_score_atol=ORIGINAL_ATOL,svm_score_atol=SVM_ATOL,
        strict_other_models_score_atol=ORIGINAL_ATOL,relative_tolerance=0,
        scope='SVM scores: fidelity/projected kernels and classical rbf_svm; all other models use strict tolerance',
        classification_disagreements_allowed=0,
        ranking_changes='Report exact rank disagreement count, affected rows and all four metric deltas; never repair scores',
        validation_table_authority='Frozen native saved validation predictions',
        test_table_authority='Original transformed-feature production inference',
        model_training_selection_and_metrics_changed=False,predictions_rounded=False,
        supplement_sha256=data.digest(here),supplement_source_file=here.name,
        choice='Fixed absolute tolerances are investigation thresholds, not universal forward-error proofs. Diagnostic gamma_n times dual-L1 bounds never expand acceptance.',
        previous_campaign_precedent=dict(source_commit='55ac5e0',
            snapshot_sha256='8e725c8bc416e70833e53444f5678fa8eef406616a0ffbcac41a91033bc0ff0f',
            role='Historical four-dataset numerical investigation only; neither source nor snapshot authorizes this campaign',
            projected_native_max_abs_score_error=6.770051186322235e-10,
            projected_independent_max_abs_score_error=6.915570338605903e-10,
            classification_disagreements=0,ranking_disagreements=0,
            artifacts=['failed-projected-replay.json','roundoff-root-cause.json','classical-rbf-roundoff-check.json']))
