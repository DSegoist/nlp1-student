import numpy as np
import itertools
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from sklearn.metrics import accuracy_score
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
import random
from part1.clslib import BagOfWords
import part1.experiment

from . import loglin
from . import smlp
from . import clslib

def cross_validation_experiment_with_bow_encoder(
    data_splits, outcome_space, tokenizer, model_option,
    save_to="tmp", loglin_args=None, smlp_args=None
):
    encoder = BagOfWords(tokenizer)

    return part1.experiment.cross_validation_experiment(
        data_splits=data_splits,
        outcome_space=outcome_space,
        encoder=encoder,
        model_option=model_option,
        save_to=save_to,
        loglin_args={} if loglin_args is None else loglin_args,
        smlp_args={} if smlp_args is None else smlp_args
    )


def compare_all_pairs(classifiers, rope_size=0.005):
    """Compare paired cross-validation accuracies and plot each posterior.

    All systems must use the same runs and validation folds.
    Returns a dictionary mapping system-name pairs to posterior summaries.
    """
    from .analysis import BenavoliCorrelatedT
    import matplotlib.pyplot as plt

    if rope_size < 0:
        raise ValueError("rope_size must be non-negative")

    results = {}
    for name_a, name_b in itertools.combinations(classifiers, 2):
        runs_a, runs_b = classifiers[name_a], classifiers[name_b]
        if not runs_a or len(runs_a) != len(runs_b):
            raise ValueError(f"{name_a} and {name_b} must have the same nonzero number of runs")

        scores_a, scores_b = [], []
        num_folds = None
        for run_a, run_b in zip(runs_a, runs_b):
            acc_a, acc_b = run_a['accuracies'], run_b['accuracies']
            if set(acc_a) != set(acc_b):
                raise ValueError(f"{name_a} and {name_b} must have matching fold IDs")
            if num_folds is None:
                num_folds = len(acc_a)
            if num_folds < 2 or len(acc_a) != num_folds:
                raise ValueError("Each run must have the same number of folds, at least two")
            for fold in sorted(acc_a):
                scores_a.append(acc_a[fold])
                scores_b.append(acc_b[fold])

        comparison = BenavoliCorrelatedT(
            scores_a, scores_b, runs=len(runs_a), names=[name_a, name_b]
        )
        p_a, p_tie, p_b = comparison.probs(rope_size)
        results[(name_a, name_b)] = {
            'mean_diff': comparison.mean_diff,
            'effect_size': comparison.effect_size,
            'p_a_better': p_a,
            'p_equivalent': p_tie,
            'p_b_better': p_b,
        }
        print(f"A={name_a} vs. B={name_b}")
        print(f"Mean accuracy difference (B - A): {comparison.mean_diff:.4f}")
        print(f"Effect size: {comparison.effect_size:.3f}")
        print(f"P(A better)={p_a:.1%}, P(equivalent)={p_tie:.1%}, P(B better)={p_b:.1%}")
        comparison.plot(rope_size)
        plt.show()

    return results
