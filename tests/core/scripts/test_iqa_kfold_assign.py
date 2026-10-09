from __future__ import annotations

import collections

from fairchem.core.scripts.iqa_kfold_assign import assign_folds


def test_folds_keep_sum_formulas_together_and_are_balanced():
    formulas = {f"m{i}.pkl": f"F{i % 7}" for i in range(70)}  # 7 formulas x 10 files
    fold_of = assign_folds(formulas, k=3, seed=0)
    folds_per_formula = collections.defaultdict(set)
    for file, fold in fold_of.items():
        folds_per_formula[formulas[file]].add(fold)
    assert all(len(f) == 1 for f in folds_per_formula.values())
    sizes = collections.Counter(fold_of.values())
    assert sorted(sizes) == [0, 1, 2]
    assert max(sizes.values()) - min(sizes.values()) <= 10  # within one group


def test_fold_assignment_is_deterministic():
    formulas = {f"m{i}.pkl": f"F{i % 5}" for i in range(25)}
    assert assign_folds(formulas, 5, seed=1) == assign_folds(formulas, 5, seed=1)
