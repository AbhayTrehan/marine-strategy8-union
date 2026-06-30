"""Run with: python3 tests/test_hyperparam_grid.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyperparam_grid import (
    DEFAULT_ALPHAS,
    DEFAULT_GMM_PRESETS,
    DEFAULT_TAUS,
    TrialConfig,
    TrialResult,
    build_grid,
    chair_f1,
    pick_best,
    select_gmm_presets,
)


def test_chair_f1_basic():
    # perfect precision, perfect recall -> f1 = 1
    assert abs(chair_f1(chair_i=0.0, recall=1.0) - 1.0) < 1e-9
    # zero precision, zero recall -> f1 = 0
    assert chair_f1(chair_i=1.0, recall=0.0) == 0.0
    # standard harmonic mean check
    f1 = chair_f1(chair_i=0.2, recall=0.5)  # P=0.8, R=0.5
    expected = 2 * 0.8 * 0.5 / (0.8 + 0.5)
    assert abs(f1 - expected) < 1e-9
    print("test_chair_f1_basic OK")


def test_chair_f1_handles_pathological_chairi():
    # CHAIRi should never exceed 1 in practice but guard anyway
    f1 = chair_f1(chair_i=1.5, recall=0.3)
    assert f1 == 0.0  # precision clipped to 0 -> f1 = 0
    print("test_chair_f1_handles_pathological_chairi OK")


def test_build_grid_full_when_small():
    trials = build_grid(gmm_presets=DEFAULT_GMM_PRESETS[:1], taus=[0.5], alphas=[0.5], max_trials=100)
    assert len(trials) == 1
    print("test_build_grid_full_when_small OK")


def test_build_grid_caps_and_is_reproducible():
    full_size = len(DEFAULT_GMM_PRESETS) * len(DEFAULT_TAUS) * len(DEFAULT_ALPHAS)
    assert full_size > 12
    trials_a = build_grid(max_trials=12, seed=42)
    trials_b = build_grid(max_trials=12, seed=42)
    trials_c = build_grid(max_trials=12, seed=7)
    assert len(trials_a) == 12
    ids_a = [t.trial_id for t in trials_a]
    ids_b = [t.trial_id for t in trials_b]
    assert ids_a == ids_b, "same seed should give same sample"
    ids_c = [t.trial_id for t in trials_c]
    assert ids_a != ids_c, "different seed should (almost certainly) differ"
    print("test_build_grid_caps_and_is_reproducible OK")


def test_build_grid_no_cap_returns_full_cross_product():
    trials = build_grid(max_trials=None)
    assert len(trials) == len(DEFAULT_GMM_PRESETS) * len(DEFAULT_TAUS) * len(DEFAULT_ALPHAS)
    print("test_build_grid_no_cap_returns_full_cross_product OK")


def test_pick_best_selects_max_f1():
    t1 = TrialConfig("a", DEFAULT_GMM_PRESETS[0], 0.5, 0.5)
    t2 = TrialConfig("b", DEFAULT_GMM_PRESETS[0], 0.4, 0.7)
    r1 = TrialResult(t1, chair_s=0.1, chair_i=0.05, recall=0.4, f1=chair_f1(0.05, 0.4), n_images=300)
    r2 = TrialResult(t2, chair_s=0.05, chair_i=0.02, recall=0.5, f1=chair_f1(0.02, 0.5), n_images=300)
    best = pick_best([r1, r2])
    assert best is r2
    print("test_pick_best_selects_max_f1 OK")


def test_pick_best_empty_raises():
    try:
        pick_best([])
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_pick_best_empty_raises OK")


def test_trial_result_roundtrip():
    t = TrialConfig("a", DEFAULT_GMM_PRESETS[-1], 0.5, 0.5)  # fixed_prior preset, has nested lists
    r = TrialResult(t, chair_s=0.1, chair_i=0.05, recall=0.4, f1=0.5, n_images=300, extra={"note": "x"})
    d = r.to_dict()
    r2 = TrialResult.from_dict(d)
    assert r2.trial.gmm_preset == t.gmm_preset
    assert r2.f1 == r.f1
    print("test_trial_result_roundtrip OK")


def test_select_gmm_presets_default_excludes_damped():
    presets = select_gmm_presets(tune_learning_rate=False)
    assert all(p["learning_rate"] == 1.0 for p in presets)
    print("test_select_gmm_presets_default_excludes_damped OK")


def test_select_gmm_presets_activated_includes_damped():
    presets = select_gmm_presets(tune_learning_rate=True)
    lrs = {p["learning_rate"] for p in presets}
    assert 1.0 in lrs
    assert any(lr < 1.0 for lr in lrs)
    print("test_select_gmm_presets_activated_includes_damped OK")


def test_default_gmm_presets_matches_deactivated_selection():
    assert DEFAULT_GMM_PRESETS == select_gmm_presets(tune_learning_rate=False)
    print("test_default_gmm_presets_matches_deactivated_selection OK")


def test_preferred_first_appears_first_without_capping():
    trials = build_grid(
        gmm_presets=DEFAULT_GMM_PRESETS[:1], taus=[0.2, 0.3, 0.4], alphas=[0.5, 0.6, 0.7],
        max_trials=None, preferred_first={"tau": 0.3, "alpha": 0.7},
    )
    assert len(trials) == 9
    assert trials[0].tau == 0.3 and trials[0].alpha == 0.7
    # everything else still present, just reordered
    assert {(t.tau, t.alpha) for t in trials} == {(t, a) for t in [0.2, 0.3, 0.4] for a in [0.5, 0.6, 0.7]}
    print("test_preferred_first_appears_first_without_capping OK")


def test_preferred_first_guaranteed_a_slot_under_capping():
    # 4x4=16 combos, cap to 3 -- without forcing, (0.3, 0.7) might easily be
    # sampled out; with preferred_first it must always be slot 0.
    trials = build_grid(
        gmm_presets=DEFAULT_GMM_PRESETS[:1], taus=DEFAULT_TAUS, alphas=DEFAULT_ALPHAS,
        max_trials=3, seed=99, preferred_first={"tau": 0.3, "alpha": 0.7},
    )
    assert len(trials) == 3
    assert trials[0].tau == 0.3 and trials[0].alpha == 0.7
    print("test_preferred_first_guaranteed_a_slot_under_capping OK")


def test_preferred_first_noop_when_not_in_grid():
    trials = build_grid(
        gmm_presets=DEFAULT_GMM_PRESETS[:1], taus=[0.2, 0.3], alphas=[0.5, 0.6],
        max_trials=None, preferred_first={"tau": 0.99, "alpha": 0.99},
    )
    assert len(trials) == 4  # unaffected, no matching entry to force to front
    print("test_preferred_first_noop_when_not_in_grid OK")


def test_default_ranges_match_user_request():
    assert DEFAULT_TAUS == [0.2, 0.3, 0.4, 0.5]
    assert DEFAULT_ALPHAS == [0.5, 0.6, 0.7, 0.8]
    print("test_default_ranges_match_user_request OK")


if __name__ == "__main__":
    test_chair_f1_basic()
    test_chair_f1_handles_pathological_chairi()
    test_build_grid_full_when_small()
    test_build_grid_caps_and_is_reproducible()
    test_build_grid_no_cap_returns_full_cross_product()
    test_pick_best_selects_max_f1()
    test_pick_best_empty_raises()
    test_trial_result_roundtrip()
    test_select_gmm_presets_default_excludes_damped()
    test_select_gmm_presets_activated_includes_damped()
    test_default_gmm_presets_matches_deactivated_selection()
    test_preferred_first_appears_first_without_capping()
    test_preferred_first_guaranteed_a_slot_under_capping()
    test_preferred_first_noop_when_not_in_grid()
    test_default_ranges_match_user_request()
    print("\nALL hyperparam_grid.py TESTS PASSED")
