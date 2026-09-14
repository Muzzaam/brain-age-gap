"""Tests for the RQ2 two-stage utility harness.  Run: python -m tests.test_rq2"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import load_data, schema
from src.preprocessing import prepare_data
from src.models.two_stage import estimate_bag, build_conditions, outcome_target
from src.evaluation.outcome_metrics import cognitive_metrics, paired_error_test
from scripts.run_rq2 import run

def _data(n=1000, seed=5): return prepare_data(load_data(source="synthetic", n=n, seed=seed), seed=seed)

def test_estimate_bag_shapes():
    d=_data(); bag=estimate_bag(d,model_name="xgboost")
    assert bag["train"].shape[0]==d["X_train"].shape[0]
    assert bag["test"].shape[0]==d["X_test"].shape[0] and np.isfinite(bag["test"]).all()

def test_four_conditions_have_expected_widths():
    d=_data(); conds=build_conditions(d,estimate_bag(d),seed=1); n=len(schema.all_biomarker_columns())
    w={k:Xtr.shape[1] for k,(Xtr,Xte) in conds.items()}
    assert w["age_only"]==1 and w["biomarkers"]==n and w["biomarkers+BAG"]==n+1 and w["biomarkers+placebo"]==n+1
    for k,(Xtr,Xte) in conds.items(): assert Xtr.shape[1]==Xte.shape[1]

def test_outcome_target_aligned():
    d=_data(); yt,ye=outcome_target(d,schema.COGNITIVE_COL)
    assert yt.shape[0]==d["X_train"].shape[0] and ye.shape[0]==d["X_test"].shape[0]

def test_paired_test_detects_real_improvement():
    rng=np.random.default_rng(0); y=rng.normal(size=200)
    res=paired_error_test(y,y+rng.normal(scale=2.0,size=200),y+rng.normal(scale=0.5,size=200))
    assert res["rmse_improvement"]>0 and res["wilcoxon_p"]<0.05

def test_paired_test_identical_predictions_gives_p_one():
    y=np.arange(10,dtype=float); res=paired_error_test(y,y+1.0,y+1.0)
    assert res["rmse_improvement"]==0 and res["wilcoxon_p"]==1.0

def test_cognitive_metrics_keys():
    m=cognitive_metrics([1,2,3],[1.1,1.9,3.2])
    assert set(m)=={"RMSE","Pearson_r"} and -1.0<=m["Pearson_r"]<=1.0

def test_run_end_to_end():
    report,y_test=run(n=800,seed=3)
    for s in ("ridge","xgboost"):
        assert s in report
        for c in ("age_only","biomarkers","biomarkers+BAG","biomarkers+placebo"):
            m=report[s]["metrics"][c]; assert np.isfinite(m["RMSE"]) and np.isfinite(m["Pearson_r"])

if __name__=="__main__":
    for nm,fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn): fn(); print(f"PASSED {nm}")
