import numpy as np
import pandas as pd

FEATURES = ["days_to_renewal","unresolved_questions","engagement_score","premium_change_pct","recent_plan_change"]
ACTIONS = ["renewal_outreach","benefit_education","plan_review"]

def make_data(n=3000, seed=42):
    rng=np.random.default_rng(seed)
    d=pd.DataFrame({
      "days_to_renewal":rng.integers(0,180,n),
      "unresolved_questions":rng.poisson(1.5,n),
      "engagement_score":rng.uniform(0,1,n),
      "premium_change_pct":rng.normal(5,8,n).clip(-15,40),
      "recent_plan_change":rng.integers(0,2,n)})
    scores=np.c_[
      3*(d.days_to_renewal<35)+1.2*d.engagement_score,
      1.5*d.unresolved_questions+1.2*d.recent_plan_change,
      .12*d.premium_change_pct+1.5*(d.days_to_renewal<75)]
    scores += rng.normal(0,.8,scores.shape)
    d["action"]=[ACTIONS[i] for i in scores.argmax(axis=1)]
    return d
