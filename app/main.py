import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml.data import FEATURES

app=FastAPI(title="Broker Next-Best-Action",version="0.1.0")
ART=Path("artifacts")
model=joblib.load(ART/"model.joblib") if (ART/"model.joblib").exists() else None
metrics=json.loads((ART/"metrics.json").read_text()) if (ART/"metrics.json").exists() else {}

class MemberFeatures(BaseModel):
    days_to_renewal:int=Field(ge=0,le=365)
    unresolved_questions:int=Field(ge=0,le=50)
    engagement_score:float=Field(ge=0,le=1)
    premium_change_pct:float=Field(ge=-100,le=200)
    recent_plan_change:int=Field(ge=0,le=1)

@app.get("/health")
def health(): return {"status":"ok","model_loaded":model is not None}

@app.post("/rank-actions")
def rank_actions(x:MemberFeatures):
    if model is None: raise HTTPException(503,"Run: python -m ml.train")
    row=pd.DataFrame([[getattr(x,f) for f in FEATURES]],columns=FEATURES)
    probs=model.predict_proba(row)[0]
    ranked=sorted([{"action":a,"probability":round(float(p),4)} for a,p in zip(model.classes_,probs)],key=lambda z:z["probability"],reverse=True)
    return {"ranked_actions":ranked,"explanation":{"global_feature_importance":metrics.get("feature_importance",{})},"disclaimer":"Synthetic demonstration decision support; not insurance advice."}

@app.post("/drift")
def drift(x:MemberFeatures):
    means=metrics.get("training_mean",{}); std=metrics.get("training_std",{})
    z={f:abs(getattr(x,f)-means[f])/max(std[f],1e-9) for f in FEATURES}
    return {"max_z_score":round(max(z.values()),3),"feature_z_scores":{k:round(v,3) for k,v in z.items()},"drift_warning":max(z.values())>3}
