import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from .data import make_data, FEATURES

OUT=Path("artifacts"); OUT.mkdir(exist_ok=True)
df=make_data(); X=df[FEATURES]; y=df["action"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,max_depth=10,min_samples_leaf=4,random_state=42,class_weight="balanced").fit(Xtr,ytr)
pred=model.predict(Xte)
metrics={"accuracy":accuracy_score(yte,pred),"report":classification_report(yte,pred,output_dict=True),"feature_importance":dict(zip(FEATURES,model.feature_importances_.tolist())),"training_mean":Xtr.mean().to_dict(),"training_std":Xtr.std().to_dict()}
joblib.dump(model,OUT/"model.joblib"); (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(f"accuracy={metrics['accuracy']:.3f}")
