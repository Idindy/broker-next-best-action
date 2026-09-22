# Broker Next-Best-Action ML Engine

Explainable ML portfolio service that ranks broker outreach actions from **synthetic member data**.

## Demonstrates
- End-to-end feature pipeline and model training
- Probability-based action ranking
- Explainability via model feature importance
- Evaluation metrics and drift monitoring
- FastAPI inference service + Docker + tests

The dataset is generated and contains no real member information. Predictions are demonstration decision-support outputs, not insurance advice.

## Train and run
```bash
pip install -r requirements.txt
python -m ml.train
uvicorn app.main:app --reload
```

Actions: `renewal_outreach`, `benefit_education`, `plan_review`.