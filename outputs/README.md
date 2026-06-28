# FictiPay Churn Pipeline

Run from this folder or pass absolute output/work paths:

```bash
python churn_pipeline.py \
  --data-dir /Users/shanzidemon/Documents/CompNSU \
  --work-dir ./work/churn_pipeline \
  --output-dir ./outputs \
  --cutoff-date 2024-04-01 \
  --optuna-trials 80 \
  --spark-partitions 800 \
  --driver-memory 12g
```

Outputs:

- `predictions.csv`: `ACCOUNT_ID,CHURN_PROB`
- `metrics.json`: AUC-ROC, Precision@10%, Recall@10%
- `shap_summary.png`: SHAP top churn drivers
- `feature_importance.csv`: LightGBM gain importance
- `lightgbm_churn_model.txt`: trained model

The raw transaction and balance tables are processed in Spark. Only the final
per-account feature matrix is collected for LightGBM/Optuna.
