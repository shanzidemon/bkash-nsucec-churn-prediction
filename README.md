# 🥈 bKash Presents NSUCEC Cybernauts Datathon 2026

# Customer Churn Prediction Pipeline

## 🏆 Competition Achievement

**Final Result: 2nd Runner Up**

Competition:
**bKash Presents NSUCEC Cybernauts Datathon 2026**

Ranking:

* Top 30 teams selected for onsite round
* Final Top 5 teams
* Achieved 2nd Runner Up position

Competition Scale:

* Participants: 589
* Teams: 207
* Total submissions: 1597

---

# 📌 Project Overview

This project implements an end-to-end **customer churn prediction system** for a fintech mobile wallet platform.

The objective was to predict customers who are likely to churn based on historical behavioral, transactional, and account-level information.

The solution focuses on:

* Large-scale data processing
* Behavioral feature engineering
* Imbalanced classification
* Machine learning optimization
* Model explainability

---

# 🏗️ Pipeline Architecture

```
Raw Transaction Data
          |
          |
   Data Processing
     (PySpark)
          |
          |
 Feature Engineering
          |
          |
 Account Level Features
          |
          |
 LightGBM Classifier
          |
          |
 Optuna Hyperparameter Tuning
          |
          |
 SHAP Explainability
          |
          |
 Churn Probability Prediction

```

---

# 🛠️ Technologies Used

## Data Processing

* Apache Spark
* PySpark
* PyArrow

## Machine Learning

* LightGBM
* Scikit-learn
* Optuna

## Explainability

* SHAP

## Programming Language

* Python

---

# 📂 File Description

| File                     | Description                               |
| ------------------------ | ----------------------------------------- |
| churn_pipeline.py        | Complete training and prediction pipeline |
| config.json              | Model and pipeline configuration          |
| predictions.csv          | Final churn predictions                   |
| metrics.json             | Model evaluation metrics                  |
| lightgbm_churn_model.txt | Trained LightGBM model                    |
| feature_importance.csv   | Feature importance ranking                |
| shap_summary.png         | SHAP explainability visualization         |
| requirements.txt         | Required Python packages                  |

---

# 📊 Model Performance

Final Model:

```
LightGBM Gradient Boosting Classifier
```

Hyperparameter optimization:

```
Optuna Bayesian Optimization
```

Evaluation:

| Metric        |  Score |
| ------------- | -----: |
| ROC-AUC       | 0.9849 |
| Precision@10% | 0.9106 |
| Recall@10%    | 0.7188 |

Training Information:

```
Training samples : 595,000
Best iteration   : 654
```

---

# 🔍 Explainability

SHAP (SHapley Additive exPlanations) was used to understand model decisions.

Generated artifacts:

```
feature_importance.csv

shap_summary.png

```

These explain:

* Most influential churn factors
* Feature contribution
* Customer behavior patterns

---

# 🚀 How To Run

## 1. Clone Repository

```bash
git clone https://github.com/shanzidemon/bkash-nsucec-churn-prediction

cd outputs
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Pipeline

```bash
python churn_pipeline.py
```

After execution, the pipeline generates:

```
predictions.csv
metrics.json
feature_importance.csv
shap_summary.png

```

---

# 📄 Prediction Format

`predictions.csv`:

```
ACCOUNT_ID,CHURN_PROB

CUST00074245,0

CUST00290083,1

```

Where:

```
0 = Non-Churn Customer

1 = Churn Customer

```

---

# 📈 Business Impact

The model can help fintech companies:

* Identify high-risk customers early
* Prioritize retention campaigns
* Reduce customer loss
* Improve customer lifetime value

Possible interventions:

* Personalized offers
* Loyalty programs
* Proactive customer support

---

# 📌 Notes

* Developed as part of bKash Presents NSUCEC Cybernauts Datathon 2026.
* The solution follows competition guidelines.
* The model was developed using open-source machine learning frameworks.

---

# 👥 Team

Team Name:

```
YOUR_TEAM_NAME
```

Members:

```
Shanzid Helal Emon
Akib Hasan
MD Al Kayes
Nazat E Rose
```

---

# 📜 License

This project is for educational and portfolio purposes.
