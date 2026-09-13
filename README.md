# Customer Churn Prediction

An end-to-end machine-learning classification project that predicts whether a customer will churn (`Yes` or `No`). The project includes data quality checks, cleaning, EDA, preprocessing, two classification models, complete evaluation, a saved pipeline, and a Streamlit application.

## Project files

- `customer_churn_data.csv` — source dataset
- `model_training.ipynb` — complete analysis and model training
- `churn_pipeline.pkl` — trained preprocessing and model pipeline
- `app.py` — Streamlit prediction application
- `requirements.txt` — Python dependencies

## Method

1. Load and inspect the customer churn dataset.
2. Report missing values, duplicates, category values, and target distribution.
3. Remove exact duplicates and exclude `customer_id` from prediction.
4. Explore churn relationships using six visualizations.
5. Split data using a stratified 80/20 train-test split.
6. Preprocess numeric and categorical features with a `ColumnTransformer`.
7. Train Logistic Regression and Random Forest pipelines.
8. Compare accuracy, precision, recall, F1, confusion matrices, and classification reports.
9. Select the final model using test-set F1 score.
10. Save and reload the complete pipeline for a manual prediction.

## Dataset findings

- Original shape: 1,220 rows and 17 columns.
- Exact duplicate rows removed: 20.
- Final modelling rows: 1,200.
- Missing values occur in three numeric and three categorical features and are handled inside the pipeline.
- Churn after de-duplication: 56.7% Yes and 43.3% No.
- Month-to-month contracts have the highest churn rate (71.7%).
- Fiber optic customers have the highest churn rate by internet-service category (69.1%).

## Model results

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.767 | 0.812 | 0.765 | 0.788 |
| Random Forest | 0.692 | 0.731 | 0.721 | 0.726 |

Logistic Regression is the final model because it achieved the strongest test-set F1 score while also producing better accuracy, precision, and recall in this experiment.

## Marking checklist

- Data understanding and quality checks: notebook sections 2–3
- Data cleaning decisions: section 4
- Six EDA visualizations and observations: section 5
- Feature preparation and preprocessing pipeline: sections 6–8
- Two trained classifiers: section 9
- Full metrics, reports, matrices, and comparison: section 10
- Evidence-based final selection: section 11
- Saved and reloaded pipeline with manual prediction: sections 12–13
- Streamlit app with all model inputs and probability: `app.py`
- GitHub and deployment instructions: this README

## Run the notebook

Place the CSV in the same folder as the notebook, then run every cell from top to bottom:

```bash
jupyter notebook model_training.ipynb
```

The notebook creates `churn_pipeline.pkl` in the project folder.

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit and test several customer profiles.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the six project files listed above. Do not upload `venv`, `.venv`, or other environment folders.
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select **Create app** and connect the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the app and test the public URL.

## Important implementation details

- The app loads the exact pipeline trained in the notebook.
- The pipeline handles missing values, one-hot encoding, and numeric scaling.
- Predictions are not hard-coded.
- `customer_id` is excluded because it is an identifier, not a generalizable customer behaviour.
- The fixed random state makes the experiment reproducible.
