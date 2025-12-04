Here is an updated `README.md` that goes straight into the project, with no coursework wording.

***

# Telco Customer Churn Prediction

This project builds and evaluates machine learning models to predict whether a telecom customer will churn (cancel their subscription) using the Telco Customer Churn dataset.[1][2]

## Project overview

The models use customer attributes such as contract type, tenure, monthly charges, and service usage to estimate churn risk.[2]
The pipeline includes exploratory data analysis, data preprocessing and feature engineering, supervised learning (Decision Tree and Neural Network), unsupervised learning (KMeans), and an ethics and deployment discussion.[1]

## Repository structure

- `CM2604_ML_Coursework_Full_Notebook.py` – main script with EDA, preprocessing, models, evaluation, and ethics section.  
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` – Telco Customer Churn dataset.[2]
- `preprocessed_data.pkl` – preprocessed train/test data and feature names.  
- `scaler.save` – fitted `StandardScaler` for reuse in deployment.  
- `best_decision_tree.joblib` – tuned Decision Tree model.  
- `neural_network_model.h5` – best Neural Network model.  
- `model_results_summary.csv` – summary of model metrics.  
- `ethics_and_deployment.txt` – notes on ethical considerations and deployment strategy.  
- `GIT_INSTRUCTIONS.txt` – internal Git usage notes (optional for public repos).

## Data and preprocessing

The project uses the public Telco Customer Churn dataset, which contains customer IDs, demographic data, contract details, service options, billing information, and a binary churn label.[1][2]
Preprocessing steps include dropping `customerID`, converting `TotalCharges` to numeric and imputing missing values, consolidating “No internet service” and “No phone service”, one‑hot encoding categorical features, stratified train–test split, optional SMOTE for class balancing, and feature scaling with `StandardScaler`.[2][1]

## Models and evaluation

- **Unsupervised: KMeans**  
  - KMeans with 2 clusters on the scaled features, visualised using 2D PCA.  
  - Cluster labels are mapped to churn/not‑churn via majority voting to compute a “cluster→label” accuracy, enabling comparison with supervised models.[1][2]

- **Decision Tree**  
  - DecisionTreeClassifier tuned with GridSearchCV over depth and split/leaf size hyperparameters using ROC AUC as the scoring metric.[1]
  - Evaluated on the test set with accuracy, ROC AUC, confusion matrix, classification report, and ROC curve.[1]

- **Neural Network**  
  - Feed‑forward network implemented in Keras with dense ReLU layers, dropout, and sigmoid output for binary classification.[1]
  - Manual hyperparameter search over hidden layer sizes, dropout rates, and epochs, with early stopping and selection based on ROC AUC on the test set.[1]

`model_results_summary.csv` aggregates accuracy and ROC AUC for the Decision Tree and Neural Network, alongside the KMeans cluster→label accuracy.[2]

## Ethics and deployment

The project discusses:[1]

- Data privacy and removal of direct identifiers, with reference to principles like data minimisation and secure storage.  
- Fairness and bias, including class imbalance and potential differential impact across customer groups.  
- Model explainability, contrasting interpretable trees with less transparent neural networks and suggesting tools such as SHAP/LIME.  
- Post‑deployment monitoring for data and concept drift, performance degradation, retraining schedules, and rollback plans.  
- Transparency with users, consent, and human review for high‑impact decisions.

These points are captured in `ethics_and_deployment.txt` and can be reused in documentation or reports.[1]

## How to run

1. Install dependencies (`pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `tensorflow`, `imblearn`, `joblib`).  
2. Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project root or update the `EXCEL_PATHS` list in the script.[2]
3. Execute `CM2604_ML_Coursework_Full_Notebook.py` from top to bottom (or convert it into a Jupyter notebook and run all cells) to reproduce preprocessing, training, evaluation, and artifact generation.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/66156121/c5b54b4f-b82e-4c80-8346-c4073898d039/Coursework-Specification_202526.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/66156121/1e005482-ff22-45bb-9fbf-3d797d437bd6/WA_Fn-UseC_-Telco-Customer-Churn.csv)
