import os
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def load_data():
    """Load UROMOL and Knowles datasets from CSV
    """
    uromol = pd.read_csv("data/uromol_preprocessed.csv")
    knowles = pd.read_csv("data/knowles_preprocessed.csv")
    
    return uromol, knowles

def split_features_labels(df):
    """Split data into features and labels (targets)
    """
    # drop ID and rfs_time columns
    df = df.drop(columns=["id", "rfs_time"], errors="ignore")
    
    # split into features and targets
    X = df.drop(columns=["recurrence"])
    y = df["recurrence"]
    
    return X, y

def reduce_features(features, k):
    """Reduce features by top k highest variances
    """
    variances = features.var(axis=0)
    top_features = variances.sort_values(ascending=False).head(k).index
    return features[top_features]

def train_elastic_net(X_train, y_train):
    """Train logistic regression model with cross-validation
    """
    model = LogisticRegressionCV(
        Cs=5,
        cv=3,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.5],
        scoring='roc_auc',
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time
    print(f"Model trained in {total_time:.2f} seconds")
    
    return model

def evaluate_model(model, X, y, name="", file=None):
    """Evaluate effectiveness of model
    Currently outputs AUC and confusion matrix
    """
    y_pred = model.predict(X)               # labels (recurrence or not)
    y_prob = model.predict_proba(X)[:, 1]   # probability of recurrence
    
    # get metrics
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    # format metrics
    output = f"\n=== {name} Evaluation ===\n"
    output += f"AUC: {round(auc, 3)}\n"
    output += "Confusion Matrix:\n"
    output += f"{cm}\n"

    print(output)
    if file:
        file.write(output)
        
def assign_risk_levels(scores):
    """Assigns Low/Medium/High risk labels based on predicted probabilities
    """
    percentiles = np.percentile(scores, [33.3, 66.6])
    return pd.cut(scores, bins=[-np.inf, percentiles[0], percentiles[1], np.inf], labels=["Low", "Medium", "High"])

def plot_kaplan_meier(rfs_time, event_observed, risk_groups, title, outpath):
    """Plots a Kaplan Meier curve with the given data and saves graph as PNG
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for group in sorted(risk_groups.unique()):
        ix = risk_groups == group
        kmf.fit(rfs_time[ix], event_observed[ix], label=f"{group} risk")
        kmf.plot(ci_show=True)

    plt.title(title)
    plt.xlabel("Recurrence-Free Survival Time (days)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    os.makedirs("outputs", exist_ok=True)
    out_metrics = open("outputs/metrics.txt", "w")

    # load data and split into features and labels
    uromol_df, knowles_df = load_data()
    X_uromol, y_uromol = split_features_labels(uromol_df)
    X_knowles, y_knowles = split_features_labels(knowles_df)
    
    # reduce features by variance
    X_uromol = reduce_features(X_uromol, 3500)
    X_knowles = X_knowles[X_uromol.columns]

    # impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_uromol_imputed = imputer.fit_transform(X_uromol)
    X_knowles_imputed = imputer.transform(X_knowles)

    # standardize features
    scaler = StandardScaler()
    X_uromol_scaled = scaler.fit_transform(X_uromol_imputed)
    X_knowles_scaled = scaler.transform(X_knowles_imputed)

    # split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split( X_uromol_scaled, y_uromol, 
                                                      test_size=0.2, stratify=y_uromol, random_state=42)

    # train model
    model = train_elastic_net(X_train, y_train)

    # evaluate model on UROMOL and Knowles
    evaluate_model(model, X_val, y_val, name="UROMOL Validation", file=out_metrics)
    evaluate_model(model, X_knowles_scaled, y_knowles, name="Knowles External", file=out_metrics)
    
    # Kaplan-Meier on UROMOL
    risk_scores_u = model.predict_proba(X_uromol_scaled)[:, 1]
    percentiles_u = np.percentile(risk_scores_u, [33.3, 66.6])
    risk_group_u = pd.cut(risk_scores_u, bins=[-np.inf, percentiles_u[0], percentiles_u[1], np.inf], labels=["Low", "Medium", "High"])
    rfs_time_u = uromol_df["rfs_time"]
    event_observed_u = uromol_df["recurrence"].astype(int)

    km_data_u = pd.DataFrame({"rfs_time": rfs_time_u, "event": event_observed_u, 
                              "risk": risk_group_u}).dropna(subset=["rfs_time", "event", "risk"])

    plot_kaplan_meier(km_data_u["rfs_time"], km_data_u["event"], km_data_u["risk"], 
                      title="Kaplan-Meier: UROMOL Risk Stratification", outpath="outputs/uromol_kaplan_meier.png")

    # Kaplan-Meier on Knowles
    risk_scores_k = model.predict_proba(X_knowles_scaled)[:, 1]
    percentiles_k = np.percentile(risk_scores_k, [33.3, 66.6])
    risk_group_k = pd.cut(risk_scores_k, bins=[-np.inf, percentiles_k[0], percentiles_k[1], np.inf], labels=["Low", "Medium", "High"])
    rfs_time_k = knowles_df["rfs_time"]
    event_observed_k = knowles_df["recurrence"].astype(int)
    
    km_data = pd.DataFrame({"rfs_time": rfs_time_k, "event": event_observed_k, 
                            "risk": risk_group_k }).dropna(subset=["rfs_time", "event", "risk"])

    plot_kaplan_meier(km_data["rfs_time"], km_data["event"], km_data["risk"], 
                      title="Kaplan-Meier: Knowles Risk Stratification", outpath="outputs/knowles_kaplan_meier.png")
    
    # assign risk levels to UROMOL
    risk_levels_uromol = assign_risk_levels(risk_scores_u)
    uromol_output = pd.DataFrame({"uromol.id": uromol_df["id"], "predicted_risk_level": risk_levels_uromol})
    uromol_output.to_csv("outputs/uromol_risk_predictions.csv", index=False)

    # assign risk levels to Knowles
    risk_levels_knowles = assign_risk_levels(risk_scores_k)
    knowles_output = pd.DataFrame({"knowles_id": knowles_df["id"], "predicted_risk_level": risk_levels_knowles})
    knowles_output.to_csv("outputs/knowles_risk_predictions.csv", index=False)

if __name__ == "__main__":
    main()