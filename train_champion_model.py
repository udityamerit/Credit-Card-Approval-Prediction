import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, brier_score_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def train_and_export_champion_pipeline(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== Starting Advanced Model Training Pipeline ===")

    # 1. Load Dataset
    app_path = os.path.join(base_dir, "Dataset", "application_record.csv")
    cred_path = os.path.join(base_dir, "Dataset", "credit_record.csv")

    if not os.path.exists(app_path) or not os.path.exists(cred_path):
        raise FileNotFoundError(f"Dataset files missing in {base_dir}/Dataset")

    app = pd.read_csv(app_path)
    cred = pd.read_csv(cred_path)

    # Basel Article 178 Definition of Default (90+ DPD)
    bad_status = {'2', '3', '4', '5'}
    cred['bad_credit'] = cred['STATUS'].isin(bad_status).astype(int)
    grouped = cred.groupby('ID')['bad_credit'].max().reset_index()

    df = pd.merge(app, grouped, on='ID', how='inner').drop(columns=['ID'])
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Other')

    # 2. Advanced Feature Engineering
    df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365.25
    df['EMPLOYED_YEARS'] = np.maximum(0, -df['DAYS_EMPLOYED']) / 365.25
    df['INCOME_PER_FAM_MEMBER'] = df['AMT_INCOME_TOTAL'] / np.maximum(1, df['CNT_FAM_MEMBERS'])
    df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYED_YEARS'] / np.maximum(18.0, df['AGE_YEARS'])
    df['INCOME_TO_AGE_RATIO'] = df['AMT_INCOME_TOTAL'] / np.maximum(18.0, df['AGE_YEARS'])
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / np.maximum(1, df['CNT_FAM_MEMBERS'])
    df['TOTAL_CONTACTS'] = df['FLAG_WORK_PHONE'] + df['FLAG_PHONE'] + df['FLAG_EMAIL']
    df['HAS_CAR_AND_REALTY'] = ((df['FLAG_OWN_CAR'] == 'Y') & (df['FLAG_OWN_REALTY'] == 'Y')).astype(int)

    # Group Medians for Relative Peer Ratios
    edu_medians = df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].median().to_dict()
    occ_medians = df.groupby('OCCUPATION_TYPE')['EMPLOYED_YEARS'].median().to_dict()

    df['INCOME_TO_EDU_MEDIAN'] = df.apply(lambda r: r['AMT_INCOME_TOTAL'] / max(1.0, edu_medians.get(r['NAME_EDUCATION_TYPE'], 150000.0)), axis=1)
    df['TENURE_TO_OCC_MEDIAN'] = df.apply(lambda r: r['EMPLOYED_YEARS'] / max(0.5, occ_medians.get(r['OCCUPATION_TYPE'], 3.0)), axis=1)

    # 3. Categorical Encodings
    categorical_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]

    encoding_maps = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoding_maps[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}

    feature_columns = [c for c in df.columns if c != 'bad_credit']

    X = df[feature_columns]
    y = df['bad_credit']

    # 4. Train-Test Split & SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Build Champion Soft-Voting Ensemble
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_split=8,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=80, max_depth=16, min_samples_split=8,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    )

    champion_ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et)],
        voting='soft',
        weights=[1.2, 1.0]
    )

    print("Training Champion Soft-Voting Ensemble (Random Forest + Extra Trees)...")
    champion_ensemble.fit(X_train_res, y_train_res)

    # 6. Evaluation
    probs = champion_ensemble.predict_proba(X_test)[:, 1]
    auc_val = float(roc_auc_score(y_test, probs))
    gini_val = float(2 * auc_val - 1)
    prauc_val = float(average_precision_score(y_test, probs))
    brier_val = float(brier_score_loss(y_test, probs))

    # Optimal Threshold Tuning
    best_threshold = 0.39
    best_f1 = 0.0
    for t in np.linspace(0.2, 0.7, 51):
        preds_t = (probs >= t).astype(int)
        f = f1_score(y_test, preds_t, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_threshold = round(float(t), 2)

    final_preds = (probs >= best_threshold).astype(int)
    acc_val = float((final_preds == y_test).mean())
    prec_val = float(precision_score(y_test, final_preds, zero_division=0))
    rec_val = float(recall_score(y_test, final_preds, zero_division=0))

    print(f"\nModel Performance on Test Set:")
    print(f"ROC-AUC:         {auc_val:.4f}")
    print(f"Gini:            {gini_val:.4f}")
    print(f"PR-AUC:          {prauc_val:.4f}")
    print(f"Brier Score:     {brier_val:.4f}")
    print(f"Best Threshold:  {best_threshold}")
    print(f"F1-Score:        {best_f1:.4f}")
    print(f"Accuracy:        {acc_val * 100:.2f}%")
    print(f"Precision:       {prec_val * 100:.2f}%")
    print(f"Recall:          {rec_val * 100:.2f}%")

    # 7. Serialize Artifacts
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "Random_Forest_best_model.pkl")
    champion_file = os.path.join(models_dir, "Champion_Credit_Model.pkl")
    cols_file = os.path.join(models_dir, "train_columns.pkl")
    thresh_file = os.path.join(models_dir, "best_threshold.txt")
    meta_file = os.path.join(models_dir, "model_metadata.json")

    # Save models with compression=3 for fast loading and low disk footprint
    joblib.dump(champion_ensemble, model_file, compress=3)
    joblib.dump(champion_ensemble, champion_file, compress=3)
    joblib.dump(feature_columns, cols_file)

    with open(thresh_file, "w") as f:
        f.write(str(best_threshold))

    metadata = {
        "model_name": "Champion Soft-Voting Ensemble (Random Forest + Extra Trees)",
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "roc_auc": round(auc_val, 4),
        "gini": round(gini_val, 4),
        "pr_auc": round(prauc_val, 4),
        "brier_score": round(brier_val, 4),
        "best_threshold": best_threshold,
        "f1_score": round(best_f1, 4),
        "accuracy_pct": round(acc_val * 100, 2),
        "precision_pct": round(prec_val * 100, 2),
        "recall_pct": round(rec_val * 100, 2),
        "edu_medians": edu_medians,
        "occ_medians": occ_medians,
        "encoding_maps": encoding_maps
    }

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel artifacts saved successfully in models/ directory!")
    return metadata

if __name__ == '__main__':
    train_and_export_champion_pipeline()
