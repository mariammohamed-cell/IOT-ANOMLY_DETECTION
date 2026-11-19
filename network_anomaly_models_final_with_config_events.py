#!/usr/bin/env python3
"""
TON_IoT Network model.
Trains eleven binary classifiers (eight baseline models plus XGBoost, LightGBM, CatBoost) and four anomaly-scoring methods.
Exports:
  export/network_predictions.csv
  export/network_config.json
  export/network_events.csv (optional post-merge events file)

Window policy: 100 consecutive source CSV rows per window. The CSV has no
timestamp, so these are row-count windows, NOT time windows.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import LedoitWolf

RANDOM_STATE = 42
TEST_SIZE = 0.20
WINDOW_SIZE = 100
THRESHOLD = 0.50

EXCLUDED = {
    "src_ip", "dst_ip", "dns_query", "http_uri", "http_user_agent",
    "ssl_subject", "ssl_issuer", "http_orig_mime_types",
    "http_resp_mime_types", "label", "type"
}
OTHER_TARGET_NAMES = {"attack", "attack_type", "class"}


def make_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=10,
            max_categories=50, sparse_output=True, dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=10,
            max_categories=50, sparse=True, dtype=np.float32
        )


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "label" not in df:
        raise ValueError("CSV must contain a 'label' column.")
    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("'label' must contain only 0 (Normal) and 1 (Attack).")
    y = y.astype("int8")

    excluded = {c for c in df.columns
                if c.lower() in EXCLUDED | OTHER_TARGET_NAMES}
    X = df.drop(columns=list(excluded), errors="ignore").copy()

    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].replace({"-": np.nan, "": np.nan, " ": np.nan})

    for c in X.select_dtypes(include=["object", "category"]).columns.tolist():
        converted = pd.to_numeric(X[c], errors="coerce")
        count = X[c].notna().sum()
        if count and converted.notna().sum() / count >= 0.95:
            X[c] = converted

    constant = [c for c in X if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant)
    if X.empty:
        raise ValueError("No usable features remain.")

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    if not numeric:
        raise ValueError("At least one numeric feature is required.")
    return df, X, y, numeric, categorical


def build_preprocessor(numeric, categorical):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler(with_centering=False))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_encoder())
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop", sparse_threshold=1.0)


def safe_metrics(y_true, pred, prob):
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_attack": float(precision_score(y_true, pred, zero_division=0)),
        "recall_attack": float(recall_score(y_true, pred, zero_division=0)),
        "f1_attack": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, prob))
        result["pr_auc"] = float(average_precision_score(y_true, prob))
    return result


def train_lstm_classifier(train_matrix, test_matrix, y_train, y_test):
    """
    LSTM baseline on flow feature vectors.
    The current CSV has no timestamp or sequence/device key, so each flow is
    treated as a sequence of length 1. This is an LSTM classifier, but it
    cannot learn temporal dependencies until real ordered windows are supplied.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dense = train_matrix.toarray() if hasattr(train_matrix, "toarray") else np.asarray(train_matrix)
    test_dense = test_matrix.toarray() if hasattr(test_matrix, "toarray") else np.asarray(test_matrix)
    train_dense = np.asarray(train_dense, dtype=np.float32)
    test_dense = np.asarray(test_dense, dtype=np.float32)

    # Keep the LSTM input size manageable and avoid non-finite values.
    train_dense = np.nan_to_num(train_dense, nan=0.0, posinf=0.0, neginf=0.0)
    test_dense = np.nan_to_num(test_dense, nan=0.0, posinf=0.0, neginf=0.0)

    class FlowLSTM(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=64,
                num_layers=1, batch_first=True
            )
            self.dropout = nn.Dropout(0.2)
            self.output = nn.Linear(64, 1)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            last_step = sequence[:, -1, :]
            return self.output(self.dropout(last_step)).squeeze(1)

    model = FlowLSTM(train_dense.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(y_train == 0).sum() / max(1, (y_train == 1).sum())],
            dtype=torch.float32, device=device
        )
    )

    x_tensor = torch.tensor(train_dense, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor), batch_size=512, shuffle=True
    )

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(
            f"LSTM epoch {epoch + 1}/{epochs} - "
            f"loss: {total_loss / len(y_tensor):.6f}",
            flush=True
        )

    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(test_dense), 4096):
            xb = torch.tensor(
                test_dense[start:start + 4096], dtype=torch.float32
            ).unsqueeze(1).to(device)
            probabilities.append(torch.sigmoid(model(xb)).cpu().numpy())

    return np.concatenate(probabilities), model, epochs


def train_classifiers(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    prep = build_preprocessor(
        X_train.attrs["numeric_cols"], X_train.attrs["categorical_cols"]
    )
    train_enc = prep.fit_transform(X_train)
    test_enc = prep.transform(X_test)

    models = {
        # CART is the Decision Tree baseline requested in the reference.
        "CART": DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE
        ),
        "LDA": LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "NaiveBayes": GaussianNB(),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, max_depth=8, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=float((y_train == 0).sum()) /
                             max(1, float((y_train == 1).sum())),
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350, learning_rate=0.08, num_leaves=31,
            class_weight="balanced", verbosity=-1,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350, depth=8, learning_rate=0.08,
            loss_function="Logloss", auto_class_weights="Balanced",
            verbose=False, random_seed=RANDOM_STATE,
            allow_writing_files=False, thread_count=-1
        )
    }

    rows, predictions = [], {}
    for name, model in models.items():
        print(f"\nTraining {name} ...", flush=True)
        a, b = train_enc, test_enc

        # These estimators require dense arrays.
        if name in {"CatBoost", "LDA", "NaiveBayes"}:
            a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
            b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)

        # RBF-SVM and KNN can be slow on the full 168k training rows.
        # Fit them on a reproducible, stratified subset while evaluating on all test rows.
        if name in {"SVM_RBF", "KNN"} and len(y_train) > 30000:
            from sklearn.model_selection import train_test_split as _split
            all_positions = np.arange(len(y_train))
            selected, _ = _split(
                all_positions, train_size=30000,
                random_state=RANDOM_STATE, stratify=y_train
            )
            a = a[selected]
            y_fit = y_train.iloc[selected] if hasattr(y_train, "iloc") else y_train[selected]
            print(f"{name}: fitting on stratified 30,000-row subset.", flush=True)
        else:
            y_fit = y_train

        model.fit(a, y_fit)
        prob = model.predict_proba(b)[:, 1]
        pred = (prob >= THRESHOLD).astype(int)
        metrics = safe_metrics(y_test, pred, prob)
        rows.append({"model": name, **metrics})
        predictions[name] = {"prob": prob, "pred": pred}
        print(f"{name} test F1: {metrics['f1_attack']:.4f}", flush=True)

    # LSTM is included as an additional baseline. PyTorch is required.
    print("\nTraining LSTM ...", flush=True)
    lstm_prob, lstm_model, lstm_epochs = train_lstm_classifier(
        train_enc, test_enc, y_train, y_test
    )
    lstm_pred = (lstm_prob >= THRESHOLD).astype(int)
    lstm_metrics = safe_metrics(y_test, lstm_pred, lstm_prob)
    rows.append({"model": "LSTM", **lstm_metrics})
    predictions["LSTM"] = {"prob": lstm_prob, "pred": lstm_pred}
    print(f"LSTM test F1: {lstm_metrics['f1_attack']:.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values("f1_attack", ascending=False)
    best_name = str(results.iloc[0]["model"])
    return results, predictions, best_name

def train_anomaly_methods(X, y, X_train, y_train, numeric_cols, test_idx):
    print("\nTraining anomaly methods on NORMAL training rows only ...", flush=True)
    normal = X_train.loc[y_train == 0, numeric_cols]
    all_numeric = X[numeric_cols]

    imputer = SimpleImputer(strategy="median")
    normal_imp = imputer.fit_transform(normal)
    all_imp = imputer.transform(all_numeric)

    scaler = RobustScaler()
    normal_scaled = scaler.fit_transform(normal_imp)
    all_scaled = scaler.transform(all_imp)

    scores = {}
    means = normal_imp.mean(axis=0)
    stds = normal_imp.std(axis=0)
    stds[stds == 0] = 1.0
    scores["Z-score"] = np.max(np.abs((all_imp - means) / stds), axis=1)

    maha = LedoitWolf().fit(normal_scaled)
    scores["Mahalanobis Distance"] = maha.mahalanobis(all_scaled)

    iso = IsolationForest(
        n_estimators=300, contamination="auto",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iso.fit(normal_scaled)
    scores["Isolation Forest"] = -iso.score_samples(all_scaled)

    try:
        import torch
        from torch import nn
        torch.manual_seed(RANDOM_STATE)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_dim = normal_scaled.shape[1]
        hidden = max(8, min(64, input_dim * 2))
        bottleneck = max(4, hidden // 2)

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, bottleneck), nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck, hidden), nn.ReLU(),
                    nn.Linear(hidden, input_dim)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))

        ae = Autoencoder().to(device)
        optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        tensor = torch.tensor(normal_scaled, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            tensor, batch_size=512, shuffle=True
        )
        epochs = 20
        ae.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = loss_fn(ae(batch), batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            print(f"Autoencoder epoch {epoch+1}/{epochs} "
                  f"- loss: {total / len(tensor):.6f}", flush=True)

        ae.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(all_scaled), 4096):
                batch = torch.tensor(
                    all_scaled[start:start+4096], dtype=torch.float32
                ).to(device)
                errors.append(torch.mean((ae(batch)-batch)**2, dim=1).cpu().numpy())
        scores["Autoencoder"] = np.concatenate(errors)
    except ImportError:
        raise RuntimeError(
            "PyTorch is required for the requested Autoencoder. "
            "Install it with: pip install torch"
        )

    test_idx = np.asarray(test_idx)
    y_test = y.iloc[test_idx].to_numpy()
    method_rows = []
    for name, values in scores.items():
        test_scores = np.asarray(values)[test_idx]
        method_rows.append({
            "method": name,
            "pr_auc": float(average_precision_score(y_test, test_scores)),
            "roc_auc": float(roc_auc_score(y_test, test_scores))
        })
    method_results = pd.DataFrame(method_rows).sort_values(
        "pr_auc", ascending=False
    )
    best_method = str(method_results.iloc[0]["method"])
    return method_results, best_method, scores


def make_window_predictions(df, y, test_idx, best_name, best_prediction):
    # Use the original row position to define fixed-size consecutive-flow windows.
    rows = pd.DataFrame({
        "row_index": np.asarray(test_idx),
        "p_attack": np.asarray(best_prediction["prob"]),
        "y_true": y.iloc[test_idx].to_numpy()
    })
    rows["chunk"] = 0
    rows["win"] = (rows["row_index"] // WINDOW_SIZE).astype(int)
    rows["split"] = "test"

    if "type" in df.columns:
        rows["true_type"] = df.loc[rows["row_index"], "type"].fillna("unknown").astype(str).to_numpy()
    else:
        rows["true_type"] = np.where(rows["y_true"] == 1, "Attack", "Normal")

    # Aggregate flow-level results into one record per window and split.
    out = rows.groupby(["chunk", "win", "split"], sort=True).agg(
        p_attack=("p_attack", "mean"),
        y_true=("y_true", lambda s: int(s.mode().iloc[0])),
        true_type=("true_type", lambda s: str(s.mode().iloc[0])),
        n_records=("row_index", "count")
    ).reset_index()
    out["pred_type"] = np.where(
        out["p_attack"] >= THRESHOLD, "Attack", "Normal"
    )
    out.insert(0, "source", "TON_IoT_Network")
    out = out[[
        "source", "chunk", "win", "split", "p_attack",
        "y_true", "true_type", "pred_type", "n_records"
    ]]
    return out


def make_window_events(df, y, test_idx, best_prediction, anomaly_scores, best_anomaly):
    """
    Optional post-merge event export.
    Uses attack_prob (not p_attack), plus anomaly_score.
    Scores are percentile-normalized over all flows for comparability.
    """
    raw_score = np.asarray(anomaly_scores[best_anomaly], dtype=float)
    ranks = pd.Series(raw_score).rank(pct=True).to_numpy()

    rows = pd.DataFrame({
        "row_index": np.asarray(test_idx),
        "attack_prob": np.asarray(best_prediction["prob"]),
        "anomaly_score": ranks[np.asarray(test_idx)],
        "y_true": y.iloc[test_idx].to_numpy()
    })
    rows["chunk"] = 0
    rows["win"] = (rows["row_index"] // WINDOW_SIZE).astype(int)
    rows["split"] = "test"

    if "type" in df.columns:
        rows["true_type"] = df.loc[rows["row_index"], "type"].fillna("unknown").astype(str).to_numpy()
    else:
        rows["true_type"] = np.where(rows["y_true"] == 1, "Attack", "Normal")

    out = rows.groupby(["chunk", "win", "split"], sort=True).agg(
        attack_prob=("attack_prob", "mean"),
        anomaly_score=("anomaly_score", "mean"),
        y_true=("y_true", lambda values: int(values.mode().iloc[0])),
        true_type=("true_type", lambda values: str(values.mode().iloc[0])),
        n_records=("row_index", "count")
    ).reset_index()
    out["pred_type"] = np.where(out["attack_prob"] >= THRESHOLD, "Attack", "Normal")
    out.insert(0, "source", "TON_IoT_Network")
    return out[[
        "source", "chunk", "win", "split", "attack_prob",
        "anomaly_score", "y_true", "true_type", "pred_type", "n_records"
    ]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input network CSV path")
    parser.add_argument("--out", default="export", help="Export directory")
    args = parser.parse_args()

    df, X, y, numeric, categorical = load_data(args.data)
    print(f"Rows: {len(df):,} | Features: {X.shape[1]}")
    print("Labels:", y.value_counts().sort_index().to_dict())

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train.attrs["numeric_cols"] = numeric
    X_train.attrs["categorical_cols"] = categorical
    test_idx = X_test.index.to_numpy()

    classifier_results, pred_dict, best_model = train_classifiers(
        X_train, X_test, y_train, y_test
    )
    anomaly_results, best_anomaly, anomaly_scores = train_anomaly_methods(
        X, y, X_train, y_train, numeric, test_idx
    )

    best_f1 = float(
        classifier_results.loc[
            classifier_results["model"] == best_model, "f1_attack"
        ].iloc[0]
    )
    exported = make_window_predictions(
        df, y, test_idx, best_model, pred_dict[best_model]
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exactly the first two requested deliverables.
    predictions_path = out_dir / "network_predictions.csv"
    config_path = out_dir / "network_config.json"
    events_path = out_dir / "network_events.csv"
    exported.to_csv(predictions_path, index=False)
    events = make_window_events(
        df, y, test_idx, pred_dict[best_model], anomaly_scores, best_anomaly
    )
    events.to_csv(events_path, index=False)

    config = {
        "source": "TON_IoT_Network",
        "binary_model": best_model,
        "multiclass_model": None,
        "n_records": int(len(df)),
        "features": list(X.columns),
        "anomaly_method": best_anomaly,
        "calibrated": False,
        "threshold": THRESHOLD,
        "epoch": 20,
        "window_seconds": None,
        "classes": ["Normal", "Attack"],
        "classification_models": ["LogisticRegression", "LDA", "KNN", "CART", "RandomForest", "NaiveBayes", "SVM_RBF", "LSTM", "XGBoost", "LightGBM", "CatBoost"],
        "lstm_sequence_note": "The CSV has no timestamp/device sequence key; current LSTM uses sequence_length=1 per flow and does not learn temporal dependencies.",
        "test_f1": best_f1,
        "val_f1": None,
        "split_rule": (
            "Stratified random 80/20 split at flow level; predictions are "
            "aggregated by source row order into windows of 100 flows. "
            "The CSV has no timestamp, so windows are not time-based."
        ),
        "block_windows": WINDOW_SIZE
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nClassifier comparison:")
    print(classifier_results.to_string(index=False))
    print("\nAnomaly method comparison:")
    print(anomaly_results.to_string(index=False))
    print("\nDONE. Deliverables created:")
    print(predictions_path.resolve())
    print(config_path.resolve())
    print(events_path.resolve())


if __name__ == "__main__":
    main()
