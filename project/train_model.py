"""
Cardiovascular Disease Prediction – Model Training
====================================================
Trains BOTH Logistic Regression AND Random Forest, evaluates them,
saves all artifacts so the Streamlit app can switch between models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Load & clean data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "cardio_train.csv")
df = pd.read_csv(DATA_PATH, sep=";")

print(f"Dataset shape: {df.shape}")

df.drop(columns=["id"], inplace=True)
df["age"] = (df["age"] / 365).round(1)

df = df[(df["ap_hi"] >= 60) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
df = df[df["ap_hi"] >= df["ap_lo"]]

for col in ["height", "weight"]:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df = df[(df[col] >= lo) & (df[col] <= hi)]

df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)

print(f"Cleaned dataset shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. Features & target
# ─────────────────────────────────────────────
TARGET   = "cardio"
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────────
# 3. Split & scale
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. Define models
# ─────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# ─────────────────────────────────────────────
# 5. Directories
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# Save shared artifacts
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

all_metrics = {}
roc_data    = {}   # store fpr/tpr for comparison plot

# ─────────────────────────────────────────────
# 6. Train, evaluate & save each model
# ─────────────────────────────────────────────
for name, clf in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(clf, X_train_sc, y_train, cv=5, scoring="accuracy")

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Feature importances (LR uses |coef|, RF uses feature_importances_)
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
    else:
        fi = np.abs(clf.coef_[0])
        fi = fi / fi.sum()   # normalise so they sum to 1

    metrics = {
        "accuracy":  round(acc, 4),
        "roc_auc":   round(auc, 4),
        "cv_mean":   round(float(cv_scores.mean()), 4),
        "cv_std":    round(float(cv_scores.std()),  4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "n_features": len(FEATURES),
        "feature_importances": dict(zip(FEATURES, fi.round(4).tolist())),
    }
    all_metrics[name] = metrics

    # Slug for filenames
    slug = name.lower().replace(" ", "_")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"confusion_matrix_{slug}.png"), dpi=120)
    plt.close(fig)

    # ── ROC curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6c63ff", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve – {name}", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"roc_curve_{slug}.png"), dpi=120)
    plt.close(fig)

    # ── Feature importance ──
    fi_df = (
        pd.DataFrame({"feature": FEATURES, "importance": fi})
        .sort_values("importance", ascending=True)
    )
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["feature"], fi_df["importance"], color=colors)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance – {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"feature_importance_{slug}.png"), dpi=120)
    plt.close(fig)

    # ── Save model ──
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{slug}_model.pkl"))
    joblib.dump(metrics, os.path.join(MODEL_DIR, f"metrics_{slug}.pkl"))
    print(f"  ✅ Saved {slug}_model.pkl")

# ─────────────────────────────────────────────
# 7. Combined ROC comparison plot
# ─────────────────────────────────────────────
palette = {"Logistic Regression": "#f59e0b", "Random Forest": "#6c63ff"}
fig, ax = plt.subplots(figsize=(7, 5))
for name, (fpr, tpr) in roc_data.items():
    auc_val = all_metrics[name]["roc_auc"]
    ax.plot(fpr, tpr, color=palette[name], lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Comparison: LR vs Random Forest", fontsize=13)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "roc_comparison.png"), dpi=120)
plt.close(fig)
print("\n✅ Saved combined ROC comparison plot.")

# ─────────────────────────────────────────────
# 8. Save combined metrics summary
# ─────────────────────────────────────────────
joblib.dump(all_metrics, os.path.join(MODEL_DIR, "all_metrics.pkl"))

print("\n" + "="*55)
print("  SUMMARY")
print("="*55)
for name, m in all_metrics.items():
    print(f"  {name:25s}  Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}")

print("\n✅ All done! Run the app with:")
print("   streamlit run app.py")
