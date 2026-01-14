import os
import joblib
import numpy as np
import pandas as pd


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1) SETTINGS
# =========================
CSV_PATH = r"C:\ML_assigment\merged.csv"

FEATURES = ["Country", "Time of Day", "Season"]
TARGET = "Weather"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Outputs
OUT_DIR = r"C:\ML_assigment\results"
os.makedirs(OUT_DIR, exist_ok=True)

MISCLASSIFIED_CSV = os.path.join(OUT_DIR, "misclassified_examples.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "model_summary.csv")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.joblib")

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# 2) LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns:\n{list(df.columns)}"
    )

keep_cols = (["id"] if "id" in df.columns else []) + FEATURES + [TARGET]
data = df[keep_cols].dropna().copy()

X = data[FEATURES]
y = data[TARGET]

print("Dataset size:", len(data))
print("\nClass distribution (Weather):")
print(y.value_counts())


# =========================
# 3) SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Keep test rows for analysis
test_rows = data.loc[X_test.index].copy()


# =========================
# 4) PREPROCESSING
# =========================
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES),
    ],
    remainder="drop"
)

scaler = StandardScaler(with_mean=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# =========================
# 5) BASELINE MODELS: KNN (k=1 and k=3)
# =========================
knn_baselines = {
    "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
}

baseline_results = {}

print("\n" + "=" * 80)
print("BASELINE MODELS: KNN (k=1 and k=3)")

for name, knn_model in knn_baselines.items():
    print("\n" + "-" * 60)
    print(name)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("scale", scaler),
        ("model", knn_model)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")

    baseline_results[name] = {
        "accuracy": acc,
        "f1_weighted": f1w,
        "model": model
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1w:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nKNN Baseline Comparison:")
for k, v in baseline_results.items():
    print(f"{k}: Accuracy={v['accuracy']:.4f}, F1={v['f1_weighted']:.4f}")

# Choose best KNN baseline by weighted F1 for fair comparison
best_knn_name = max(baseline_results, key=lambda k: baseline_results[k]["f1_weighted"])
baseline_acc = baseline_results[best_knn_name]["accuracy"]
baseline_f1w = baseline_results[best_knn_name]["f1_weighted"]

print("\nBest baseline selected for comparison:", best_knn_name)
print(f"Baseline Accuracy: {baseline_acc:.4f}")
print(f"Baseline F1 (weighted): {baseline_f1w:.4f}")


# =========================
# 6) TWO MODELS + HYPERPARAM TUNING
# =========================
models = {
    "SVM (RBF/Gaussian, balanced)": (
        Pipeline(steps=[
            ("preprocess", preprocess),
            ("scale", scaler),
            ("model", SVC(kernel="rbf", class_weight="balanced"))
        ]),
        {
            "model__C": [0.1, 1, 10, 50, 100],  # >=4 values
            "model__gamma": ["scale", "auto"],
        }
    ),

    "Random Forest (balanced)": (
        Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
        ]),
        {
            "model__n_estimators": [100, 200, 400, 700],  # >=4 values
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
    )
}

summary_rows = []
best_model = None
best_name = None
best_cv_f1 = -1

for name, (pipe, grid) in models.items():
    print("\n" + "=" * 80)
    print("MODEL:", name)
    print("Tuning hyperparameters:", {k: v for k, v in grid.items()})

    gs = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    model = gs.best_estimator_
    cv_f1 = gs.best_score_

    print("Best params:", gs.best_params_)
    print(f"Best CV F1 (weighted): {cv_f1:.4f}")

    # Evaluate on test
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1w = f1_score(y_test, pred, average="weighted")

    print("\nTEST RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1w:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))

    # Compare to best KNN baseline
    print("\nCompared to baseline (best KNN):", best_knn_name)
    print(f"Δ Accuracy = {acc - baseline_acc:+.4f}")
    print(f"Δ F1 (weighted) = {f1w - baseline_f1w:+.4f}")

    summary_rows.append({
        "model": name,
        "best_cv_f1_weighted": cv_f1,
        "test_accuracy": acc,
        "test_f1_weighted": f1w,
        "best_params": str(gs.best_params_),
        "baseline_used": best_knn_name,
        "baseline_accuracy": baseline_acc,
        "baseline_f1_weighted": baseline_f1w,
        "delta_acc_vs_baseline": acc - baseline_acc,
        "delta_f1w_vs_baseline": f1w - baseline_f1w,
    })

    # Track best by CV F1
    if cv_f1 > best_cv_f1:
        best_cv_f1 = cv_f1
        best_model = model
        best_name = name


# Save summary
summary_df = pd.DataFrame(summary_rows).sort_values("best_cv_f1_weighted", ascending=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("\n" + "=" * 80)
print("Saved model summary to:", SUMMARY_CSV)
print(summary_df.to_string(index=False))

# Save best model
joblib.dump(best_model, BEST_MODEL_PATH)
print("\n Best model (by CV F1):", best_name)
print(" Saved best model to:", BEST_MODEL_PATH)


# =========================
# 7) PERFORMANCE ANALYSIS (ERROR ANALYSIS) - SECTION 5
# =========================
print("\n" + "=" * 80)
print("PERFORMANCE ANALYSIS FOR BEST MODEL:", best_name)

best_pred = best_model.predict(X_test)

analysis_df = test_rows.copy()
analysis_df["y_true"] = y_test
analysis_df["y_pred"] = best_pred
analysis_df["is_correct"] = (analysis_df["y_true"] == analysis_df["y_pred"])

mis_df = analysis_df[~analysis_df["is_correct"]].copy()
mis_df.to_csv(MISCLASSIFIED_CSV, index=False)

print("Saved misclassified examples to:", MISCLASSIFIED_CSV)
print("Number of misclassifications:", len(mis_df), "out of", len(analysis_df))

# Error rate by class
print("\nError rate by TRUE label:")
acc_by_label = analysis_df.groupby("y_true")["is_correct"].mean().sort_values()
for label, acc_label in acc_by_label.items():
    count = int((analysis_df["y_true"] == label).sum())
    print(f"{label:>10}: accuracy={acc_label:.3f} | error_rate={1-acc_label:.3f} | count={count}")

# Confusion matrix + top confusions
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

print("\nConfusion matrix (best model):")
print(cm_df)

conf_pairs = []
for i, t in enumerate(labels):
    for j, p in enumerate(labels):
        if i != j and cm[i, j] > 0:
            conf_pairs.append((cm[i, j], t, p))
conf_pairs.sort(reverse=True)

print("\nTop misclassification pairs (count, true | predicted):")
for count, t, p in conf_pairs[:10]:
    print(f"{count:>3}  {t} | {p}")

# Patterns by feature values
def show_error_patterns(col, top_n=10):
    tmp = analysis_df.groupby(col)["is_correct"].agg(["count", "mean"]).reset_index()
    tmp["error_rate"] = 1 - tmp["mean"]
    tmp = tmp.sort_values(["error_rate", "count"], ascending=[False, False]).head(top_n)
    print(f"\nTop {top_n} hardest {col} values (by error rate):")
    for _, r in tmp.iterrows():
        print(f"{str(r[col])[:25]:>25} | count={int(r['count']):>4} | error_rate={r['error_rate']:.3f}")

show_error_patterns("Country", top_n=10)
show_error_patterns("Time of Day", top_n=10)
show_error_patterns("Season", top_n=10)

print("\nDone.")


# =========================
# 8) PLOTS (RESULT VISUALIZATION) - SAVE ONLY (NO GUI)
# =========================

# Plot 1: Class distribution
plt.figure()
y.value_counts().plot(kind="bar")
plt.title("Class Distribution (Weather)")
plt.xlabel("Weather Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=200)
plt.close()

# Plot 2: Model comparison (accuracy and F1)
rows_for_plot = []

# KNN baselines
for name, vals in baseline_results.items():
    rows_for_plot.append({
        "model": name,
        "test_accuracy": vals["accuracy"],
        "test_f1_weighted": vals["f1_weighted"]
    })

# Tuned models
for _, r in summary_df.iterrows():
    rows_for_plot.append({
        "model": r["model"],
        "test_accuracy": r["test_accuracy"],
        "test_f1_weighted": r["test_f1_weighted"]
    })

plot_df = pd.DataFrame(rows_for_plot)

plt.figure()
plt.bar(plot_df["model"], plot_df["test_accuracy"])
plt.title("Test Accuracy by Model")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "test_accuracy_by_model.png"), dpi=200)
plt.close()

plt.figure()
plt.bar(plot_df["model"], plot_df["test_f1_weighted"])
plt.title("Test Weighted F1 by Model")
plt.xlabel("Model")
plt.ylabel("Weighted F1")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "test_f1_by_model.png"), dpi=200)
plt.close()

# Plot 3: Confusion matrix for best model
plt.figure()
plt.imshow(cm_df.values)
plt.title(f"Confusion Matrix (Best Model: {best_name})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=30, ha="right")
plt.yticks(ticks=np.arange(len(labels)), labels=labels)

for i in range(cm_df.shape[0]):
    for j in range(cm_df.shape[1]):
        plt.text(j, i, int(cm_df.values[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_best_model.png"), dpi=200)
plt.close()

# Plot 4: Error rate by true label
err_stats = analysis_df.groupby("y_true")["is_correct"].agg(["count", "mean"]).reset_index()
err_stats["error_rate"] = 1 - err_stats["mean"]
err_stats = err_stats.sort_values("error_rate", ascending=False)

plt.figure()
plt.bar(err_stats["y_true"], err_stats["error_rate"])
plt.title(f"Error Rate by True Class (Best Model: {best_name})")
plt.xlabel("True Weather Class")
plt.ylabel("Error Rate")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "error_rate_by_class.png"), dpi=200)
plt.close()

print("\n Plots saved in:", PLOTS_DIR)
