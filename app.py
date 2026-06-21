"""
DNA Sequence Analysis — 3-Model Pipeline
=========================================
Models:
  1. Organism Classification    (Bacteria / Dog / Human)
  2. Mutation Detection         (Mutated / Not Mutated)
  3. Disease Severity           (Low / Medium / High)

Dataset : dna_clean_balanced_final_v6.csv
Algorithm: Random Forest (scikit-learn)

Usage:+++++
  # Train models + launch interactive prediction:
  python dna_models_pipeline.py --data path/to/dataset.csv --interactive

  # Train models only (no prediction):
  python dna_models_pipeline.py --data path/to/dataset.csv
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = "dna_clean_balanced_final_v6.csv"
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
N_ESTIMATORS      = 200
FEATURES          = [
    'Sequence_Length', 'GC_Content', 'AT_Content',
    'Num_A', 'Num_T', 'Num_C', 'Num_G',
    'kmer_3_freq', 'GC_deviation',
    'Gene_Family_enc', 'Organism_enc'
]

# Colour palette
C_BLUE   = "#185FA5"
C_GREEN  = "#3B6D11"
C_AMBER  = "#854F0B"
C_RED    = "#A32D2D"
C_TEAL   = "#0F6E56"
C_CORAL  = "#993C1D"
C_PURPLE = "#534AB7"
C_LIGHT  = "#F7F6F3"
C_DARK   = "#2C2C2A"

MODEL_COLORS = {
    "Organism": C_BLUE,
    "Mutation": C_GREEN,
    "Severity": C_AMBER
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.6,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#FAFAF8',
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare(path: str) -> tuple:
    """Load CSV, encode categoricals, add derived features."""
    print(f"\n{'='*60}")
    print("  LOADING DATASET")
    print(f"{'='*60}")
    df = pd.read_csv(path)
    print(f"  Rows : {len(df):,}")
    print(f"  Cols : {df.shape[1]}")
    print(f"\n  Organism distribution:")
    print(df['Organism'].value_counts().to_string())
    print(f"\n  Mutation balance:")
    print(df.groupby(['Organism', 'Mutation_Flag']).size().to_string())
    print(f"\n  Disease Severity:")
    print(df['Disease_Severity'].value_counts().to_string())

    le_gene = LabelEncoder()
    le_org  = LabelEncoder()
    df['Gene_Family_enc'] = le_gene.fit_transform(df['Gene_Family'])
    df['Organism_enc']    = le_org.fit_transform(df['Organism'])
    df['GC_deviation']    = abs(df['GC_Content'] - 50)

    return df, le_org, le_gene


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X, y, class_weight=None) -> tuple:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return clf, X_tr, X_te, y_tr, y_te, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def styled_confusion_matrix(ax, cm, labels, title, color):
    """Draw a clean annotated confusion matrix heatmap."""
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#FFFFFF', color], N=256
    )
    n = len(labels)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual',    fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12, color=color)
    ax.grid(False)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else C_DARK,
                    fontweight='bold' if i == j else 'normal')

    # Highlight diagonal
    for i in range(n):
        rect = FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)


def plot_feature_importance(ax, importances, feature_names, title, color):
    """Horizontal bar chart of feature importances."""
    idx   = np.argsort(importances)
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    bars = ax.barh(names, vals, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, color=C_DARK)

    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.set_xlabel('Importance (Gini)', fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(0, max(vals) * 1.18)


def plot_roc_multiclass(ax, clf, X_te, y_te, classes, title, color):
    """Plot one-vs-rest ROC curves for multi-class models."""
    y_bin   = label_binarize(y_te, classes=sorted(set(y_te)))
    y_score = clf.predict_proba(X_te)
    n_cls   = len(classes)

    palette = [C_BLUE, C_TEAL, C_CORAL, C_PURPLE, C_AMBER][:n_cls]
    for i, (cls, col) in enumerate(zip(classes, palette)):
        if y_bin.shape[1] == 1:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False positive rate', fontsize=10)
    ax.set_ylabel('True positive rate',  fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.8)


def plot_class_metrics(ax, report_dict, classes, title, color):
    """Grouped bar chart: Precision / Recall / F1 per class."""
    x     = np.arange(len(classes))
    width = 0.25
    prec  = [report_dict[c]['precision'] * 100 for c in classes]
    rec   = [report_dict[c]['recall']    * 100 for c in classes]
    f1    = [report_dict[c]['f1-score']  * 100 for c in classes]

    ax.bar(x - width, prec, width, label='Precision', color=color,   alpha=0.9)
    ax.bar(x,          rec,  width, label='Recall',    color=C_TEAL,  alpha=0.9)
    ax.bar(x + width, f1,   width, label='F1',        color=C_PURPLE, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=7.5, color=C_DARK)


def accuracy_gauge(ax, accuracy, title, color):
    """Simple donut gauge showing accuracy."""
    sizes  = [accuracy, 100 - accuracy]
    colors = [color, '#ECECEC']
    wedges, _ = ax.pie(
        sizes, startangle=90, colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    ax.text(0, 0, f'{accuracy:.1f}%',
            ha='center', va='center', fontsize=20, fontweight='bold', color=color)
    ax.text(0, -0.55, title,
            ha='center', va='center', fontsize=11, color=C_DARK, fontweight='bold')
    ax.set_aspect('equal')


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL 1 — ORGANISM CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def run_organism_model(df, le_org):
    print(f"\n{'='*60}")
    print("  MODEL 1 — ORGANISM CLASSIFICATION")
    print(f"{'='*60}")

    feat_cols = [f for f in FEATURES if f != 'Organism_enc']
    X = df[feat_cols]
    y = df['Organism_enc']

    clf, X_tr, X_te, y_tr, y_te, y_pred = train_model(X, y)
    acc  = accuracy_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)
    rep  = classification_report(y_te, y_pred,
                                  target_names=le_org.classes_, output_dict=True)

    print(f"  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_te, y_pred, target_names=le_org.classes_))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Title banner
    fig.text(0.5, 0.97, 'Model 1 — Organism Classification',
             ha='center', va='top', fontsize=18, fontweight='bold', color=C_BLUE)
    fig.text(0.5, 0.935,
             f'Random Forest  ·  {N_ESTIMATORS} trees  ·  {int(TEST_SIZE*100)}% test set  ·  '
             f'Dataset: {len(df):,} sequences',
             ha='center', fontsize=11, color='#666')

    # Gauge
    ax0 = fig.add_subplot(gs[0, 0])
    accuracy_gauge(ax0, acc * 100, 'Overall Accuracy', C_BLUE)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 1])
    styled_confusion_matrix(ax1, cm, le_org.classes_,
                             'Confusion Matrix', C_BLUE)

    # Per-class metrics bar
    ax2 = fig.add_subplot(gs[0, 2])
    plot_class_metrics(ax2, rep, le_org.classes_,
                       'Precision / Recall / F1 per Class', C_BLUE)

    # ROC curves
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_roc_multiclass(ax3, clf, X_te, y_te, le_org.classes_,
                        'ROC Curves (One-vs-Rest)', C_BLUE)

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 2])
    plot_feature_importance(ax4, clf.feature_importances_, feat_cols,
                             'Feature Importance', C_BLUE)

    plt.savefig('model1_organism_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved → model1_organism_results.png")
    plt.show()

    return clf, X_te, y_te, y_pred, rep, acc


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL 2 — MUTATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def run_mutation_model(df):
    print(f"\n{'='*60}")
    print("  MODEL 2 — MUTATION DETECTION")
    print(f"{'='*60}")

    X = df[FEATURES]
    y = df['Mutation_Flag']
    mut_labels = ['No Mutation', 'Mutation']

    clf, X_tr, X_te, y_tr, y_te, y_pred = train_model(X, y, class_weight='balanced')
    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    rep = classification_report(y_te, y_pred,
                                 target_names=mut_labels, output_dict=True)

    print(f"  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_te, y_pred, target_names=mut_labels))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    fig.text(0.5, 0.97, 'Model 2 — Mutation Detection',
             ha='center', va='top', fontsize=18, fontweight='bold', color=C_GREEN)
    fig.text(0.5, 0.935,
             f'Random Forest  ·  Balanced class weights  ·  {int(TEST_SIZE*100)}% test set  ·  '
             f'Dataset: {len(df):,} sequences',
             ha='center', fontsize=11, color='#666')

    # Gauge
    ax0 = fig.add_subplot(gs[0, 0])
    accuracy_gauge(ax0, acc * 100, 'Overall Accuracy', C_GREEN)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 1])
    styled_confusion_matrix(ax1, cm, mut_labels, 'Confusion Matrix', C_GREEN)

    # Per-class metrics
    ax2 = fig.add_subplot(gs[0, 2])
    plot_class_metrics(ax2, rep, mut_labels,
                       'Precision / Recall / F1 per Class', C_GREEN)

    # ROC
    ax3 = fig.add_subplot(gs[1, 0:2])
    y_score = clf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color=C_GREEN, lw=2.5, label=f'AUC = {roc_auc:.3f}')
    ax3.fill_between(fpr, tpr, alpha=0.12, color=C_GREEN)
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.02])
    ax3.set_xlabel('False positive rate', fontsize=10)
    ax3.set_ylabel('True positive rate', fontsize=10)
    ax3.set_title('ROC Curve — Mutation Detection', fontsize=12,
                  fontweight='bold', color=C_GREEN, pad=10)
    ax3.legend(fontsize=10, framealpha=0.8)

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 2])
    plot_feature_importance(ax4, clf.feature_importances_, FEATURES,
                             'Feature Importance', C_GREEN)

    plt.savefig('model2_mutation_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved → model2_mutation_results.png")
    plt.show()

    return clf, X_te, y_te, y_pred, rep, acc


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL 3 — DISEASE SEVERITY
# ─────────────────────────────────────────────────────────────────────────────
def run_severity_model(df):
    print(f"\n{'='*60}")
    print("  MODEL 3 — DISEASE SEVERITY (Low / Medium / High)")
    print(f"{'='*60}")

    sev_labels = ['High', 'Low', 'Medium']
    X = df[FEATURES]
    y = df['Disease_Severity']

    clf, X_tr, X_te, y_tr, y_te, y_pred = train_model(X, y, class_weight='balanced')
    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred, labels=sev_labels)
    rep = classification_report(y_te, y_pred,
                                 labels=sev_labels,
                                 target_names=sev_labels, output_dict=True)

    print(f"  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_te, y_pred, labels=sev_labels, target_names=sev_labels))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    fig.text(0.5, 0.97, 'Model 3 — Disease Severity Prediction',
             ha='center', va='top', fontsize=18, fontweight='bold', color=C_AMBER)
    fig.text(0.5, 0.935,
             f'Random Forest  ·  Balanced class weights  ·  {int(TEST_SIZE*100)}% test set  ·  '
             f'Dataset: {len(df):,} sequences',
             ha='center', fontsize=11, color='#666')

    # Gauge
    ax0 = fig.add_subplot(gs[0, 0])
    accuracy_gauge(ax0, acc * 100, 'Overall Accuracy', C_AMBER)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 1])
    styled_confusion_matrix(ax1, cm, sev_labels, 'Confusion Matrix', C_AMBER)

    # Per-class metrics
    ax2 = fig.add_subplot(gs[0, 2])
    plot_class_metrics(ax2, rep, sev_labels,
                       'Precision / Recall / F1 per Class', C_AMBER)

    # ROC
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_roc_multiclass(ax3, clf, X_te, y_te, sev_labels,
                        'ROC Curves (One-vs-Rest)', C_AMBER)

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 2])
    plot_feature_importance(ax4, clf.feature_importances_, FEATURES,
                             'Feature Importance', C_AMBER)

    plt.savefig('model3_severity_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved → model3_severity_results.png")
    plt.show()

    return clf, X_te, y_te, y_pred, rep, acc


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMBINED SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(df, acc1, acc2, acc3,
                            rep1, rep2, rep3,
                            fi1, fi2, fi3, le_org):
    """One-page summary comparing all 3 models."""
    print(f"\n{'='*60}")
    print("  GENERATING COMBINED SUMMARY DASHBOARD")
    print(f"{'='*60}")

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    fig.text(0.5, 0.98,
             'DNA Sequence Analysis — All Models Summary',
             ha='center', va='top', fontsize=20, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.955,
             f'Dataset: {len(df):,} real DNA sequences  ·  '
             f'Random Forest {N_ESTIMATORS} trees  ·  '
             f'{int(TEST_SIZE*100)}% test set  ·  3 organisms × 2 mutation classes',
             ha='center', fontsize=11, color='#777')

    # ── Row 0: Accuracy gauges ────────────────────────────────────────────────
    for i, (acc, title, color) in enumerate([
        (acc1 * 100, 'Organism\nClassification', C_BLUE),
        (acc2 * 100, 'Mutation\nDetection',       C_GREEN),
        (acc3 * 100, 'Disease\nSeverity',          C_AMBER),
    ]):
        ax = fig.add_subplot(gs[0, i])
        accuracy_gauge(ax, acc, title, color)

    # ── Dataset composition (Row 0, col 3) ───────────────────────────────────
    ax_ds = fig.add_subplot(gs[0, 3])
    orgs  = df['Organism'].value_counts()
    bars  = ax_ds.bar(orgs.index, orgs.values,
                      color=[C_BLUE, C_TEAL, C_CORAL], edgecolor='white', linewidth=1.2)
    for bar in bars:
        ax_ds.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                   f'{bar.get_height():,}', ha='center', fontsize=9, color=C_DARK)
    ax_ds.set_title('Samples per organism', fontsize=11, fontweight='bold', color=C_DARK)
    ax_ds.set_ylabel('Count', fontsize=9)
    ax_ds.tick_params(axis='x', labelsize=9)

    # ── Row 1: F1 comparison across models ───────────────────────────────────
    ax_f1 = fig.add_subplot(gs[1, 0:2])
    model_labels = list(le_org.classes_)  # Bacteria, Dog, Human
    f1_vals = [rep1[c]['f1-score'] * 100 for c in model_labels]
    x = np.arange(len(model_labels))
    ax_f1.bar(x, f1_vals, color=[C_BLUE, C_TEAL, C_CORAL],
              edgecolor='white', linewidth=1)
    ax_f1.set_xticks(x); ax_f1.set_xticklabels(model_labels, fontsize=10)
    ax_f1.set_ylim(0, 110); ax_f1.set_ylabel('F1 Score (%)', fontsize=10)
    ax_f1.set_title('Organism Model — F1 per class', fontsize=11,
                    fontweight='bold', color=C_BLUE)
    for bar in ax_f1.patches:
        ax_f1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    # Mutation F1 comparison
    ax_mf = fig.add_subplot(gs[1, 2])
    mut_classes = ['No Mutation', 'Mutation']
    mut_f1 = [rep2[c]['f1-score'] * 100 for c in mut_classes]
    ax_mf.bar(mut_classes, mut_f1, color=[C_GREEN, C_RED],
              edgecolor='white', linewidth=1)
    ax_mf.set_ylim(0, 110); ax_mf.set_ylabel('F1 Score (%)', fontsize=10)
    ax_mf.set_title('Mutation — F1 per class', fontsize=11,
                    fontweight='bold', color=C_GREEN)
    for bar in ax_mf.patches:
        ax_mf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    # Severity F1 comparison
    ax_sf = fig.add_subplot(gs[1, 3])
    sev_classes = ['Low', 'High', 'Medium']
    sev_f1 = [rep3[c]['f1-score'] * 100 for c in sev_classes]
    sev_colors = [C_GREEN, C_RED, C_AMBER]
    ax_sf.bar(sev_classes, sev_f1, color=sev_colors,
              edgecolor='white', linewidth=1)
    ax_sf.set_ylim(0, 110); ax_sf.set_ylabel('F1 Score (%)', fontsize=10)
    ax_sf.set_title('Severity — F1 per class', fontsize=11,
                    fontweight='bold', color=C_AMBER)
    for bar in ax_sf.patches:
        ax_sf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    # ── Row 2: Feature importance comparison ─────────────────────────────────
    feat_labels_short = [
        'Seq len', 'GC%', 'AT%', 'Num A', 'Num T',
        'Num C', 'Num G', 'k-mer', 'GC dev', 'Gene fam', 'Organism'
    ]
    x = np.arange(len(feat_labels_short))
    w = 0.26

    ax_fi = fig.add_subplot(gs[2, 0:4])
    ax_fi.bar(x - w,   fi1, w, label='Organism', color=C_BLUE,   alpha=0.88, edgecolor='white')
    ax_fi.bar(x,       fi2, w, label='Mutation', color=C_GREEN,  alpha=0.88, edgecolor='white')
    ax_fi.bar(x + w,   fi3, w, label='Severity', color=C_AMBER,  alpha=0.88, edgecolor='white')
    ax_fi.set_xticks(x)
    ax_fi.set_xticklabels(feat_labels_short, fontsize=9, rotation=15, ha='right')
    ax_fi.set_ylabel('Importance (Gini)', fontsize=10)
    ax_fi.set_title('Feature Importance Comparison — All 3 Models',
                    fontsize=12, fontweight='bold', color=C_DARK)
    ax_fi.legend(fontsize=10, framealpha=0.8)

    plt.savefig('summary_dashboard.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved → summary_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. SEQUENCE FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_from_sequence(sequence: str, gene_family: str = 'Other',
                                    organism_hint: str = 'Unknown') -> dict:
    """
    Convert a raw DNA sequence string into a feature dict
    matching the training feature set.
    """
    seq = re.sub(r'[^ATGCNatgcn]', '', sequence).upper()
    if len(seq) < 10:
        raise ValueError(f"Sequence too short after cleaning: {len(seq)} bp (min 10 required)")

    n     = len(seq)
    num_a = seq.count('A')
    num_t = seq.count('T')
    num_g = seq.count('G')
    num_c = seq.count('C')

    gc_content = round((num_g + num_c) / n * 100, 3)
    at_content = round((num_a + num_t) / n * 100, 3)
    gc_dev     = round(abs(gc_content - 50), 3)

    kmers     = [seq[i:i+3] for i in range(n - 2)]
    from collections import Counter
    kmer_freq = round(max(Counter(kmers).values()) / len(kmers), 6) if kmers else 0.0

    return {
        'sequence':       seq,
        'Sequence_Length': n,
        'GC_Content':      gc_content,
        'AT_Content':      at_content,
        'Num_A':           num_a,
        'Num_T':           num_t,
        'Num_C':           num_c,
        'Num_G':           num_g,
        'kmer_3_freq':     kmer_freq,
        'GC_deviation':    gc_dev,
        'gene_family_raw': gene_family,
        'organism_hint':   organism_hint,
    }


def parse_fasta_file(fasta_path: str) -> list:
    """Parse a FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    header, seq = None, []
    with open(fasta_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    sequences.append((header, ''.join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header and seq:
        sequences.append((header, ''.join(seq)))
    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# 9. PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
class DNAPredictor:
    """
    Wraps all 3 trained models and encoders.
    Call .predict(sequence) to get organism, mutation status and risk level.
    """

    RISK_COLORS = {'Low': '\033[92m', 'Medium': '\033[93m', 'High': '\033[91m'}
    RESET       = '\033[0m'
    BOLD        = '\033[1m'

    def __init__(self, clf_organism, clf_mutation, clf_severity,
                 le_org, le_gene, df_reference):
        self.clf_organism  = clf_organism
        self.clf_mutation  = clf_mutation
        self.clf_severity  = clf_severity
        self.le_org        = le_org
        self.le_gene       = le_gene
        self.df_reference  = df_reference

        # Known gene families from training
        self.known_genes = list(le_gene.classes_)

    def _encode_gene(self, gene_family: str) -> int:
        """Encode gene family; fall back to 'Other' if unseen."""
        gf = gene_family if gene_family in self.known_genes else 'Other'
        return int(self.le_gene.transform([gf])[0])

    def _encode_organism(self, organism: str) -> int:
        """Encode organism; use most common if unknown."""
        if organism in self.le_org.classes_:
            return int(self.le_org.transform([organism])[0])
        # Default to median encoding (middle organism alphabetically)
        return int(len(self.le_org.classes_) // 2)

    def predict(self, sequence: str,
                gene_family: str = 'Other',
                organism_hint: str = 'Unknown',
                verbose: bool = True) -> dict:
        """
        Predict organism, mutation status and disease severity for one sequence.

        Parameters
        ----------
        sequence      : raw DNA string (ATGC...)
        gene_family   : optional known gene name e.g. 'BRCA1', 'TP53'
        organism_hint : optional organism hint; if Unknown, organism is predicted
        verbose       : print formatted results to console

        Returns
        -------
        dict with keys: organism, mutation, severity, probabilities, features
        """
        # Extract features
        feats = extract_features_from_sequence(sequence, gene_family, organism_hint)

        gene_enc = self._encode_gene(gene_family)
        org_enc  = self._encode_organism(organism_hint)

        # ── Model 1: Organism ─────────────────────────────────────────────────
        feat_org = np.array([[
            feats['Sequence_Length'], feats['GC_Content'], feats['AT_Content'],
            feats['Num_A'], feats['Num_T'], feats['Num_C'], feats['Num_G'],
            feats['kmer_3_freq'], feats['GC_deviation'], gene_enc
        ]])  # 10 features (no organism_enc for organism model)

        org_pred_idx  = self.clf_organism.predict(feat_org)[0]
        org_proba     = self.clf_organism.predict_proba(feat_org)[0]
        org_label     = self.le_org.inverse_transform([org_pred_idx])[0]
        org_conf      = round(float(org_proba[org_pred_idx]) * 100, 1)

        # Update org_enc with predicted organism for downstream models
        org_enc_pred  = self._encode_organism(org_label)

        # ── Model 2: Mutation ─────────────────────────────────────────────────
        feat_full = np.array([[
            feats['Sequence_Length'], feats['GC_Content'], feats['AT_Content'],
            feats['Num_A'], feats['Num_T'], feats['Num_C'], feats['Num_G'],
            feats['kmer_3_freq'], feats['GC_deviation'],
            gene_enc, org_enc_pred
        ]])  # 11 features

        mut_pred      = self.clf_mutation.predict(feat_full)[0]
        mut_proba     = self.clf_mutation.predict_proba(feat_full)[0]
        mut_label     = 'Mutated' if mut_pred == 1 else 'Not Mutated'
        mut_conf      = round(float(max(mut_proba)) * 100, 1)

        # ── Model 3: Disease Severity ─────────────────────────────────────────
        sev_pred      = self.clf_severity.predict(feat_full)[0]
        sev_proba     = self.clf_severity.predict_proba(feat_full)[0]
        sev_classes   = self.clf_severity.classes_
        sev_conf      = round(float(max(sev_proba)) * 100, 1)

        # Build result
        result = {
            'organism':  {'label': org_label,  'confidence': org_conf,
                          'probabilities': dict(zip(self.le_org.classes_,
                                                    [round(p*100,1) for p in org_proba]))},
            'mutation':  {'label': mut_label,   'confidence': mut_conf,
                          'probabilities': {'Not Mutated': round(float(mut_proba[0])*100,1),
                                            'Mutated':     round(float(mut_proba[1])*100,1)}},
            'severity':  {'label': sev_pred,    'confidence': sev_conf,
                          'probabilities': dict(zip(sev_classes,
                                                    [round(p*100,1) for p in sev_proba]))},
            'features':  feats,
        }

        if verbose:
            self._print_result(result, sequence)

        return result

    def _print_result(self, result: dict, sequence: str):
        """Pretty-print prediction results to terminal."""
        seq_preview = sequence[:40].upper() + ('...' if len(sequence) > 40 else '')
        sev_color   = self.RISK_COLORS.get(result['severity']['label'], '')

        print(f"\n{'═'*60}")
        print(f"{self.BOLD}  DNA SEQUENCE ANALYSIS RESULTS{self.RESET}")
        print(f"{'═'*60}")
        print(f"  Sequence   : {seq_preview}")
        print(f"  Length     : {result['features']['Sequence_Length']:,} bp")
        print(f"  GC content : {result['features']['GC_Content']}%")
        print(f"  AT content : {result['features']['AT_Content']}%")
        print(f"{'─'*60}")

        # Organism
        org = result['organism']
        print(f"\n  {self.BOLD}🔬 ORGANISM{self.RESET}")
        print(f"     Prediction  : {self.BOLD}{org['label']}{self.RESET}  "
              f"({org['confidence']}% confidence)")
        print(f"     Breakdown   : ", end='')
        for label, prob in org['probabilities'].items():
            bar = '█' * int(prob / 5)
            print(f"{label} {prob}% {bar}  ", end='')
        print()

        # Mutation
        mut = result['mutation']
        mut_icon = '⚠️ ' if mut['label'] == 'Mutated' else '✅ '
        print(f"\n  {self.BOLD}{mut_icon}MUTATION STATUS{self.RESET}")
        print(f"     Prediction  : {self.BOLD}{mut['label']}{self.RESET}  "
              f"({mut['confidence']}% confidence)")
        for label, prob in mut['probabilities'].items():
            bar = '█' * int(prob / 5)
            print(f"     {label:<15}: {prob:>5.1f}%  {bar}")

        # Severity
        sev = result['severity']
        print(f"\n  {self.BOLD}🏥 DISEASE SEVERITY{self.RESET}")
        print(f"     Risk Level  : {sev_color}{self.BOLD}{sev['label'].upper()}{self.RESET}  "
              f"({sev['confidence']}% confidence)")
        for label, prob in sorted(sev['probabilities'].items(),
                                   key=lambda x: ['Low','Medium','High'].index(x[0])
                                   if x[0] in ['Low','Medium','High'] else 99):
            color = self.RISK_COLORS.get(label, '')
            bar   = '█' * int(prob / 5)
            print(f"     {label:<8}: {prob:>5.1f}%  {color}{bar}{self.RESET}")

        # Interpretation
        print(f"\n{'─'*60}")
        print(f"  {self.BOLD}📋 INTERPRETATION{self.RESET}")
        interp = _interpret(result)
        for line in interp:
            print(f"     {line}")
        print(f"{'═'*60}\n")

    def predict_batch(self, sequences: list, verbose: bool = True) -> list:
        """
        Predict on a list of (header, sequence) tuples.
        Returns list of result dicts.
        """
        results = []
        print(f"\n  Running batch prediction on {len(sequences)} sequence(s)...\n")
        for i, (header, seq) in enumerate(sequences):
            print(f"  [{i+1}/{len(sequences)}] {header[:60]}")
            try:
                r = self.predict(seq, verbose=verbose)
                r['header'] = header
                results.append(r)
            except ValueError as e:
                print(f"    ⚠ Skipped: {e}")
        return results

    def predict_to_csv(self, sequences: list, output_path: str = 'predictions.csv'):
        """Predict batch and save results to CSV."""
        results = self.predict_batch(sequences, verbose=False)
        rows = []
        for r in results:
            rows.append({
                'Header':               r.get('header', 'Input'),
                'Sequence_Length':      r['features']['Sequence_Length'],
                'GC_Content':           r['features']['GC_Content'],
                'AT_Content':           r['features']['AT_Content'],
                'kmer_3_freq':          r['features']['kmer_3_freq'],
                'Organism':             r['organism']['label'],
                'Organism_Confidence':  r['organism']['confidence'],
                'Mutation_Status':      r['mutation']['label'],
                'Mutation_Confidence':  r['mutation']['confidence'],
                'Disease_Severity':     r['severity']['label'],
                'Severity_Confidence':  r['severity']['confidence'],
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(output_path, index=False)
        print(f"\n  ✅ Predictions saved to: {output_path}")
        print(df_out.to_string(index=False))
        return df_out


def _interpret(result: dict) -> list:
    """Generate human-readable interpretation lines."""
    lines = []
    org  = result['organism']['label']
    mut  = result['mutation']['label']
    sev  = result['severity']['label']
    gc   = result['features']['GC_Content']

    lines.append(f"Sequence appears to originate from {org}.")

    if gc < 35:
        lines.append(f"Low GC content ({gc}%) — typical of AT-rich genomic regions.")
    elif gc > 65:
        lines.append(f"High GC content ({gc}%) — characteristic of bacterial or regulatory regions.")
    else:
        lines.append(f"GC content ({gc}%) is within a normal coding range.")

    if mut == 'Mutated':
        if sev == 'High':
            lines.append("⚠ Mutation detected with HIGH disease risk — likely frameshift or nonsense variant.")
            lines.append("  Recommend clinical validation and functional assay.")
        elif sev == 'Medium':
            lines.append("⚠ Mutation detected with MEDIUM disease risk — possible missense variant.")
            lines.append("  Monitor and consider further sequencing.")
        else:
            lines.append("Mutation detected but disease risk appears LOW.")
    else:
        lines.append("No significant mutation signature detected.")
        lines.append("Sequence consistent with reference / wild-type profile.")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# FLASK WEB APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
import io
import base64
import threading
from flask import Flask, request, jsonify, render_template_string, send_from_directory

flask_app = Flask(__name__)

# Global predictor & training stats — populated once at startup
_predictor: DNAPredictor = None
_training_stats: dict = {}
_model_ready = threading.Event()


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = open("templates/index.html", encoding="utf-8").read()

@flask_app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@flask_app.route("/api/status")
def api_status():
    return jsonify({
        "ready": _model_ready.is_set(),
        "stats": _training_stats,
    })


@flask_app.route("/api/predict", methods=["POST"])
def api_predict():
    if not _model_ready.is_set():
        return jsonify({"error": "Models are still training. Please wait."}), 503

    data = request.get_json(force=True)
    sequence   = (data.get("sequence") or "").strip()
    gene_family = (data.get("gene_family") or "Other").strip() or "Other"

    if not sequence:
        return jsonify({"error": "No sequence provided."}), 400

    try:
        result = _predictor.predict(sequence, gene_family=gene_family, verbose=False)
        interp = _interpret(result)

        return jsonify({
            "ok": True,
            "sequence_preview": sequence[:60].upper() + ("..." if len(sequence) > 60 else ""),
            "length": result["features"]["Sequence_Length"],
            "gc": result["features"]["GC_Content"],
            "at": result["features"]["AT_Content"],
            "organism": result["organism"],
            "mutation": result["mutation"],
            "severity": result["severity"],
            "interpretation": interp,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@flask_app.route("/api/sample_sequences")
def api_sample_sequences():
    """Return sample sequences the user can load with one click."""
    samples = [
        {
            "label": "Bacterial (TP53-like)",
            "gene": "TP53",
            "sequence": "GCGCGCGCTATAGCGCGCGATCGATCGAGCTAGCTAGCGATCGATCGATCGCGCGTATAGCGATCGATCGATCGATCGCGCGATCGATCG"
        },
        {
            "label": "Human BRCA1-like",
            "gene": "BRCA1",
            "sequence": "ATGCGTAACGTTAGCTAGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        },
        {
            "label": "Mutated (N-rich)",
            "gene": "Other",
            "sequence": "ATGCGTAACNNNTAGCTAGCNNNGATCGATCGATCGATCNNNGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        },
        {
            "label": "Demo sequence",
            "gene": "Other",
            "sequence": "ATGTCTGCTACGATACCCGCTCGCGATCTTCGCAACCACACCGCCGAGGTACTGCGGCGAGTTGCCGCCGGCGAGGAAATCGAGGTGCTCAAGGACAATCGCCCCGTAGCGCGCATCGTTCCGCTCAAGCGGCGCCGCCAATGGTTGCCAGCTGCCGAGGTGATCGGCGAACTGGTGCGCTTGGGCCCCGATACCACCAATCTGGGCGAGGAGCTGCGAGAGACGCTGACGCAAACCACGGACGATGTGCGGTGGTGA"
        },
    ]
    return jsonify(samples)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — train then serve
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DNA 3-model pipeline — Flask Web GUI")
    parser.add_argument("--data",    default=DEFAULT_DATA_PATH, help="Path to training CSV")
    parser.add_argument("--host",    default="127.0.0.1",       help="Flask host (default 127.0.0.1)")
    parser.add_argument("--port",    default=5000, type=int,    help="Flask port (default 5000)")
    parser.add_argument("--no_plots", action="store_true",      help="Skip matplotlib plots")
    args = parser.parse_args()

    global _predictor, _training_stats

    # ── Train ─────────────────────────────────────────────────────────────────
    df, le_org, le_gene = load_and_prepare(args.data)

    print("\n  Training all 3 models...")
    clf1, X_te1, y_te1, y_pred1, rep1, acc1 = run_organism_model(df, le_org)
    clf2, X_te2, y_te2, y_pred2, rep2, acc2 = run_mutation_model(df)
    clf3, X_te3, y_te3, y_pred3, rep3, acc3 = run_severity_model(df)

    if not args.no_plots:
        fi1_padded = np.append(clf1.feature_importances_, 0.0)
        plot_summary_dashboard(df, acc1, acc2, acc3,
                               rep1, rep2, rep3,
                               fi1_padded, clf2.feature_importances_,
                               clf3.feature_importances_, le_org)

    _training_stats = {
        "organism_accuracy":  round(acc1 * 100, 2),
        "mutation_accuracy":  round(acc2 * 100, 2),
        "severity_accuracy":  round(acc3 * 100, 2),
        "dataset_rows":       len(df),
        "n_estimators":       N_ESTIMATORS,
        "organisms":          list(le_org.classes_),
    }

    _predictor = DNAPredictor(
        clf_organism=clf1, clf_mutation=clf2, clf_severity=clf3,
        le_org=le_org, le_gene=le_gene, df_reference=df
    )
    _model_ready.set()

    print(f"\n{'='*60}")
    print(f"  ✅ All models ready — launching Flask GUI")
    print(f"  Open http://{args.host}:{args.port} in your browser")
    print(f"{'='*60}\n")

    flask_app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
