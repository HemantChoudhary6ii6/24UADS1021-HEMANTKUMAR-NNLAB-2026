"""
bonus_comparison.py
====================
BONUS: Comprehensive comparison of RNN (LSTM/GRU) vs Transformer
for driving behaviour classification.(Add this file in the project to get all bonus outputs.)

Generates 8 comparison charts covering every parameter:
  1.  Training & validation accuracy curves (all 3 models)
  2.  Training & validation loss curves (all 3 models)
  3.  F1 score comparison (per class + weighted)
  4.  AUC-ROC comparison
  5.  Training time comparison
  6.  Confusion matrices side by side
  7.  Driving score distribution per model
  8.  Mean driving score per true rating (sanity check)
  9.  Radar chart — all metrics at a glance
  10. Summary table

Run AFTER the main pipeline:
    python run_pipeline.py
    python bonus_comparison.py

Output: outputs/bonus/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.makedirs('outputs/bonus', exist_ok=True)

RISK_W  = np.array([0, 25, 50, 75, 100], dtype=np.float32)
COLORS  = {'lstm': '#2563EB', 'gru': '#0E7490', 'transformer': '#C2410C'}
LABELS  = {'lstm': 'LSTM (RNN)', 'gru': 'GRU (RNN)', 'transformer': 'Transformer'}
RATINGS = [1, 2, 3, 4, 5]
NAMES   = ['lstm', 'gru', 'transformer']


# Load TransformerBlock
def _make_transformer_block(tf):
    keras    = tf.keras
    register = None
    for _try in [
        lambda: keras.saving.register_keras_serializable,
        lambda: keras.utils.register_keras_serializable,
        lambda: __import__('keras').saving.register_keras_serializable,
    ]:
        try:
            fn = _try()
            if callable(fn): register = fn; break
        except Exception: pass

    class TransformerBlock(keras.layers.Layer):
        def __init__(self, d, heads, ff, drop=0.2, **kw):
            super().__init__(**kw)
            self.d=d; self.heads=heads; self.ff=ff; self.drop=drop
            self.attn  = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=d//heads)
            self.ffn   = keras.Sequential([keras.layers.Dense(ff, activation='gelu'), keras.layers.Dense(d)])
            self.ln1   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.ln2   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = keras.layers.Dropout(drop)
            self.drop2 = keras.layers.Dropout(drop)
        def call(self, x, training=False):
            x = self.ln1(x + self.drop1(self.attn(x, x), training=training))
            x = self.ln2(x + self.drop2(self.ffn(x),     training=training))
            return x
        def get_config(self):
            cfg = super().get_config()
            cfg.update(d=self.d, heads=self.heads, ff=self.ff, drop=self.drop)
            return cfg

    if register:
        try: TransformerBlock = register(package='DrivingRisk', name='TransformerBlock')(TransformerBlock)
        except Exception: pass
    return TransformerBlock


def load_models(tf):
    TransformerBlock = _make_transformer_block(tf)
    models = {}
    for name in NAMES:
        path = f'models/{name}_model.keras'
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(
                path, custom_objects={'TransformerBlock': TransformerBlock})
    return models


def evaluate_all(models, X_te, y_te):
    from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                                  precision_score, recall_score)
    from sklearn.preprocessing import label_binarize

    results = {}
    for name, model in models.items():
        proba  = model.predict(X_te, verbose=0)
        pred0  = proba.argmax(axis=1)
        pred   = pred0 + 1

        f1_w   = f1_score(y_te, pred, average='weighted', zero_division=0)
        f1_mac = f1_score(y_te, pred, average='macro',    zero_division=0)
        f1_cls = f1_score(y_te, pred, average=None,       labels=RATINGS, zero_division=0)
        prec   = precision_score(y_te, pred, average='weighted', zero_division=0)
        rec    = recall_score(y_te, pred, average='weighted', zero_division=0)

        try:
            y_bin = label_binarize(y_te - 1, classes=[0,1,2,3,4])
            auc   = roc_auc_score(y_bin, proba, multi_class='ovr', average='weighted')
        except Exception:
            auc = float('nan')

        cm     = confusion_matrix(y_te, pred, labels=RATINGS)
        scores = (proba * RISK_W).sum(axis=1)

        results[name] = {
            'f1_weighted':  round(f1_w,   4),
            'f1_macro':     round(f1_mac, 4),
            'f1_per_class': f1_cls,
            'precision':    round(prec, 4),
            'recall':       round(rec,  4),
            'auc':          round(auc,  4),
            'confusion':    cm,
            'scores':       scores,
            'pred':         pred,
            'proba':        proba,
        }
        print(f'  {LABELS[name]:<18}  F1={f1_w:.4f}  AUC={auc:.4f}  '
              f'Prec={prec:.4f}  Rec={rec:.4f}')
    return results


# Chart 1 — Accuracy curves
def plot_accuracy_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, name in zip(axes, NAMES):
        h = histories[name]; c = COLORS[name]
        epochs = range(1, len(h['accuracy']) + 1)
        ax.plot(epochs, h['accuracy'],     color=c, lw=2,          label='Train')
        ax.plot(epochs, h['val_accuracy'], color=c, lw=2, ls='--', label='Validation', alpha=0.75)
        ax.fill_between(epochs, h['accuracy'], h['val_accuracy'],
                        alpha=0.08, color=c)
        ax.set_title(f'{LABELS[name]}\nBest val: {max(h["val_accuracy"]):.4f}',
                     fontsize=11, fontweight='bold', color=c)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
    plt.suptitle('Training vs Validation Accuracy — RNN vs Transformer',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/1_accuracy_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/1_accuracy_curves.png')


#Chart 2 — Loss curves
def plot_loss_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, name in zip(axes, NAMES):
        h = histories[name]; c = COLORS[name]
        epochs = range(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'],     color=c, lw=2,          label='Train')
        ax.plot(epochs, h['val_loss'], color=c, lw=2, ls='--', label='Validation', alpha=0.75)
        ax.fill_between(epochs, h['loss'], h['val_loss'],
                        alpha=0.08, color=c)
        ax.set_title(f'{LABELS[name]}\nBest val loss: {min(h["val_loss"]):.4f}',
                     fontsize=11, fontweight='bold', color=c)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
    plt.suptitle('Training vs Validation Loss — RNN vs Transformer',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/2_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/2_loss_curves.png')


# Chart 3 — F1 per class
def plot_f1_per_class(results):
    x     = np.arange(len(RATINGS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(NAMES):
        bars = ax.bar(x + (i-1)*width,
                      results[name]['f1_per_class'],
                      width, color=COLORS[name],
                      label=LABELS[name],
                      edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rating {r}' for r in RATINGS], fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('F1 Score per Rating Class — LSTM vs GRU vs Transformer\n'
                 'RNNs typically better on sequential patterns, Transformer on global context',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig('outputs/bonus/3_f1_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/3_f1_per_class.png')


# Chart 4 — All metrics bar comparison 
def plot_metrics_comparison(results, times):
    metrics = ['f1_weighted', 'f1_macro', 'precision', 'recall', 'auc']
    labels  = ['Weighted F1', 'Macro F1', 'Precision', 'Recall', 'AUC-ROC']
    x       = np.arange(len(metrics))
    width   = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(NAMES):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + (i-1)*width, vals, width,
                      color=COLORS[name], label=LABELS[name],
                      edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('All Classification Metrics — RNN vs Transformer',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig('outputs/bonus/4_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/4_all_metrics.png')


#Chart 5 — Training time
def plot_training_time(times, histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time bar
    ax = axes[0]
    t_vals = [times[n] for n in NAMES]
    bars   = ax.bar([LABELS[n] for n in NAMES], t_vals,
                    color=[COLORS[n] for n in NAMES],
                    edgecolor='white', linewidth=0.5, width=0.5)
    ax.bar_label(bars, fmt='%.0fs', padding=5, fontsize=12, fontweight='bold')
    ax.set_ylabel('Training time (seconds)', fontsize=11)
    ax.set_title('Total Training Time\n(30 epochs, same hardware)',
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # Time per epoch
    ax = axes[1]
    for name in NAMES:
        n_epochs     = len(histories[name]['loss'])
        time_per_ep  = times[name] / n_epochs
        ax.bar(LABELS[name], time_per_ep,
               color=COLORS[name], edgecolor='white', linewidth=0.5, width=0.5)
        ax.text(list(NAMES).index(name), time_per_ep + 0.5,
                f'{time_per_ep:.1f}s', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Seconds per epoch', fontsize=11)
    ax.set_title('Average Time per Epoch\n(lower = more efficient)',
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Training Efficiency — RNN vs Transformer', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/5_training_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/5_training_time.png')


# Chart 6 — Confusion matrices
def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, NAMES):
        r = results[name]
        # Normalised confusion matrix (shows %)
        cm_norm = r['confusion'].astype(float)
        cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=RATINGS, yticklabels=RATINGS, ax=ax,
                    linewidths=0.3, linecolor='white',
                    annot_kws={'size': 10},
                    vmin=0, vmax=100)
        ax.set_title(f'{LABELS[name]}\nF1={r["f1_weighted"]:.4f}  AUC={r["auc"]:.4f}',
                     fontsize=11, fontweight='bold', color=COLORS[name])
        ax.set_xlabel('Predicted rating', fontsize=10)
        ax.set_ylabel('True rating', fontsize=10)
    plt.suptitle('Normalised Confusion Matrices (%) — RNN vs Transformer\n'
                 'Diagonal = correct predictions. Off-diagonal = misclassifications.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/6_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/6_confusion_matrices.png')


#Chart 7 — Score distributions
def plot_score_distributions(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    rc = {1: '#991b1b', 2: '#ef4444', 3: '#f59e0b', 4: '#84cc16', 5: '#22c55e'}
    for ax, name in zip(axes, NAMES):
        r = results[name]
        for rating in RATINGS:
            mask = (r['pred'] == rating)
            if mask.sum() > 0:
                ax.hist(r['scores'][mask], bins=20, alpha=0.55,
                        color=rc[rating], label=f'Rating {rating}')
        ax.axvline(r['scores'].mean(), color='black', lw=1.5, ls='--',
                   label=f'Mean={r["scores"].mean():.1f}')
        ax.set_title(f'{LABELS[name]}\nMean score: {r["scores"].mean():.1f}/100',
                     fontsize=11, fontweight='bold', color=COLORS[name])
        ax.set_xlabel('Driving score (0=dangerous, 100=perfect)', fontsize=10)
        ax.set_ylabel('Window count', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        if ax == axes[0]:
            ax.legend(fontsize=8, title='Predicted\nrating')
    plt.suptitle('Driving Score Distribution — RNN vs Transformer\n'
                 'Well-separated distributions = better discrimination',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/7_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/7_score_distributions.png')


#Chart 8 — Mean score per rating
def plot_score_by_rating(results, y_te):
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(RATINGS))
    width = 0.25
    for i, name in enumerate(NAMES):
        r     = results[name]
        means = [r['scores'][y_te == rt].mean() if (y_te == rt).sum() > 0 else 0
                 for rt in RATINGS]
        bars = ax.bar(x + (i-1)*width, means, width,
                      color=COLORS[name], label=LABELS[name],
                      edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rating {r}' for r in RATINGS], fontsize=11)
    ax.set_ylabel('Mean driving score (0–100)', fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title('Mean Driving Score per True Rating\n'
                 'Score must increase from Rating 1 → Rating 5 to be trustworthy',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    # Ideal line
    ideal = [0, 25, 50, 75, 100]
    ax.plot(x, ideal, 'k--', lw=1, alpha=0.3, label='Ideal')
    plt.tight_layout()
    plt.savefig('outputs/bonus/8_score_by_rating.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/8_score_by_rating.png')


# Chart 9 — Radar chart
def plot_radar(results, times):
    categories  = ['F1\n(weighted)', 'AUC-\nROC', 'Precision', 'Recall',
                   'Speed\n(inv)', 'Val Acc\n(best)']
    histories   = joblib.load('models/train_history.pkl')
    max_time    = max(times[n] for n in NAMES)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    N       = len(categories)
    angles  = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(color='gray', alpha=0.3)

    for name in NAMES:
        r      = results[name]
        speed  = 1 - (times[name] / max_time)   # invert so higher = faster
        best_v = max(histories[name]['val_accuracy'])
        vals   = [r['f1_weighted'], r['auc'], r['precision'],
                  r['recall'],      speed,    best_v]
        vals  += vals[:1]
        ax.plot(angles, vals, lw=2, color=COLORS[name], label=LABELS[name])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[name])

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    ax.set_title('Model Comparison Radar\nRNN vs Transformer — all metrics',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/bonus/9_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/9_radar_chart.png')


#Chart 10 — Convergence speed
def plot_convergence(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name in NAMES:
        h      = histories[name]
        vals   = h['val_accuracy']
        epochs = range(1, len(vals) + 1)
        ax.plot(epochs, vals, color=COLORS[name], lw=2, label=LABELS[name])
        # Mark epoch where 90% val_acc first reached
        for ep, v in enumerate(vals):
            if v >= 0.90:
                ax.axvline(ep + 1, color=COLORS[name], lw=0.8, ls=':', alpha=0.6)
                ax.annotate(f'{ep+1}', xy=(ep+1, 0.90),
                            color=COLORS[name], fontsize=9, ha='center')
                break
    ax.axhline(0.90, color='gray', lw=0.8, ls='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation accuracy', fontsize=11)
    ax.set_title('Convergence Speed\n(dotted line = first epoch to reach 90%)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2)

    ax = axes[1]
    for name in NAMES:
        h      = histories[name]
        vals   = h['val_loss']
        epochs = range(1, len(vals) + 1)
        ax.plot(epochs, vals, color=COLORS[name], lw=2, label=LABELS[name])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation loss', fontsize=11)
    ax.set_title('Validation Loss Convergence\n(lower and smoother = better)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2)

    plt.suptitle('Convergence Analysis — RNN vs Transformer',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/bonus/10_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved → outputs/bonus/10_convergence.png')


# Summary table
def save_summary_table(results, times, histories):
    rows = []
    for name in NAMES:
        r     = results[name]
        h     = histories[name]
        rows.append({
            'Model':             LABELS[name],
            'Type':              'RNN' if name in ('lstm', 'gru') else 'Transformer',
            'Weighted F1':       r['f1_weighted'],
            'Macro F1':          r['f1_macro'],
            'AUC-ROC':           r['auc'],
            'Precision':         r['precision'],
            'Recall':            r['recall'],
            'Best val accuracy': round(max(h['val_accuracy']), 4),
            'Final val accuracy':round(h['val_accuracy'][-1], 4),
            'Best val loss':     round(min(h['val_loss']), 4),
            'Training time (s)': times[name],
            'Time per epoch (s)':round(times[name] / len(h['loss']), 1),
            'Mean score /100':   round(float(r['scores'].mean()), 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv('outputs/bonus/summary_table.csv', index=False)

    print('\n  ╔══════════════════════════════════════════════════════════════╗')
    print(  '  ║  BONUS COMPARISON SUMMARY — RNN vs Transformer              ║')
    print(  '  ╠══════════════════════════════════════════════════════════════╣')
    print(f'  {"Model":<20} {"F1":>8} {"AUC":>8} {"Prec":>8} {"Rec":>8} {"Time":>8}')
    print(  '  ─────────────────────────────────────────────────────────────')
    for _, row in df.iterrows():
        marker = ' ← best' if row['Weighted F1'] == df['Weighted F1'].max() else ''
        print(f'  {row["Model"]:<20} {row["Weighted F1"]:>8.4f} {row["AUC-ROC"]:>8.4f} '
              f'{row["Precision"]:>8.4f} {row["Recall"]:>8.4f} '
              f'{row["Training time (s)"]:>7.0f}s{marker}')
    print(  '  ╚══════════════════════════════════════════════════════════════╝')
    print(f'\n  Saved → outputs/bonus/summary_table.csv')

    # Key findings
    best   = df.loc[df['Weighted F1'].idxmax(), 'Model']
    fastest= df.loc[df['Training time (s)'].idxmin(), 'Model']
    rnn_f1 = df[df['Type']=='RNN']['Weighted F1'].mean()
    tfm_f1 = df[df['Type']=='Transformer']['Weighted F1'].mean()
    print(f'\n  KEY FINDINGS:')
    print(f'  Best accuracy  : {best}')
    print(f'  Fastest model  : {fastest}')
    print(f'  RNN avg F1     : {rnn_f1:.4f}')
    print(f'  Transformer F1 : {tfm_f1:.4f}')
    winner = 'RNN models' if rnn_f1 > tfm_f1 else 'Transformer'
    print(f'  Winner         : {winner} (+{abs(rnn_f1-tfm_f1):.4f} F1 difference)')


#Main
def main():
    print('\n' + '='*60)
    print('  BONUS — RNN vs Transformer Comprehensive Comparison')
    print('='*60)

    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print(f'  TensorFlow {tf.__version__}')
    except ImportError:
        print('  ERROR: pip install tensorflow')
        raise SystemExit(1)

    for f in ['data/sequences_X.npy', 'data/sequences_y.npy',
              'data/split_indices.pkl', 'models/train_history.pkl',
              'models/train_times.json']:
        if not os.path.exists(f):
            print(f'  ERROR: {f} missing. Run run_pipeline.py first.')
            raise SystemExit(1)

    X       = np.load('data/sequences_X.npy')
    y       = np.load('data/sequences_y.npy')
    splits  = joblib.load('data/split_indices.pkl')
    X_te    = X[splits['test']]
    y_te    = y[splits['test']]
    hist    = joblib.load('models/train_history.pkl')
    with open('models/train_times.json') as f:
        times = json.load(f)

    print(f'\n  Test windows : {len(X_te):,}')
    print(f'\n  Evaluating all models ...')

    models  = load_models(tf)
    results = evaluate_all(models, X_te, y_te)

    print(f'\n  Generating {10} comparison charts ...')
    plot_accuracy_curves(hist)
    plot_loss_curves(hist)
    plot_f1_per_class(results)
    plot_metrics_comparison(results, times)
    plot_training_time(times, hist)
    plot_confusion_matrices(results)
    plot_score_distributions(results)
    plot_score_by_rating(results, y_te)
    plot_radar(results, times)
    plot_convergence(hist)
    save_summary_table(results, times, hist)

    print(f'\n  All outputs saved to outputs/bonus/')
    print(f'  Add these charts to your report for the bonus marks.')


if __name__ == '__main__':
    main()