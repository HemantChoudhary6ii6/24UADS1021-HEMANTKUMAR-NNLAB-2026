import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import fashion_mnist

# 1.  LOAD & PREPROCESS
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test  = x_test .astype("float32")[..., np.newaxis] / 255.0

x_val, y_val = x_train[-6000:], y_train[-6000:]
x_tr,  y_tr  = x_train[:-6000], y_train[:-6000]

print(f"Train: {x_tr.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

# 2.  MODEL BUILDER
def build_model(filter_size=3, use_regularization=False, optimizer='adam'):
    """
    Two conv-blocks + dense classifier.
    filter_size        : kernel size (3 or 5)
    use_regularization : L2 weight decay + Dropout when True
    optimizer          : string or keras Optimizer instance
    """
    l2 = regularizers.l2(1e-4) if use_regularization else None

    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, filter_size, padding='same', activation='relu',
                      kernel_regularizer=l2, input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, filter_size, padding='same', activation='relu',
                      kernel_regularizer=l2),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25) if use_regularization else layers.Lambda(lambda x: x),

        # Block 2
        layers.Conv2D(64, filter_size, padding='same', activation='relu',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(64, filter_size, padding='same', activation='relu',
                      kernel_regularizer=l2),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25) if use_regularization else layers.Lambda(lambda x: x),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=l2),
        layers.Dropout(0.50) if use_regularization else layers.Lambda(lambda x: x),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3.  TRAINING HELPER
EPOCHS = 5

def train_and_evaluate(label, filter_size=3, use_reg=False,
                       batch_size=64, optimizer='adam'):
    print(f"\n{'━'*60}\n  {label}\n{'━'*60}")
    model   = build_model(filter_size, use_reg, optimizer)
    history = model.fit(x_tr, y_tr,
                        validation_data=(x_val, y_val),
                        epochs=EPOCHS, batch_size=batch_size, verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc*100:.2f}%  |  Loss: {test_loss:.4f}")
    return history, test_acc, test_loss, model

# 4.  RUN EXPERIMENTS
# A - Filter size
hist_f3,    acc_f3,    loss_f3,    _          = train_and_evaluate("Filter 3x3 (baseline)", filter_size=3)
hist_f5,    acc_f5,    loss_f5,    _          = train_and_evaluate("Filter 5x5",             filter_size=5)

# B - Regularization
hist_noreg, acc_noreg, loss_noreg, _          = train_and_evaluate("No Regularization",  use_reg=False)
hist_reg,   acc_reg,   loss_reg,   _          = train_and_evaluate("L2 + Dropout",        use_reg=True)

# C - Batch size
hist_b32,   acc_b32,   loss_b32,   _          = train_and_evaluate("Batch 32",  batch_size=32)
hist_b128,  acc_b128,  loss_b128,  _          = train_and_evaluate("Batch 128", batch_size=128)

# D - Optimizer
hist_adam,  acc_adam,  loss_adam,  best_model = train_and_evaluate("Adam",    optimizer=keras.optimizers.Adam(1e-3))
hist_sgd,   acc_sgd,   loss_sgd,   _          = train_and_evaluate("SGD+Mom", optimizer=keras.optimizers.SGD(0.01, momentum=0.9))
hist_rmsp,  acc_rmsp,  loss_rmsp,  _          = train_and_evaluate("RMSprop", optimizer=keras.optimizers.RMSprop(1e-3))

# 5.  SUMMARY TABLE
results = [
    ("Filter 3x3 (baseline)",    acc_f3,    loss_f3),
    ("Filter 5x5",                acc_f5,    loss_f5),
    ("No Regularization",         acc_noreg, loss_noreg),
    ("L2 + Dropout",              acc_reg,   loss_reg),
    ("Batch size 32",             acc_b32,   loss_b32),
    ("Batch size 128",            acc_b128,  loss_b128),
    ("Optimizer: Adam",           acc_adam,  loss_adam),
    ("Optimizer: SGD+Momentum",   acc_sgd,   loss_sgd),
    ("Optimizer: RMSprop",        acc_rmsp,  loss_rmsp),
]

print("\n\n" + "="*58)
print(f"  {'Experiment':<36} {'Acc (%)':>8}  {'Loss':>8}")
print("="*58)
for name, acc, loss in results:
    print(f"  {name:<36} {acc*100:>7.2f}%  {loss:>8.4f}")
print("="*58)

# 6.  PLOTTING UTILITIES
BG, PANEL    = '#0d0f18', '#12152a'
TC, LC, GC   = '#e8e8e8', '#aaaaaa', '#252545'
C = ['#00d4ff', '#ff6b6b', '#a8ff78', '#ffbe0b', '#8338ec', '#06d6a0']

ep = range(1, EPOCHS + 1)

def style(ax, title, ylabel='Accuracy'):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TC, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', color=LC, fontsize=9)
    ax.set_ylabel(ylabel, color=LC, fontsize=9)
    ax.tick_params(colors=LC)
    ax.grid(True, color=GC, ls='--', alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(GC)
    ax.legend(facecolor=PANEL, edgecolor=GC, labelcolor=TC, fontsize=8)

# FIGURE 1 — Training curves (accuracy + loss) for all 4 experiments
fig1, axes1 = plt.subplots(4, 2, figsize=(16, 22), facecolor=BG)
fig1.suptitle('CNN on Fashion-MNIST  Training Curves (5 Epochs)',
              fontsize=15, fontweight='bold', color='white', y=0.995)

experiment_rows = [
    ([(hist_f3,    'Filter 3x3', C[0]),
      (hist_f5,    'Filter 5x5', C[1])],
     'A  Filter Size  Accuracy', 'A  Filter Size  Loss'),

    ([(hist_noreg, 'No Reg',     C[2]),
      (hist_reg,   'L2+Dropout', C[3])],
     'B  Regularization  Accuracy', 'B  Regularization  Loss'),

    ([(hist_b32,  'Batch 32',   C[4]),
      (hist_b128, 'Batch 128',  C[5])],
     'C  Batch Size  Accuracy', 'C  Batch Size  Loss'),

    ([(hist_adam, 'Adam',       C[0]),
      (hist_sgd,  'SGD+Mom',    C[1]),
      (hist_rmsp, 'RMSprop',    C[2])],
     'D  Optimizer  Accuracy', 'D  Optimizer  Loss'),
]

for row, (hist_list, title_acc, title_loss) in enumerate(experiment_rows):
    ax_acc = axes1[row, 0]
    ax_los = axes1[row, 1]
    for hist, lbl, color in hist_list:
        ax_acc.plot(ep, hist.history['accuracy'],     color=color, lw=2, ls='--', label=f'{lbl} train')
        ax_acc.plot(ep, hist.history['val_accuracy'], color=color, lw=2,          label=f'{lbl} val')
        ax_los.plot(ep, hist.history['loss'],         color=color, lw=2, ls='--', label=f'{lbl} train')
        ax_los.plot(ep, hist.history['val_loss'],     color=color, lw=2,          label=f'{lbl} val')
    style(ax_acc, title_acc, 'Accuracy')
    style(ax_los, title_loss, 'Loss')

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig('fig1_training_curves.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nSaved -> fig1_training_curves.png")

# FIGURE 2 — Test accuracy bar + overfitting gap
fig2, (ax_bar, ax_gap) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
fig2.suptitle('CNN on Fashion-MNIST  Performance Comparison',
              fontsize=14, fontweight='bold', color='white')

# Bar: test accuracy
bar_labels = [r[0] for r in results]
accs       = [r[1] * 100 for r in results]
bcols      = [C[i % len(C)] for i in range(len(results))]

ax_bar.set_facecolor(PANEL)
bars = ax_bar.bar(bar_labels, accs, color=bcols, edgecolor=BG, width=0.6)
for bar, acc in zip(bars, accs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{acc:.1f}%', ha='center', va='bottom',
                color=TC, fontsize=9, fontweight='bold')
ax_bar.set_ylim(min(accs) - 3, 101)
ax_bar.set_ylabel('Test Accuracy (%)', color=LC, fontsize=11)
ax_bar.set_title('Test Accuracy  All Experiments', color=TC, fontsize=12, fontweight='bold')
ax_bar.tick_params(axis='x', colors=LC, rotation=25)
ax_bar.tick_params(axis='y', colors=LC)
ax_bar.grid(True, axis='y', color=GC, ls='--', alpha=0.6)
for sp in ax_bar.spines.values():
    sp.set_color(GC)

# Bar: train-val gap (overfitting indicator) at final epoch
gap_labels = ['Filter 3x3', 'Filter 5x5',
              'No Reg',      'L2+Drop',
              'Batch 32',    'Batch 128',
              'Adam',        'SGD+Mom',   'RMSprop']

gap_values = [
    (hist_f3.history['accuracy'][-1]    - hist_f3.history['val_accuracy'][-1])    * 100,
    (hist_f5.history['accuracy'][-1]    - hist_f5.history['val_accuracy'][-1])    * 100,
    (hist_noreg.history['accuracy'][-1] - hist_noreg.history['val_accuracy'][-1]) * 100,
    (hist_reg.history['accuracy'][-1]   - hist_reg.history['val_accuracy'][-1])   * 100,
    (hist_b32.history['accuracy'][-1]   - hist_b32.history['val_accuracy'][-1])   * 100,
    (hist_b128.history['accuracy'][-1]  - hist_b128.history['val_accuracy'][-1])  * 100,
    (hist_adam.history['accuracy'][-1]  - hist_adam.history['val_accuracy'][-1])  * 100,
    (hist_sgd.history['accuracy'][-1]   - hist_sgd.history['val_accuracy'][-1])   * 100,
    (hist_rmsp.history['accuracy'][-1]  - hist_rmsp.history['val_accuracy'][-1])  * 100,
]

gcols = ['#ff6b6b' if g > 5 else '#a8ff78' for g in gap_values]
ax_gap.set_facecolor(PANEL)
gbars = ax_gap.bar(gap_labels, gap_values, color=gcols, edgecolor=BG, width=0.6)
for bar, g in zip(gbars, gap_values):
    ax_gap.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{g:.1f}%', ha='center', va='bottom',
                color=TC, fontsize=9, fontweight='bold')
ax_gap.axhline(5, color='#ffbe0b', ls='--', lw=1.5, label='Overfitting threshold (5%)')
ax_gap.set_ylabel('Train - Val Accuracy (%)', color=LC, fontsize=11)
ax_gap.set_title('Overfitting Gap  (lower = better generalisation)',
                 color=TC, fontsize=12, fontweight='bold')
ax_gap.tick_params(axis='x', colors=LC, rotation=25)
ax_gap.tick_params(axis='y', colors=LC)
ax_gap.grid(True, axis='y', color=GC, ls='--', alpha=0.6)
for sp in ax_gap.spines.values():
    sp.set_color(GC)
ax_gap.legend(facecolor=PANEL, edgecolor=GC, labelcolor=TC, fontsize=9)

plt.tight_layout()
plt.savefig('fig2_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved -> fig2_comparison.png")

# FIGURE 3 — Confusion matrix for best model (Adam)
y_pred = best_model.predict(x_test, verbose=0).argmax(axis=1)
cm     = confusion_matrix(y_test, y_pred)

fig3, ax3 = plt.subplots(figsize=(11, 9), facecolor=BG)
ax3.set_facecolor(PANEL)
im = ax3.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax3)
ax3.set_xticks(range(10));  ax3.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', color=LC)
ax3.set_yticks(range(10));  ax3.set_yticklabels(CLASS_NAMES, color=LC)
ax3.set_xlabel('Predicted Label', color=LC, fontsize=11)
ax3.set_ylabel('True Label',      color=LC, fontsize=11)
ax3.set_title('Confusion Matrix  Best Model (Adam, 5 Epochs)',
              color=TC, fontsize=13, fontweight='bold', pad=12)
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                 color='white' if cm[i, j] > thresh else 'black')
for sp in ax3.spines.values():
    sp.set_color(GC)
ax3.tick_params(colors=LC)
plt.tight_layout()
plt.savefig('fig3_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved -> fig3_confusion_matrix.png")

# FIGURE 4 — Sample predictions grid (3 rows x 8 cols = 24 images)
fig4, axes4 = plt.subplots(3, 8, figsize=(18, 7), facecolor=BG)
fig4.suptitle('Sample Predictions  Best Model  (blue = correct  |  red = wrong)',
              color='white', fontsize=13, fontweight='bold')

for i, ax in enumerate(axes4.flat):
    ax.imshow(x_test[i, :, :, 0], cmap='gray')
    pred_l = CLASS_NAMES[y_pred[i]]
    true_l = CLASS_NAMES[y_test[i]]
    color  = '#00d4ff' if y_pred[i] == y_test[i] else '#ff6b6b'
    ax.set_title(f'P: {pred_l}\nT: {true_l}', color=color, fontsize=7, pad=2)
    ax.axis('off')

plt.tight_layout()
plt.savefig('fig4_sample_predictions.png', dpi=130, bbox_inches='tight', facecolor=BG)
print("Saved -> fig4_sample_predictions.png")

# FIGURE 5 — Per-class accuracy bar chart
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

fig5, ax5 = plt.subplots(figsize=(12, 6), facecolor=BG)
ax5.set_facecolor(PANEL)
pcols = [C[i % len(C)] for i in range(10)]
pbars = ax5.bar(CLASS_NAMES, per_class_acc, color=pcols, edgecolor=BG, width=0.6)
for bar, acc in zip(pbars, per_class_acc):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f'{acc:.1f}%', ha='center', va='bottom',
             color=TC, fontsize=9, fontweight='bold')
ax5.set_ylim(0, 110)
ax5.axhline(per_class_acc.mean(), color='#ffbe0b', ls='--', lw=2,
            label=f'Mean Accuracy: {per_class_acc.mean():.1f}%')
ax5.set_ylabel('Accuracy (%)', color=LC, fontsize=11)
ax5.set_title('Per-Class Accuracy  Best Model (Adam)',
              color=TC, fontsize=13, fontweight='bold')
ax5.tick_params(axis='x', colors=LC, rotation=20)
ax5.tick_params(axis='y', colors=LC)
ax5.grid(True, axis='y', color=GC, ls='--', alpha=0.6)
for sp in ax5.spines.values():
    sp.set_color(GC)
ax5.legend(facecolor=PANEL, edgecolor=GC, labelcolor=TC, fontsize=10)

plt.tight_layout()
plt.savefig('fig5_per_class_accuracy.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Saved -> fig5_per_class_accuracy.png")

print("\nAll done!")