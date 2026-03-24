import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1.  Reproducibility
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2.  Generate Sample Time Series Dataset
def generate_time_series(n_points: int = 1000) -> np.ndarray:
    """
    Synthetic time series: sine wave + small Gaussian noise.
    Easy to visualise, yet non-trivial enough to test the RNN.
    """
    t = np.linspace(0, 8 * np.pi, n_points)
    series = np.sin(t) + 0.1 * np.random.randn(n_points)
    return series.astype(np.float32)

# 3.  PyTorch Dataset
class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset.
    Each sample  →  (sequence of length seq_len, next scalar target)
    """

    def __init__(self, series: np.ndarray, seq_len: int = 30):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for i in range(len(series) - seq_len):
            self.X.append(series[i : i + seq_len])
            self.y.append(series[i + seq_len])
        self.X = torch.tensor(np.array(self.X)).unsqueeze(-1)   # (N, seq_len, 1)
        self.y = torch.tensor(np.array(self.y)).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4.  RNN Model
class RNNModel(nn.Module):
    """
    A simple many-to-one RNN for time-series forecasting.

    Architecture
    ────────────
    RNN  →  Dropout  →  Linear
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)          # out: (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :]) # take last time-step
        return self.fc(out)               # (batch, output_size)

# 5.  Training Function
def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred  = model(X_batch)
        loss  = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

# 6.  Evaluation Function
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss   = 0.0
    all_preds    = []
    all_targets  = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred   = model(X_batch)
            loss   = criterion(pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)

# 7.  Metrics
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mse  = np.mean((preds - targets) ** 2)
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R²": r2}

# 8.  Plotting  (6 comprehensive panels)
def plot_results(
    train_losses: list,
    val_losses: list,
    preds: np.ndarray,
    targets: np.ndarray,
    series_full: np.ndarray,
    n_train: int,
    n_val: int,
    metrics: dict,
    n_show: int = 120,
):
    errors = preds - targets
    palette = {
        "blue":   "#2563EB",
        "red":    "#DC2626",
        "green":  "#059669",
        "amber":  "#D97706",
        "purple": "#7C3AED",
        "slate":  "#475569",
    }

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "RNN Time-Series Prediction — Full Evaluation Dashboard",
        fontsize=16, fontweight="bold", color="#1E293B", y=0.98,
    )

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)

    # ── Panel 1: Full dataset overview with train/val/test split ──────────
    ax1 = fig.add_subplot(gs[0, :2])   # wide, spans 2 columns
    n_test = len(series_full) - n_train - n_val
    t_all  = np.arange(len(series_full))
    # shade regions
    ax1.axvspan(0,                    n_train,          alpha=0.08, color=palette["blue"],  label="Train region")
    ax1.axvspan(n_train,              n_train + n_val,  alpha=0.12, color=palette["amber"], label="Val region")
    ax1.axvspan(n_train + n_val,      len(series_full), alpha=0.10, color=palette["green"], label="Test region")
    ax1.plot(t_all, series_full, color=palette["slate"], linewidth=0.9, alpha=0.8, label="Raw series")
    ax1.set_title("① Full Time-Series with Train / Val / Test Split", fontweight="bold")
    ax1.set_xlabel("Time Step"); ax1.set_ylabel("Value")
    ax1.legend(fontsize=8, ncol=4); ax1.grid(alpha=0.25)

    # ── Panel 2: Training & Validation loss curves ────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, color=palette["blue"],  linewidth=1.8, label="Train MSE")
    ax2.plot(epochs, val_losses,   color=palette["red"],   linewidth=1.8, linestyle="--", label="Val MSE")
    ax2.fill_between(epochs, train_losses, val_losses,
                     where=[v > t for v, t in zip(val_losses, train_losses)],
                     alpha=0.08, color=palette["red"])
    ax2.set_title("② Train vs Validation Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
    ax2.legend(); ax2.grid(alpha=0.25)

    # ── Panel 3: Predictions vs Actuals (zoomed test window) ──────────────
    ax3 = fig.add_subplot(gs[1, :2])
    t_win = np.arange(n_show)
    ax3.plot(t_win, targets[:n_show], color=palette["green"], linewidth=2.0, label="Actual")
    ax3.plot(t_win, preds[:n_show],   color=palette["amber"], linewidth=1.8,
             linestyle="--", alpha=0.9, label="Predicted")
    ax3.fill_between(t_win, targets[:n_show], preds[:n_show],
                     alpha=0.12, color=palette["amber"])
    ax3.set_title(f"③ Predictions vs Actuals — first {n_show} test steps", fontweight="bold")
    ax3.set_xlabel("Time Step"); ax3.set_ylabel("Value")
    ax3.legend(); ax3.grid(alpha=0.25)

    # ── Panel 4: Scatter — Actual vs Predicted ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(targets, preds, alpha=0.35, s=12,
                color=palette["purple"], edgecolors="none")
    lo, hi = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax4.plot([lo, hi], [lo, hi], color=palette["red"], linewidth=1.5,
             linestyle="--", label="Perfect fit")
    ax4.set_title(f"④ Actual vs Predicted  (R²={metrics['R²']:.3f})", fontweight="bold")
    ax4.set_xlabel("Actual"); ax4.set_ylabel("Predicted")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.25)

    plt.show()
    plt.close()
    # Second figure — Error & Distribution analysis (2 panels)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor("#F8FAFC")
    fig2.suptitle("RNN — Residual & Distribution Analysis",
                  fontsize=14, fontweight="bold", color="#1E293B")

    # Panel 5: Residuals over time
    ax5 = axes2[0]
    ax5.plot(errors, color=palette["purple"], linewidth=0.9, alpha=0.8)
    ax5.axhline(0,            color=palette["red"],   linewidth=1.2, linestyle="--")
    ax5.axhline( errors.std(), color=palette["slate"], linewidth=0.8, linestyle=":")
    ax5.axhline(-errors.std(), color=palette["slate"], linewidth=0.8, linestyle=":",
                label=f"±1 std ({errors.std():.3f})")
    ax5.fill_between(range(len(errors)), errors, 0,
                     where=(errors > 0), alpha=0.15, color=palette["amber"])
    ax5.fill_between(range(len(errors)), errors, 0,
                     where=(errors < 0), alpha=0.15, color=palette["blue"])
    ax5.set_title("⑤ Residuals (Predicted − Actual) over Test Steps", fontweight="bold")
    ax5.set_xlabel("Test Step"); ax5.set_ylabel("Error")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.25)

    #Panel 6: Error distribution histogram
    ax6 = axes2[1]
    ax6.hist(errors, bins=30, color=palette["blue"], alpha=0.72,
             edgecolor="white", linewidth=0.5)
    ax6.axvline(errors.mean(), color=palette["red"],   linewidth=1.8,
                linestyle="--", label=f"Mean={errors.mean():.4f}")
    ax6.axvline(errors.mean() + errors.std(), color=palette["amber"],
                linewidth=1.2, linestyle=":", label=f"±1σ={errors.std():.4f}")
    ax6.axvline(errors.mean() - errors.std(), color=palette["amber"],
                linewidth=1.2, linestyle=":")
    ax6.set_title("⑥ Error Distribution Histogram", fontweight="bold")
    ax6.set_xlabel("Prediction Error"); ax6.set_ylabel("Count")
    ax6.legend(fontsize=8); ax6.grid(alpha=0.25)

    fig2.tight_layout()
    plt.show()
    plt.close()

# 9.  Main
def main():
    #Hyper-parameters 
    SEQ_LEN    = 30
    BATCH_SIZE = 64
    EPOCHS     = 50
    LR         = 1e-3
    TRAIN_FRAC = 0.70
    VAL_FRAC   = 0.15
    # TEST_FRAC  = 0.15  (remainder)

    #Data 
    series = generate_time_series(n_points=1000)
    n      = len(series)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_series = series[:n_train]
    val_series   = series[n_train : n_train + n_val]
    test_series  = series[n_train + n_val :]

    train_ds = TimeSeriesDataset(train_series, SEQ_LEN)
    val_ds   = TimeSeriesDataset(val_series,   SEQ_LEN)
    test_ds  = TimeSeriesDataset(test_series,  SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    print(f"Dataset splits  →  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    #Model, Loss, Optimiser
    model     = RNNModel(hidden_size=64, num_layers=2, dropout=0.2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Training Loop 
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state    = None

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
    print("─" * 36)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, train_loader, criterion, optimizer)
        vl_loss, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>12.6f}  {vl_loss:>10.6f}")

    # Restore Best Weights 
    model.load_state_dict(best_state)
    print(f"\n✓ Best validation loss: {best_val_loss:.6f}")

    #Test Evaluation
    test_loss, preds, targets = evaluate(model, test_loader, criterion)
    metrics = compute_metrics(preds, targets)

    print("\n" + "═" * 40)
    print("        TEST SET EVALUATION")
    print("═" * 40)
    print(f"  Test MSE  : {metrics['MSE']:.6f}")
    print(f"  Test MAE  : {metrics['MAE']:.6f}")
    print(f"  Test RMSE : {metrics['RMSE']:.6f}")
    print(f"  Test R²   : {metrics['R²']:.4f}")
    print("═" * 40)

    #Plot
    plot_results(
        train_losses, val_losses,
        preds, targets,
        series_full=series,
        n_train=n_train,
        n_val=n_val,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()