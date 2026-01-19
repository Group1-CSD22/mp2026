"""
Research-Grade Training Pipeline v4.0 - BEAUTIFUL PLOTS EDITION
- IEEE/Springer standard figures (Times New Roman, 300 DPI)
- Smoothed training curves
- Enhanced error analysis
- Session-based splitting (Leakage Fixed)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, roc_curve, auc, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler, label_binarize
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from scipy.signal import savgol_filter

# --- PUBLICATION STYLE CONFIGURATION ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class ResearchTrainingPipeline:
    """Training pipeline with publication-quality outputs"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch': [], 'lr': []
        }

        self.label_map = {
            0: 'Deep Work', 1: 'Active Work', 2: 'Research',
            3: 'Communication', 4: 'Distracted', 5: 'Idle'
        }

        # Short names for tight plots
        self.state_abbrev = {
            0: 'DW', 1: 'AW', 2: 'Res',
            3: 'Com', 4: 'Dis', 5: 'Idle'
        }

        # Setup directories
        self.output_dir = Path('research_outputs')
        self.figures_dir = self.output_dir / 'figures'
        self.metrics_dir = self.output_dir / 'metrics'
        self.models_dir = self.output_dir / 'models'

        for d in [self.output_dir, self.figures_dir, self.metrics_dir, self.models_dir]:
            d.mkdir(exist_ok=True, parents=True)

        print(f"✅ Pipeline initialized on {device}")

    def prepare_data(self, train_data, val_data, test_data):
        """Normalize features"""
        print("\n" + "="*60 + "\nDATA NORMALIZATION\n" + "="*60)

        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        n_features = X_train.shape[2]

        # Fit scaler on TRAINING data only
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)

        # Transform Val/Test
        X_val = X_val.reshape(-1, n_features)
        X_val = self.scaler.transform(X_val).reshape(val_data[0].shape)

        X_test = X_test.reshape(-1, n_features)
        X_test = self.scaler.transform(X_test).reshape(test_data[0].shape)

        # Save class distribution for plotting later
        self.class_dist = {
            'Train': np.unique(y_train, return_counts=True)[1],
            'Val': np.unique(y_val, return_counts=True)[1],
            'Test': np.unique(y_test, return_counts=True)[1]
        }

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self, train_data, val_data, epochs=100, batch_size=64, lr=0.0003):
        """Train loop with stabilized hyperparameters"""
        print(f"\nTraining for {epochs} epochs | Batch: {batch_size} | LR: {lr}")

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Convert to tensors
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        criterion = nn.CrossEntropyLoss()

        # FIX 1: Lower LR and add Weight Decay for regularization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # FIX 2: More patient scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # FIX 3: Tighter Gradient Clipping (0.5) to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            avg_loss = total_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
                _, val_pred = torch.max(val_out.data, 1)
                val_acc = (val_pred == y_val_t).sum().item() / len(y_val_t)

            # Update history
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch'].append(epoch)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            scheduler.step(val_loss)

            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch + 1:3d} | TLoss: {avg_loss:.4f} | TAcc: {train_acc:.4f} | VLoss: {val_loss:.4f} | VAcc: {val_acc:.4f}")

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.models_dir / 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:  # Increased early stopping patience
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(self.models_dir / 'best_model.pth', weights_only=True))

    def evaluate(self, test_data):
        """Evaluate and generate ALL figures"""
        print("\n" + "="*60 + "\nGENERATING FIGURES & METRICS\n" + "="*60)

        X_test, y_test = test_data
        X_test = torch.FloatTensor(X_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.max(outputs, 1)[1].cpu().numpy()

        # Metrics
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro')
        cm = confusion_matrix(y_test, preds)

        print(f"Test Accuracy: {acc*100:.2f}%")

        # --- GENERATE PLOTS ---
        print("1. Plotting Training Curves...")
        self._plot_training_curves()

        print("2. Plotting Confusion Matrix...")
        self._plot_confusion_matrix(cm)

        print("3. Plotting Per-Class Metrics...")
        self._plot_per_class_metrics(y_test, preds)

        print("4. Plotting ROC Curves...")
        self._plot_roc_curves(y_test, probs)

        print("5. Plotting Class Distribution...")
        self._plot_class_distribution()

        print("6. Plotting Error Analysis...")
        self._plot_error_analysis(y_test, preds)

        # Save Metrics JSON
        results = {
            'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        with open(self.metrics_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.save_model()
        print("\n✅ All outputs saved successfully!")

    # ================= PLOTTING METHODS =================

    def _plot_training_curves(self):
        """Smoothed training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss (with smoothing)
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']

        # Smooth curve if we have enough points
        if len(epochs) > 10:
            try:
                train_smooth = savgol_filter(train_loss, 9, 3)
                ax1.plot(epochs, train_smooth, '-', color='#2E86C1', alpha=1.0, linewidth=2, label='Train Loss (Smooth)')
                ax1.plot(epochs, train_loss, '-', color='#2E86C1', alpha=0.2) # Ghost line
            except:
                ax1.plot(epochs, train_loss, 'o-', color='#2E86C1', label='Train Loss')
        else:
            ax1.plot(epochs, train_loss, 'o-', color='#2E86C1', label='Train Loss')

        ax1.plot(epochs, val_loss, 's--', color='#E74C3C', linewidth=2, label='Validation Loss')

        ax1.set_title('Learning Progression (Loss)', fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross Entropy Loss')
        ax1.legend()

        # Accuracy
        ax2.plot(epochs, self.history['train_acc'], '-', color='#27AE60', linewidth=2, label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 's--', color='#F39C12', linewidth=2, label='Val Acc')

        ax2.set_title('Model Accuracy', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_training_curves.png')
        plt.close()

    def _plot_confusion_matrix(self, cm):
        """Beautiful Seaborn Heatmap"""
        plt.figure(figsize=(10, 8))

        # Calculate percentages
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100

        # Annotations: Count + Percentage
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    annot[i, j] = f"{c}\n({p:.1f}%)"
                else:
                    annot[i, j] = f"{c}\n({p:.1f}%)"

        labels = [self.label_map[i] for i in range(6)]

        sns.heatmap(cm_perc, annot=annot, fmt='', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Percentage (%)'},
                    annot_kws={"size": 9})

        plt.title('Confusion Matrix (Counts & %)', fontweight='bold', pad=20)
        plt.xlabel('Predicted State', labelpad=10)
        plt.ylabel('True State', labelpad=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_confusion_matrix.png')
        plt.close()

    def _plot_per_class_metrics(self, y_true, y_pred):
        """Bar chart for P/R/F1"""
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

        labels = [self.state_abbrev[i] for i in range(6)]
        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, p, width, label='Precision', color='#3498DB', edgecolor='white')
        plt.bar(x, r, width, label='Recall', color='#2ECC71', edgecolor='white')
        plt.bar(x + width, f1, width, label='F1 Score', color='#9B59B6', edgecolor='white')

        plt.title('Per-Class Performance Metrics', fontweight='bold')
        plt.xticks(x, labels)
        plt.ylim(0, 1.1)
        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_per_class_metrics.png')
        plt.close()

    def _plot_roc_curves(self, y_test, probs):
        """Multi-class ROC"""
        y_bin = label_binarize(y_test, classes=[0,1,2,3,4,5])
        n_classes = 6

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("bright", n_classes)

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{self.label_map[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Class', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig4_roc_curves.png')
        plt.close()

    def _plot_class_distribution(self):
        """Dataset Balance Plot"""
        plt.figure(figsize=(10, 6))

        states = [self.state_abbrev[i] for i in range(6)]

        # Data from prepare_data
        train_counts = self.class_dist['Train']
        val_counts = self.class_dist['Val']
        test_counts = self.class_dist['Test']

        # Normalize to percentage
        train_pct = train_counts / train_counts.sum() * 100
        val_pct = val_counts / val_counts.sum() * 100
        test_pct = test_counts / test_counts.sum() * 100

        x = np.arange(len(states))
        width = 0.25

        plt.bar(x - width, train_pct, width, label='Train', color='#34495E')
        plt.bar(x, val_pct, width, label='Validation', color='#95A5A6')
        plt.bar(x + width, test_pct, width, label='Test', color='#BDC3C7')

        plt.ylabel('Percentage of Split (%)')
        plt.title('Dataset Class Distribution (Balance Check)', fontweight='bold')
        plt.xticks(x, states)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig5_class_distribution.png')
        plt.close()

    def _plot_error_analysis(self, y_true, y_pred):
        """Top Confusion Pairs"""
        confusions = []
        for t, p in zip(y_true, y_pred):
            if t != p:
                confusions.append(f"{self.state_abbrev[t]} → {self.state_abbrev[p]}")

        if not confusions:
            return

        from collections import Counter
        counts = Counter(confusions).most_common(10)
        labels, values = zip(*counts)

        plt.figure(figsize=(10, 6))
        # FIX: Correct seaborn syntax to remove warning
        sns.barplot(x=list(values), y=list(labels), hue=list(labels), palette='Reds_r', legend=False)

        plt.title('Top 10 Misclassifications (Ground Truth → Predicted)', fontweight='bold')
        plt.xlabel('Number of Errors')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig6_error_analysis.png')
        plt.close()

    def save_model(self):
        """Save final artifact"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_map': self.label_map
        }, self.models_dir / 'trained_model.pth')


if __name__ == "__main__":
    # Import model architecture
    sys.path.insert(0, '.')
    from updated_htan import UpdatedHTAN, prepare_data_for_training_fixed

    print("🚀 Starting Research Training Pipeline...")

    # Load
    dataset_path = Path('datasets/improved_dataset.json')
    if not dataset_path.exists():
        print("❌ Run improved_datagen.py first!")
        sys.exit(1)

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Split
    splits = prepare_data_for_training_fixed(data, seq_len=5)

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UpdatedHTAN(input_dim=30, hidden_dim=128, num_classes=6, seq_len=5)

    # Run Pipeline
    pipeline = ResearchTrainingPipeline(model, device)
    splits_norm = pipeline.prepare_data(*splits)
    pipeline.train(splits_norm[0], splits_norm[1], epochs=50) # 50 is enough for this data
    pipeline.evaluate(splits_norm[2])