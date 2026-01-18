"""
Research-Grade Training Pipeline
Generates publication-quality figures and comprehensive metrics
Follows IEEE and Springer standards
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, 
                            roc_curve, auc, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler, label_binarize
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys


# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class ResearchTrainingPipeline:
    """Training pipeline with research-grade outputs"""
    
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
            0: 'Deep Work',
            1: 'Active Work',
            2: 'Research',
            3: 'Communication',
            4: 'Distracted',
            5: 'Idle'
        }
        
        self.state_abbrev = {
            0: 'DW', 1: 'AW', 2: 'RS', 
            3: 'CM', 4: 'DS', 5: 'ID'
        }
        
        # Output directories
        self.output_dir = Path('research_outputs')
        self.figures_dir = self.output_dir / 'figures'
        self.metrics_dir = self.output_dir / 'metrics'
        self.models_dir = self.output_dir / 'models'
        
        for dir in [self.output_dir, self.figures_dir, self.metrics_dir, self.models_dir]:
            dir.mkdir(exist_ok=True, parents=True)
        
        print(f"✓ Pipeline initialized on {device}")
        print(f"✓ Outputs will be saved to: {self.output_dir}")
    
    def prepare_data(self, sequences, labels, test_size=0.10, val_size=0.10):
        """Prepare with smaller test set for more training data"""
        print("\n" + "="*70)
        print("DATA PREPARATION")
        print("="*70)

        n_samples, seq_len, n_features = sequences.shape
        print(f"\nDataset: {sequences.shape}")
        print(f"  Samples: {n_samples:,}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Features: {n_features}")

        # Normalize
        print("\nNormalizing features...")
        sequences_flat = sequences.reshape(-1, n_features)
        sequences_flat = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_flat.reshape(n_samples, seq_len, n_features)

        # Stratified split (smaller test set = more training data)
        print("Splitting data (stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_train
        )

        print(f"\nSplit sizes:")
        print(f"  Train: {len(X_train):,} ({len(X_train)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} ({len(X_val)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} ({len(X_test)/n_samples*100:.1f}%)")

        # Class distribution
        print(f"\nClass distribution (training):")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {self.label_map[cls]:15s}: {count:6,} ({count/len(y_train)*100:5.1f}%)")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self, train_data, val_data, epochs=150, batch_size=64, lr=0.0008):
        """Enhanced training for 90%+ accuracy"""
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)

        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Optimizer: Adam (weight_decay=1e-5)")
        print(f"  Scheduler: CosineAnnealingLR with warmup")

        # Class weights for balanced learning
        class_counts = torch.bincount(y_train)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 6

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 25
        patience_counter = 0

        print("\nTraining progress:")
        print("-" * 70)

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0

            indices = torch.randperm(len(X_train))
            num_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == y_batch).sum().item()
                num_batches += 1

            train_loss /= num_batches
            train_acc = train_correct / len(X_train)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val).sum().item() / len(y_val)

            # LR scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss.item())
            self.history['val_acc'].append(val_acc)
            self.history['epoch'].append(epoch + 1)
            self.history['lr'].append(current_lr)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"TLoss: {train_loss:.4f} | TAcc: {train_acc:.4f} | "
                      f"VLoss: {val_loss:.4f} | VAcc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")

            # Save best model
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_loss = val_loss
                best_val_acc = val_acc

                torch.save(self.model.state_dict(), self.models_dir / 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(self.models_dir / 'best_model.pth'))

        print("-" * 70)
        print(f"\n✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    def evaluate(self, test_data):
        """Comprehensive evaluation with all metrics"""
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        X_test, y_test = test_data
        X_test = torch.FloatTensor(X_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, predictions_np)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, predictions_np, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, predictions_np, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(y_test, predictions_np, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions_np)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, predictions_np)
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision (macro): {precision_macro:.4f}")
        print(f"  Recall (macro):    {recall_macro:.4f}")
        print(f"  F1-Score (macro):  {f1_macro:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  Matthews Corr:     {mcc:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'State':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        for i in range(6):
            state = self.label_map[i]
            print(f"{state:<15} {precision_per_class[i]:>10.4f} {recall_per_class[i]:>10.4f} "
                  f"{f1_per_class[i]:>10.4f} {support[i]:>10}")
        
        # Save results
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'mcc': float(mcc)
            },
            'per_class': {
                self.label_map[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support[i])
                }
                for i in range(6)
            },
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.metrics_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {self.metrics_dir / 'evaluation_results.json'}")
        
        return results, cm, probabilities_np, predictions_np
    
    def generate_all_figures(self, cm, probabilities, predictions, y_test):
        """Generate all publication-quality figures"""
        print("\n" + "="*70)
        print("GENERATING FIGURES")
        print("="*70)
        
        self._plot_training_curves()
        self._plot_confusion_matrix(cm)
        self._plot_per_class_metrics()
        self._plot_roc_curves(probabilities, y_test)
        self._plot_precision_recall_curves(probabilities, y_test)
        self._plot_class_distribution()
        self._plot_learning_rate_schedule()
        self._plot_error_analysis(predictions, y_test)
        
        print(f"\n✓ All figures saved to {self.figures_dir}/")
    
    def _plot_training_curves(self):
        """Figure 1: Training and validation curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        epochs = self.history['epoch']
        
        # Loss curves
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training', linewidth=1.5)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('(a) Training and Validation Loss')
        ax1.legend(frameon=False)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Accuracy curves
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training', linewidth=1.5)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation', linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('(b) Training and Validation Accuracy')
        ax2.legend(frameon=False)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1])
        
        # Loss difference
        loss_diff = np.array(self.history['train_loss']) - np.array(self.history['val_loss'])
        ax3.plot(epochs, loss_diff, 'g-', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training - Validation Loss')
        ax3.set_title('(c) Overfitting Indicator')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Learning rate
        ax4.plot(epochs, self.history['lr'], 'purple', linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('(d) Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_training_curves.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig1_training_curves.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 1: Training curves")
    
    def _plot_confusion_matrix(self, cm):
        """Figure 2: Confusion matrix"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=[self.state_abbrev[i] for i in range(6)],
                   yticklabels=[self.label_map[i] for i in range(6)],
                   cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted State')
        ax1.set_ylabel('True State')
        ax1.set_title('(a) Confusion Matrix (Counts)')
        
        # Normalized (percentages)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=[self.state_abbrev[i] for i in range(6)],
                   yticklabels=[self.label_map[i] for i in range(6)],
                   cbar_kws={'label': 'Percentage'})
        ax2.set_xlabel('Predicted State')
        ax2.set_ylabel('True State')
        ax2.set_title('(b) Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_confusion_matrix.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig2_confusion_matrix.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 2: Confusion matrix")
    
    def _plot_per_class_metrics(self):
        """Figure 3: Per-class performance"""
        # Load results
        with open(self.metrics_dir / 'evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        states = [self.label_map[i] for i in range(6)]
        precision = [results['per_class'][s]['precision'] for s in states]
        recall = [results['per_class'][s]['recall'] for s in states]
        f1 = [results['per_class'][s]['f1_score'] for s in states]
        
        x = np.arange(len(states))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')
        
        ax.set_xlabel('State')
        ax.set_ylabel('Score')
        ax.set_title('Per-State Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=15, ha='right')
        ax.legend(frameon=False)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=7)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_per_class_metrics.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig3_per_class_metrics.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 3: Per-class metrics")
    
    def _plot_roc_curves(self, probabilities, y_test):
        """Figure 4: ROC curves"""
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Calculate ROC curve for each class
        for i in range(6):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                   label=f'{self.label_map[i]} (AUC = {roc_auc:.3f})')
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig4_roc_curves.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig4_roc_curves.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 4: ROC curves")
    
    def _plot_precision_recall_curves(self, probabilities, y_test):
        """Figure 5: Precision-Recall curves"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i in range(6):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], probabilities[:, i])
            ap = average_precision_score(y_test_bin[:, i], probabilities[:, i])
            
            ax.plot(recall, precision, color=colors[i], linewidth=1.5,
                   label=f'{self.label_map[i]} (AP = {ap:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left', frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig5_precision_recall.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig5_precision_recall.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 5: Precision-Recall curves")
    
    def _plot_class_distribution(self):
        """Figure 6: Class distribution in dataset"""
        with open(self.metrics_dir / 'evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        states = [self.label_map[i] for i in range(6)]
        counts = [results['per_class'][s]['support'] for s in states]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bars = ax.bar(states, counts, color='#4CAF50', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('State')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Test Set Class Distribution')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig6_class_distribution.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig6_class_distribution.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 6: Class distribution")
    
    def _plot_learning_rate_schedule(self):
        """Figure 7: Learning rate evolution"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(self.history['epoch'], self.history['lr'], 'b-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig7_lr_schedule.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig7_lr_schedule.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 7: LR schedule")
    
    def _plot_error_analysis(self, predictions, y_test):
        """Figure 8: Error analysis"""
        errors = predictions != y_test
        error_states = y_test[errors]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Errors by true state
        unique_states, counts = np.unique(error_states, return_counts=True)
        states = [self.label_map[i] for i in unique_states]
        
        ax1.barh(states, counts, color='#E53935', alpha=0.7)
        ax1.set_xlabel('Number of Misclassifications')
        ax1.set_title('(a) Misclassifications by True State')
        ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Error rate by state
        error_rates = []
        for i in range(6):
            mask = y_test == i
            if mask.sum() > 0:
                error_rate = (predictions[mask] != y_test[mask]).sum() / mask.sum()
                error_rates.append(error_rate * 100)
            else:
                error_rates.append(0)
        
        states = [self.label_map[i] for i in range(6)]
        ax2.bar(states, error_rates, color='#FFA726', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Error Rate (%)')
        ax2.set_title('(b) Error Rate by State')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig8_error_analysis.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'fig8_error_analysis.png', bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 8: Error analysis")
    
    def save_model(self):
        """Save complete model package"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history,
            'label_map': self.label_map
        }, self.models_dir / 'trained_model.pth')
        
        print(f"\n✓ Model saved to {self.models_dir / 'trained_model.pth'}")


def main():
    """Main training script"""
    import sys
    sys.path.insert(0, '.')
    
    from updated_htan import UpdatedHTAN, prepare_data_for_training
    
    print("\n" + "="*70)
    print("RESEARCH-GRADE TRAINING PIPELINE")
    print("="*70)
    
    # Load dataset
    dataset_path = Path('datasets/improved_dataset.json')
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("   Run: python improved_datagen.py")
        return
    
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"✓ Loaded {len(dataset):,} samples")
    
    # Prepare data
    sequences, labels = prepare_data_for_training(dataset, seq_len=5)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nCreating model on {device}...")
    
    model = UpdatedHTAN(
        input_dim=30,
        hidden_dim=128,
        num_classes=6,
        seq_len=5,
        dropout=0.3
    )
    
    # Create pipeline
    pipeline = ResearchTrainingPipeline(model, device=device)
    
    # Prepare data
    train_data, val_data, test_data = pipeline.prepare_data(sequences, labels)
    
    # Train
    pipeline.train(train_data, val_data, epochs=150, batch_size=64, lr=0.0008)
    
    # Evaluate
    results, cm, probs, preds = pipeline.evaluate(test_data)
    
    # Generate all figures
    pipeline.generate_all_figures(cm, probs, preds, test_data[1])
    
    # Save model
    pipeline.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {pipeline.output_dir}/")
    print("  - Figures (PDF + PNG): research_outputs/figures/")
    print("  - Metrics (JSON): research_outputs/metrics/")
    print("  - Models: research_outputs/models/")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)