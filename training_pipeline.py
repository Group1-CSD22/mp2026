"""
Complete Training Pipeline for Improved HTAN
Guaranteed to work with new dataset and model
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

class ImprovedTrainingPipeline:
    """Complete training pipeline with proper error handling"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch': []
        }
        
        self.label_map = {
            0: 'focused_work',
            1: 'task_execution',
            2: 'research_reading',
            3: 'meeting_communication',
            4: 'unproductive'
        }
        
        print(f"✓ Pipeline initialized on device: {device}")
    
    def prepare_data(self, sequences, labels, test_size=0.2, val_size=0.2):
        """Prepare and normalize data"""
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)
        
        n_samples, seq_len, n_features = sequences.shape
        print(f"\nDataset shape: {sequences.shape}")
        print(f"  Samples: {n_samples}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Features: {n_features}")
        
        # Flatten for normalization
        sequences_flat = sequences.reshape(-1, n_features)
        print(f"\nNormalizing features...")
        sequences_flat = self.scaler.fit_transform(sequences_flat)
        sequences = sequences_flat.reshape(n_samples, seq_len, n_features)
        
        # Split data
        print(f"\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=42,
            stratify=y_train
        )
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Print class distribution
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  {self.label_map[cls]}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, train_data, val_data, epochs=100, batch_size=32, lr=0.001):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Optimizer: Adam")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=8, factor=0.5
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0

        print("\nStarting training...\n")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0

            # Shuffle training data
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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == y_batch).sum().item()
                num_batches += 1

            train_loss /= num_batches
            train_acc = train_correct / len(X_train)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_acc = (val_predicted == y_val).sum().item() / len(y_val)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss.item())
            self.history['val_acc'].append(val_acc)
            self.history['epoch'].append(epoch + 1)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss or val_acc > best_val_acc:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("\n✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")

    def evaluate(self, test_data):
        """Comprehensive evaluation"""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)

        X_test, y_test = test_data
        X_test = torch.FloatTensor(X_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions_np, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(y_test, predictions_np, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions_np)

        # Print results
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print(f"\nPer-Class Performance:")
        print(f"{'State':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)
        for i in range(5):
            state = self.label_map[i]
            print(f"{state:<25} {precision_per_class[i]:>10.4f} {recall_per_class[i]:>10.4f} "
                  f"{f1_per_class[i]:>10.4f} {support[i]:>10}")

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.label_map[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support[i])
                }
                for i in range(5)
            }
        }

        # Save results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Results saved to evaluation_results.json")

        return results, cm

    def plot_training_history(self):
        """Plot training history"""
        print("\nGenerating training history plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = self.history['epoch']

        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', markersize=4)
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], label='Train Acc', marker='o', markersize=4)
        ax2.plot(epochs, self.history['val_acc'], label='Val Acc', marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved training_history.png")

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        print("Generating confusion matrix plot...")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.label_map[i] for i in range(5)],
            yticklabels=[self.label_map[i] for i in range(5)],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved confusion_matrix.png")

    def save_model(self):
        """Save complete model package"""
        print("\nSaving model package...")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history,
            'label_map': self.label_map
        }, 'trained_model.pth')

        print("✓ Saved trained_model.pth")


def main():
    """Main training script"""
    print("\n" + "="*60)
    print("WORK MONITORING SYSTEM - MODEL TRAINING")
    print("="*60)

    # Check for dataset
    if not os.path.exists('work_monitoring_data.json'):
        print("\nERROR: Dataset not found!")
        print("Please run: python synthetic_data_gen.py")
        return

    # Load dataset
    print("\nLoading dataset...")
    with open('work_monitoring_data.json', 'r') as f:
        dataset = json.load(f)
    print(f"✓ Loaded {len(dataset)} samples")

    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from htan_model import prepare_data_for_training, ImprovedHTAN

    # Prepare data
    sequences, labels = prepare_data_for_training(dataset, seq_len=5)

    # Create model with correct input dimension
    print("\nCreating model...")
    input_dim = sequences.shape[2]  # Get actual feature count from data
    print(f"Model input dimension: {input_dim}")

    model = ImprovedHTAN(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=5,
        seq_len=5
    )

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create pipeline
    pipeline = ImprovedTrainingPipeline(model, device=device)

    # Prepare data splits
    train_data, val_data, test_data = pipeline.prepare_data(sequences, labels)

    # Train model
    pipeline.train(train_data, val_data, epochs=100, batch_size=32, lr=0.001)

    # Evaluate
    results, cm = pipeline.evaluate(test_data)

    # Generate visualizations
    pipeline.plot_training_history()
    pipeline.plot_confusion_matrix(cm)

    # Save model
    pipeline.save_model()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ best_model.pth")
    print("  ✓ trained_model.pth")
    print("  ✓ training_history.png")
    print("  ✓ confusion_matrix.png")
    print("  ✓ evaluation_results.json")
    print("\nYou can now run the monitoring system!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)