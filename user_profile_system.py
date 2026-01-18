"""
User Profile Management with Online Learning
Creates and updates user behavioral profiles
Implements anomaly detection and manual corrections
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import pickle


class UserProfile:
    """Manages user behavioral profile with online learning"""
    
    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.profile_dir = Path("user_profiles")
        self.profile_dir.mkdir(exist_ok=True)
        
        self.profile_path = self.profile_dir / f"{user_id}_profile.pkl"
        self.corrections_path = self.profile_dir / f"{user_id}_corrections.json"
        
        # Load or initialize
        self.baseline = None
        self.session_count = 0
        self.total_windows = 0
        self.correction_history = []
        
        # Online learning buffer
        self.recent_features = deque(maxlen=100)
        self.recent_labels = deque(maxlen=100)
        
        # Anomaly detection
        self.anomaly_threshold = 3.0  # Standard deviations
        self.anomaly_history = []
        
        # Load existing profile
        self.load_profile()
    
    def load_profile(self):
        """Load existing profile from disk"""
        if self.profile_path.exists():
            with open(self.profile_path, 'rb') as f:
                data = pickle.load(f)
                self.baseline = data['baseline']
                self.session_count = data['session_count']
                self.total_windows = data['total_windows']
                
                print(f"✓ Loaded profile for {self.user_id}")
                print(f"  Sessions: {self.session_count}")
                print(f"  Windows: {self.total_windows}")
                print(f"  Baseline established: {self.baseline is not None}")
        else:
            print(f"Creating new profile for {self.user_id}")
        
        # Load corrections
        if self.corrections_path.exists():
            with open(self.corrections_path, 'r') as f:
                self.correction_history = json.load(f)
    
    def save_profile(self):
        """Save profile to disk"""
        data = {
            'baseline': self.baseline,
            'session_count': self.session_count,
            'total_windows': self.total_windows,
            'user_id': self.user_id,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.profile_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save corrections
        with open(self.corrections_path, 'w') as f:
            json.dump(self.correction_history, f, indent=2)
    
    def add_window_data(self, features, predicted_label=None, true_label=None):
        """Add data from a window"""
        self.total_windows += 1
        
        # Store in buffer
        self.recent_features.append(features)
        if true_label is not None:
            self.recent_labels.append(true_label)
        elif predicted_label is not None:
            self.recent_labels.append(predicted_label)
        
        # Update baseline if we have enough data
        if len(self.recent_features) >= 20:
            self._update_baseline()
    
    def _update_baseline(self):
        """Update behavioral baseline"""
        features_array = np.array(list(self.recent_features))
        
        if self.baseline is None:
            # Create initial baseline
            self.baseline = {
                'mean': np.mean(features_array, axis=0),
                'std': np.std(features_array, axis=0) + 1e-6,
                'min': np.min(features_array, axis=0),
                'max': np.max(features_array, axis=0),
                'sample_count': len(features_array)
            }
            print(f"\n✓ User baseline created from {len(features_array)} windows")
        else:
            # Incremental update
            old_count = self.baseline['sample_count']
            new_count = old_count + len(features_array)
            
            # Update mean
            old_mean = self.baseline['mean']
            new_data_mean = np.mean(features_array, axis=0)
            updated_mean = (old_mean * old_count + new_data_mean * len(features_array)) / new_count
            
            # Update std
            old_var = self.baseline['std'] ** 2
            new_var = np.var(features_array, axis=0)
            updated_var = (old_var * old_count + new_var * len(features_array)) / new_count
            updated_std = np.sqrt(updated_var) + 1e-6
            
            self.baseline['mean'] = updated_mean
            self.baseline['std'] = updated_std
            self.baseline['min'] = np.minimum(self.baseline['min'], np.min(features_array, axis=0))
            self.baseline['max'] = np.maximum(self.baseline['max'], np.max(features_array, axis=0))
            self.baseline['sample_count'] = new_count
            
            print(f"Updated baseline (now {new_count} windows)")
        
        # Save after update
        self.save_profile()
    
    def detect_anomaly(self, features):
        """Detect if features are anomalous"""
        if self.baseline is None:
            return {
                'is_anomaly': False,
                'severity': 'building_baseline',
                'score': 0.0,
                'anomalous_features': []
            }
        
        features_array = np.array(features)
        
        # Calculate z-scores
        z_scores = np.abs((features_array - self.baseline['mean']) / self.baseline['std'])
        
        # Find anomalous features
        anomalous_indices = np.where(z_scores > self.anomaly_threshold)[0]
        anomaly_score = float(np.mean(z_scores))
        
        # Determine severity
        num_anomalous = len(anomalous_indices)
        
        if num_anomalous > 15:
            severity = 'critical'
            is_anomaly = True
        elif num_anomalous > 10:
            severity = 'high'
            is_anomaly = True
        elif num_anomalous > 5:
            severity = 'moderate'
            is_anomaly = True
        else:
            severity = 'normal'
            is_anomaly = False
        
        result = {
            'is_anomaly': is_anomaly,
            'severity': severity,
            'score': anomaly_score,
            'anomalous_features': anomalous_indices.tolist(),
            'num_anomalous': num_anomalous,
            'timestamp': datetime.now().isoformat()
        }
        
        if is_anomaly:
            self.anomaly_history.append(result)
        
        return result
    
    def add_correction(self, predicted, actual, features, metadata):
        """Record manual correction for learning"""
        correction = {
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted,
            'actual': actual,
            'window_number': self.total_windows,
            'session': self.session_count,
            'metadata': metadata
        }
        
        self.correction_history.append(correction)
        
        # Add to recent labels with correct label
        if len(self.recent_features) > 0:
            # Update the most recent label
            if len(self.recent_labels) > 0:
                self.recent_labels[-1] = actual
        
        print(f"✓ Correction recorded: {predicted} → {actual}")
        
        self.save_profile()
        
        return correction
    
    def get_correction_stats(self):
        """Get statistics about corrections"""
        if not self.correction_history:
            return None
        
        # Count by predicted state
        prediction_errors = {}
        for correction in self.correction_history:
            pred = correction['predicted']
            if pred not in prediction_errors:
                prediction_errors[pred] = {'count': 0, 'corrections_to': {}}
            
            prediction_errors[pred]['count'] += 1
            
            actual = correction['actual']
            if actual not in prediction_errors[pred]['corrections_to']:
                prediction_errors[pred]['corrections_to'][actual] = 0
            prediction_errors[pred]['corrections_to'][actual] += 1
        
        return {
            'total_corrections': len(self.correction_history),
            'by_predicted_state': prediction_errors,
            'correction_rate': len(self.correction_history) / max(self.total_windows, 1)
        }
    
    def start_session(self):
        """Mark start of new session"""
        self.session_count += 1
        print(f"\n{'='*60}")
        print(f"Session #{self.session_count} started")
        print(f"{'='*60}")
    
    def end_session(self):
        """Mark end of session and save"""
        self.save_profile()
        
        print(f"\n{'='*60}")
        print(f"Session #{self.session_count} ended")
        print(f"  Windows this session: {self.total_windows}")
        
        if self.correction_history:
            stats = self.get_correction_stats()
            print(f"  Total corrections: {stats['total_corrections']}")
            print(f"  Correction rate: {stats['correction_rate']*100:.1f}%")
        
        print(f"{'='*60}\n")


class OnlineLearningManager:
    """Manages online learning and model updates"""
    
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Correction buffer for batch updates
        self.correction_buffer = []
        self.update_threshold = 10  # Update after 10 corrections
    
    def add_correction(self, features_sequence, correct_label):
        """Add corrected sample for learning"""
        self.correction_buffer.append((features_sequence, correct_label))
        
        # Perform update if threshold reached
        if len(self.correction_buffer) >= self.update_threshold:
            self.perform_update()
    
    def perform_update(self):
        """Update model with corrected samples"""
        if not self.correction_buffer:
            return
        
        print(f"\nPerforming online learning update with {len(self.correction_buffer)} corrections...")
        
        self.model.train()
        total_loss = 0
        
        for features_seq, label in self.correction_buffer:
            # Convert to tensor
            x = torch.FloatTensor(features_seq).unsqueeze(0).to(self.device)
            y = torch.LongTensor([label]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.correction_buffer)
        print(f"✓ Update complete. Average loss: {avg_loss:.4f}")
        
        # Clear buffer
        self.correction_buffer.clear()
        
        self.model.eval()


import torch

if __name__ == "__main__":
    # Test profile creation
    profile = UserProfile("test_user")
    
    # Simulate adding data
    for i in range(30):
        fake_features = np.random.randn(30)
        profile.add_window_data(fake_features, predicted_label=0)
    
    # Test anomaly detection
    normal_features = np.random.randn(30)
    anomaly = profile.detect_anomaly(normal_features)
    print(f"\nAnomaly check: {anomaly}")
    
    # Test correction
    profile.add_correction("focused_work", "research_reading", normal_features, {})
    
    # End session
    profile.end_session()
