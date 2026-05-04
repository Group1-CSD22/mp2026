"""
COMPLETE LIVE MONITORING SYSTEM v3.0
Fully integrated: data collection, prediction, learning, corrections
"""

import torch
import numpy as np
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
import sys

# Check dependencies
try:
    from pynput import keyboard, mouse
    import psutil
    import win32gui
    import win32process
    LIVE_MODE = True
except ImportError:
    LIVE_MODE = False
    print("⚠️  Install: pip install pynput psutil pywin32")

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from live_monitor import EnhancedDataCollector, AppCategorizer
from user_profile_system import UserProfile, OnlineLearningManager
from updated_htan import UpdatedHTAN, OnlineLearningOptimizer


class CompleteLiveMonitor:
    """Complete integrated monitoring system"""
    
    def __init__(self, user_id="default_user", model_path='research_outputs/models/trained_model.pth'):
        print("\n" + "="*80)
        print(" "*20 + "LIVE WORK MONITORING SYSTEM v3.0")
        print("="*80)
        
        if not LIVE_MODE:
            print("\n❌ Cannot run in live mode. Install dependencies.")
            return
        
        self.user_id = user_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model, self.scaler, self.online_learner = self._load_model(model_path)
        
        if self.model is None:
            print("\n❌ Model not found! Train first:")
            print("   python research_training.py")
            return
        
        # Initialize components
        self.collector = EnhancedDataCollector(window_size=60)  # 1 minute
        self.user_profile = UserProfile(user_id)
        
        # State mapping
        self.label_map = {
            0: 'Deep Work',
            1: 'Active Work',
            2: 'Research',
            3: 'Communication',
            4: 'Distracted',
            5: 'Idle'
        }
        
        self.state_to_label = {v: k for k, v in self.label_map.items()}
        
        # Monitoring state
        self.is_monitoring = False
        self.feature_buffer = deque(maxlen=5)  # 5-minute history
        self.window_count = 0
        
        # Session tracking
        self.session_start = None
        self.session_predictions = []
        
        # Correction mode
        self.correction_mode = True  # Enable manual corrections
        
        # Session logging
        self.session_dir = Path("session_logs/live")
        self.session_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n✓ System initialized for user: {user_id}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Correction mode: {'ON' if self.correction_mode else 'OFF'}")
        print(f"✓ Profile status: {'Loaded' if self.user_profile.baseline else 'New'}")

        self.stop_event = threading.Event()
        self.correction_event = threading.Event()  # NEW: Pauses the thread
        self.waiting_for_correction = False  # NEW: Tells the UI we are paused
        self.ui_mode = False  # NEW: Tells backend a UI is controlling it
        self.supervised_mode = False  # NEW: Toggled by UI
    
    def _load_model(self, path):
        """Load trained model"""
        model_path = Path(path)
        
        if not model_path.exists():
            return None, None, None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            model = UpdatedHTAN(
                input_dim=30,
                hidden_dim=128,
                num_classes=6,
                seq_len=5
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            scaler = checkpoint['scaler']
            
            # Create online learner
            online_learner = OnlineLearningOptimizer(model, learning_rate=0.0003, device=self.device)
            
            print(f"✓ Model loaded from {model_path}")
            return model, scaler, online_learner
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    
    def start(self):
        """Start live monitoring"""
        if not LIVE_MODE or self.model is None:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.session_start = datetime.now()
        self.user_profile.start_session()
        
        print("\n" + "="*80)
        print(f"MONITORING STARTED: {self.session_start.strftime('%I:%M %p')}")
        print("="*80)
        print("Window: 1 minute | Predictions: Every minute (after 5-min warm-up)")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        try:
            while not self.stop_event.is_set():
                # Collect 1-minute window
                elapsed = (datetime.now() - self.session_start).seconds // 60
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting window #{self.window_count + 1}... ")
                
                self.collector.start_collection()
                stopped_early = self.stop_event.wait(60)
                self.collector.stop_collection()

                if stopped_early:
                    break
                
                # Get features
                features, metadata = self.collector.get_features()
                
                # Add to buffer
                self.feature_buffer.append(features)
                self.window_count += 1
                
                print(f"✓ ({elapsed + 1} min elapsed)")
                
                # Check if we can predict
                if len(self.feature_buffer) < 5:
                    remaining = 5 - len(self.feature_buffer)
                    print(f"   Building history... {remaining} more window(s) needed\n")
                    continue

                if self.supervised_mode and self.ui_mode:
                    print("\n⏸️ Supervised Mode: Paused. Waiting for UI confirmation...")
                    self.waiting_for_correction = True
                    self.correction_event.clear()
                    self.correction_event.wait()  # THREAD FREEZES HERE UNTIL 'APPLY' IS CLICKED
                    self.waiting_for_correction = False
                    print("▶️ Resuming data collection...")
                
                # Make prediction
                self._process_prediction(features, metadata)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring stopped by user")
        finally:
            self._end_session()

    def _process_prediction(self, current_features, metadata):
        """Make prediction and handle corrections"""
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        seq_flat = sequence.reshape(-1, 30)
        seq_norm = self.scaler.transform(seq_flat)
        sequence_norm = seq_norm.reshape(1, 5, 30)

        predicted_label, probabilities, confidence = self.online_learner.predict(sequence_norm)
        predicted_state = self.label_map[predicted_label]
        anomaly_result = self.user_profile.detect_anomaly(current_features)

        self._display_prediction(predicted_state, confidence, probabilities, current_features, metadata, anomaly_result)

        prediction_record = {
            'window': self.window_count,
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_state,
            'confidence': confidence,
            'probabilities': {self.label_map[i]: float(probabilities[i]) for i in range(6)},
            'metadata': metadata,
            'anomaly': anomaly_result
        }
        self.session_predictions.append(prediction_record)

        # --- UI SAFE CORRECTION LOGIC ---
        actual_state = predicted_state

        if self.correction_mode and not self.ui_mode:
            actual_state = self._ask_for_correction(predicted_state, current_features, metadata)
            # Add window data handles profile update
            actual_label = self.state_to_label[actual_state]
            self.user_profile.add_window_data(current_features, predicted_label, actual_label)

        elif self.ui_mode:
            actual_label = self.state_to_label[actual_state]
            self.user_profile.add_window_data(current_features, predicted_label, actual_label)

            # Store data so apply_ui_correction can use it!
            self.latest_correction_data = {
                'features': current_features,
                'metadata': metadata,
                'predicted_state': predicted_state,
                'predicted_label': predicted_label,
                'sequence_norm': sequence_norm
            }

    def _display_prediction(self, state, confidence, probs, features, metadata, anomaly):
        """Display prediction results"""
        print("\n" + "=" * 80)
        print(f"PREDICTION #{self.window_count}")
        print("=" * 80)

        print(f"\n📊 STATE: {state.upper()}")
        print(f"   Confidence: {confidence * 100:.1f}%")

        prob_dict = {self.label_map[i]: probs[i] for i in range(6)}
        top_3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n   Top predictions:")
        for i, (s, p) in enumerate(top_3, 1):
            bar = '█' * int(p * 30)
            print(f"   {i}. {s:15s} {bar:30s} {p * 100:5.1f}%")

        # --- YOUR METRICS ARE BACK! ---
        print(f"\n📈 WINDOW METRICS:")
        print(f"   Typing: {features[2]:.0f} WPM | {int(features[0])} keys")
        print(f"   Mouse: {features[8]:.0f}px moved | {int(features[17])} clicks")
        print(f"   Apps: {int(features[24])} switches | Productivity: {features[26] * 100:.0f}%")

        if metadata.get('current_app'):
            print(f"   Current app: {metadata['current_app']} ({metadata.get('app_category', 'unknown')})")

        # (Anomaly printing is kept hidden for UI cleanliness)

        elapsed = (datetime.now() - self.session_start).seconds // 60
        print(f"\n⏱️  Session: {elapsed} min | Windows: {self.window_count}")
        print("=" * 80)
    
    def _ask_for_correction(self, predicted_state, features, metadata):
        """Ask user for manual correction"""
        print(f"\n❓ Correction mode:")
        print(f"   Predicted: {predicted_state}")
        
        response = input("   Is this correct? (y/n): ").strip().lower()
        
        if response == 'n':
            print(f"\n   What was the actual state?")
            for i, state in self.label_map.items():
                print(f"   {i}: {state}")
            
            choice = input("   Enter number (0-5): ").strip()
            
            try:
                choice_num = int(choice)
                if 0 <= choice_num <= 5:
                    actual_state = self.label_map[choice_num]
                    
                    # Record correction
                    self.user_profile.add_correction(
                        predicted_state,
                        actual_state,
                        features,
                        metadata
                    )
                    
                    print(f"   ✓ Correction recorded: {predicted_state} → {actual_state}")
                    return actual_state
            except:
                pass
        
        return predicted_state
    
    def _end_session(self):
        """End monitoring session"""
        self.is_monitoring = False
        self.collector.stop_collection()
        
        # End user profile session
        self.user_profile.end_session()
        
        # Save session log
        self._save_session()
        
        # Print summary
        self._print_summary()
    
    def _save_session(self):
        """Save session data"""
        session_file = self.session_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            'user_id': self.user_id,
            'start_time': self.session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.session_start).seconds // 60,
            'total_windows': self.window_count,
            'predictions': self.session_predictions,
            'profile_stats': {
                'session_number': self.user_profile.session_count,
                'total_windows': self.user_profile.total_windows,
                'has_baseline': self.user_profile.baseline is not None
            }
        }
        
        if self.user_profile.correction_history:
            stats = self.user_profile.get_correction_stats()
            session_data['corrections'] = stats
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n✓ Session saved to {session_file}")
    
    def _print_summary(self):
        """Print session summary"""
        duration = (datetime.now() - self.session_start).seconds // 60
        
        print("\n" + "="*80)
        print(" "*30 + "SESSION SUMMARY")
        print("="*80)
        
        print(f"\nUser: {self.user_id}")
        print(f"Duration: {duration} minutes")
        print(f"Windows collected: {self.window_count}")
        
        if self.session_predictions:
            # State distribution
            state_counts = {}
            for pred in self.session_predictions:
                state = pred['predicted']
                state_counts[state] = state_counts.get(state, 0) + 1
            
            print(f"\nState Distribution:")
            for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.session_predictions)) * 100
                print(f"  {state:20s}: {count:3d} ({pct:5.1f}%)")
            
            # Average confidence
            avg_conf = np.mean([p['confidence'] for p in self.session_predictions])
            print(f"\nAverage Confidence: {avg_conf*100:.1f}%")
        
        # Corrections
        if self.user_profile.correction_history:
            stats = self.user_profile.get_correction_stats()
            print(f"\nCorrections: {stats['total_corrections']}")
            print(f"Correction Rate: {stats['correction_rate']*100:.1f}%")
        
        # Profile status
        print(f"\nProfile Status:")
        print(f"  Session: #{self.user_profile.session_count}")
        print(f"  Total windows: {self.user_profile.total_windows}")
        print(f"  Baseline: {'Established' if self.user_profile.baseline else 'Building'}")
        
        print("\n" + "="*80 + "\n")

    def stop(self):
        """Thread-safe stop trigger"""
        print("\nStopping monitor via UI...")
        self.stop_event.set()

    def apply_ui_correction(self, actual_state):
        """Called safely by the UI when Supervised Mode is ON"""
        # ALWAYS unpause the thread, even if it fails
        if hasattr(self, 'correction_event'):
            self.correction_event.set()

        if not hasattr(self, 'latest_correction_data') or not self.latest_correction_data:
            return False

        data = self.latest_correction_data
        predicted_state = data['predicted_state']

        # If user just confirms the state is correct
        if actual_state == predicted_state:
            print(f"✓ State confirmed as: {actual_state}")
            return True

        actual_label = self.state_to_label[actual_state]

        # Record correction
        self.user_profile.add_correction(predicted_state, actual_state, data['features'], data['metadata'])
        self.online_learner.add_experience(data['sequence_norm'][0], actual_label)

        # Print progress to terminal!
        current_buffer = len(self.online_learner.memory)
        print(f"✓ Correction recorded: {predicted_state} -> {actual_state} (Buffer: {current_buffer}/10)")

        if current_buffer >= 10:
            loss = self.online_learner.update(batch_size=8, epochs=3)
            print(f"\n   🔄 Model updated via UI (loss: {loss:.4f})")

        return True


def run_test_mode(duration_minutes=10):
    """Run quick test with simulated data"""
    print("\n" + "=" * 80)
    print(" " * 25 + "TEST MODE (Simulated Data)")
    print("=" * 80)

    from improved_datagen import ImprovedDatasetGenerator

    gen = ImprovedDatasetGenerator()
    test_dir = Path("session_logs/test")
    test_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    model_path = Path('research_outputs/models/trained_model.pth')
    if not model_path.exists():
        print("\n❌ Model not found. Train first:")
        print("   python research_training.py")
        return

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = UpdatedHTAN(input_dim=30, hidden_dim=128, num_classes=6, seq_len=5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']

    label_map = {
        0: 'Deep Work', 1: 'Active Work', 2: 'Research',
        3: 'Communication', 4: 'Distracted', 5: 'Idle'
    }

    print(f"\nRunning {duration_minutes}-minute test...\n")

    num_windows = duration_minutes
    feature_buffer = deque(maxlen=5)
    predictions = []

    for i in range(num_windows):
        state = np.random.choice(list(gen.states.keys()))
        sample = gen.generate_sample(state)

        feature_names = [
            'keystroke_count', 'keystroke_rate', 'typing_speed_wpm',
            'keystroke_variance', 'keystroke_burst_ratio', 'avg_inter_keystroke',
            'alpha_ratio', 'mouse_movement_count', 'total_distance',
            'avg_distance_per_move', 'coverage_score', 'x_variance',
            'y_variance', 'direction_changes', 'is_idle', 'click_count',
            'click_rate', 'avg_click_interval', 'scroll_count', 'scroll_rate',
            'total_scroll', 'app_switch_count', 'unique_apps',
            'app_productivity', 'focus_stability', 'productive_ratio',
            'digit_ratio', 'special_ratio', 'click_variance', 'avg_scroll'
        ]

        features = [sample[f] for f in feature_names]
        feature_buffer.append(features)

        if len(feature_buffer) >= 5:
            sequence = np.array(list(feature_buffer), dtype=np.float32)
            seq_flat = sequence.reshape(-1, 30)
            seq_norm = scaler.transform(seq_flat)
            sequence_norm = seq_norm.reshape(1, 5, 30)

            with torch.no_grad():
                x = torch.FloatTensor(sequence_norm)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            predicted_state = label_map[pred.item()]

            predictions.append({
                'window': i + 1,
                'true_state': state,
                'predicted_state': predicted_state,
                'confidence': conf.item(),
                'match': state.replace('_', ' ').title() == predicted_state
            })

            match_symbol = '✓' if predictions[-1]['match'] else '✗'
            print(f"Window {i + 1:2d}: True: {state:20s} | Pred: {predicted_state:15s} "
                  f"({conf.item() * 100:5.1f}%) {match_symbol}")

    if predictions:
        accuracy = sum(p['match'] for p in predictions) / len(predictions)
        avg_conf = np.mean([p['confidence'] for p in predictions])

        print(f"\n{'=' * 80}")
        print(f"TEST RESULTS:")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Accuracy: {accuracy * 100:.1f}%")
        print(f"  Avg Confidence: {avg_conf * 100:.1f}%")
        print(f"{'=' * 80}\n")

        test_file = test_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(test_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"✓ Test results saved to {test_file}\n")


# -------- PASTE ABOVE THIS LINE --------
if __name__ == "__main__":
    import sys
    # ...

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Test mode
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            run_test_mode(duration)
        else:
            # Live mode with user ID
            user_id = sys.argv[1]
            monitor = CompleteLiveMonitor(user_id)
            
            if LIVE_MODE and monitor.model:
                monitor.start()
    else:
        # Default user
        monitor = CompleteLiveMonitor()
        
        if LIVE_MODE and monitor.model:
            monitor.start()
