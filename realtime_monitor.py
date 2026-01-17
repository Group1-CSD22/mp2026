"""
Real-Time Work Monitoring System with Anomaly Detection
Monitors user behavior in real-time and detects intrusions/anomalies
"""

import torch
import numpy as np
import json
import time
import threading
from datetime import datetime
from collections import deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Detects anomalies and potential intrusions in user behavior"""

    def __init__(self, sensitivity=2.5):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.user_baseline = None
        self.baseline_samples = deque(maxlen=100)  # Build baseline from first 100 samples
        self.anomaly_history = []
        self.is_baseline_built = False

    def build_baseline(self, features):
        """Build user's normal behavior baseline"""
        self.baseline_samples.append(features)

        if len(self.baseline_samples) >= 20 and not self.is_baseline_built:
            # Calculate baseline statistics
            samples_array = np.array(self.baseline_samples)
            self.user_baseline = {
                'mean': np.mean(samples_array, axis=0),
                'std': np.std(samples_array, axis=0) + 1e-6,  # Avoid division by zero
                'min': np.min(samples_array, axis=0),
                'max': np.max(samples_array, axis=0)
            }
            self.is_baseline_built = True
            return True
        return False

    def detect_anomaly(self, features):
        """Detect if current behavior is anomalous"""
        if not self.is_baseline_built:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'baseline_building',
                'details': 'Building user baseline...'
            }

        features_array = np.array(features)

        # Calculate z-scores
        z_scores = np.abs((features_array - self.user_baseline['mean']) / self.user_baseline['std'])

        # Check for anomalies
        anomaly_mask = z_scores > self.sensitivity
        anomaly_count = np.sum(anomaly_mask)
        anomaly_score = np.mean(z_scores)

        # Determine anomaly type
        is_anomaly = anomaly_count > 5  # More than 5 features are anomalous
        anomaly_type = 'normal'
        details = []

        if is_anomaly:
            # Check specific patterns
            if features_array[0] > self.user_baseline['mean'][0] * 3:  # Keystroke count
                anomaly_type = 'suspicious_high_activity'
                details.append('Extremely high typing activity')

            if features_array[7] > self.user_baseline['mean'][7] * 3:  # Mouse movement
                anomaly_type = 'suspicious_mouse_activity'
                details.append('Unusual mouse movement patterns')

            if features_array[23] < 0.3:  # App focus stability
                anomaly_type = 'rapid_app_switching'
                details.append('Rapid application switching detected')

            # Check for potential intrusion patterns
            if anomaly_count > 10:
                anomaly_type = 'potential_intrusion'
                details.append('ALERT: Multiple behavioral deviations detected')
                details.append('This may indicate unauthorized access')

        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'anomaly_type': anomaly_type,
            'anomalous_features': int(anomaly_count),
            'details': ' | '.join(details) if details else 'Normal behavior',
            'timestamp': datetime.now().isoformat()
        }

        if is_anomaly:
            self.anomaly_history.append(result)

        return result


class RealTimeMonitor:
    """Real-time monitoring system with live predictions"""

    def __init__(self, model_path='trained_model.pth'):
        print("\n" + "="*70)
        print("REAL-TIME WORK MONITORING SYSTEM")
        print("WITH ANOMALY & INTRUSION DETECTION")
        print("="*70)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {self.device}")

        # Load model
        self.model, self.scaler = self.load_model(model_path)

        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(sensitivity=2.5)

        # State definitions
        self.label_map = {
            0: 'focused_work',
            1: 'task_execution',
            2: 'research_reading',
            3: 'meeting_communication',
            4: 'unproductive'
        }

        # Monitoring state
        self.is_monitoring = False
        self.feature_buffer = deque(maxlen=5)
        self.current_state = 'initializing'
        self.prediction_count = 0

        # Session data
        self.session_start = None
        self.anomalies_detected = []
        self.state_history = []

        # Real-time data collection (simulated for demo)
        self.collection_thread = None

        print("\n✓ Real-time monitoring system initialized")
        print("✓ Anomaly detection enabled")

    def load_model(self, path):
        """Load trained model"""
        try:
            from htan_model import ImprovedHTAN

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            first_layer_weight = checkpoint['model_state_dict']['feature_embedding.0.weight']
            input_dim = first_layer_weight.shape[1]

            model = ImprovedHTAN(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=5,
                seq_len=5
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            scaler = checkpoint['scaler']

            print(f"✓ Loaded model from {path}")
            print(f"  Input features: {input_dim}")
            return model, scaler

        except Exception as e:
            print(f"ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("Already monitoring!")
            return

        self.is_monitoring = True
        self.session_start = datetime.now()

        print("\n" + "="*70)
        print("REAL-TIME MONITORING STARTED")
        print("="*70)
        print(f"Start time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Update interval: Every 3 minutes")
        print(f"Anomaly detection: Active")
        print("\nCollecting data...")
        print("="*70 + "\n")

        # Start collection thread
        self.collection_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.collection_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("\n" + "="*70)
        print("MONITORING STOPPED")
        print("="*70)
        self.print_summary()

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            # Collect 3-minute window of data
            features = self._collect_window_data()

            # Process features
            self.process_features(features)

            # Wait 3 minutes (or skip for demo)
            time.sleep(180)  # 3 minutes

    def _collect_window_data(self):
        """Collect data for 3-minute window (simulated)"""
        # In real implementation, this would use the data collector
        # For now, generate realistic features
        import random

        states = ['focused_work', 'task_execution', 'research_reading',
                 'meeting_communication', 'unproductive']
        state = random.choice(states)

        # Generate features based on state
        if state == 'focused_work':
            features = {
                'keystroke_count': random.uniform(200, 400),
                'keystroke_rate': random.uniform(1.1, 2.2),
                'typing_speed_wpm': random.uniform(30, 50),
                'keystroke_variance': random.uniform(0.01, 0.05),
                'keystroke_burst_ratio': random.uniform(0.4, 0.7),
                'avg_inter_keystroke_time': random.uniform(0.06, 0.12),
                'keystroke_rhythm_std': random.uniform(0.05, 0.15),
                'mouse_movement_count': random.uniform(300, 800),
                'mouse_distance': random.uniform(5000, 15000),
                'mouse_velocity': random.uniform(50, 150),
                'mouse_velocity_std': random.uniform(10, 30),
                'mouse_acceleration': random.uniform(15, 40),
                'mouse_direction_changes': random.randint(20, 50),
                'mouse_idle_ratio': random.uniform(0.1, 0.3),
                'click_count': random.randint(15, 40),
                'click_rate': random.uniform(0.08, 0.22),
                'click_clustering': random.uniform(0.5, 1.2),
                'avg_click_interval': random.uniform(8, 20),
                'click_pattern_regularity': random.uniform(0.6, 0.9),
                'scroll_count': random.randint(10, 30),
                'scroll_rate': random.uniform(0.05, 0.17),
                'scroll_intensity': random.uniform(1, 3),
                'scroll_direction_changes': random.randint(2, 8),
                'app_switch_count': random.randint(0, 3),
                'app_switch_rate': random.uniform(0, 1),
                'unique_apps_count': random.randint(1, 2),
                'app_focus_stability': random.uniform(0.7, 1.0)
            }
        else:
            features = {
                'keystroke_count': random.uniform(50, 200),
                'keystroke_rate': random.uniform(0.3, 1.1),
                'typing_speed_wpm': random.uniform(10, 30),
                'keystroke_variance': random.uniform(0.02, 0.1),
                'keystroke_burst_ratio': random.uniform(0.2, 0.5),
                'avg_inter_keystroke_time': random.uniform(0.15, 0.4),
                'keystroke_rhythm_std': random.uniform(0.1, 0.3),
                'mouse_movement_count': random.uniform(200, 1000),
                'mouse_distance': random.uniform(3000, 20000),
                'mouse_velocity': random.uniform(30, 180),
                'mouse_velocity_std': random.uniform(15, 50),
                'mouse_acceleration': random.uniform(10, 60),
                'mouse_direction_changes': random.randint(15, 70),
                'mouse_idle_ratio': random.uniform(0.2, 0.6),
                'click_count': random.randint(10, 50),
                'click_rate': random.uniform(0.05, 0.28),
                'click_clustering': random.uniform(0.4, 1.5),
                'avg_click_interval': random.uniform(5, 25),
                'click_pattern_regularity': random.uniform(0.3, 0.8),
                'scroll_count': random.randint(5, 60),
                'scroll_rate': random.uniform(0.03, 0.33),
                'scroll_intensity': random.uniform(0.5, 5),
                'scroll_direction_changes': random.randint(1, 15),
                'app_switch_count': random.randint(0, 12),
                'app_switch_rate': random.uniform(0, 4),
                'unique_apps_count': random.randint(1, 5),
                'app_focus_stability': random.uniform(0.3, 0.9)
            }

        # Occasionally inject anomalous behavior
        if random.random() < 0.15:  # 15% chance of anomaly
            features['keystroke_count'] *= 5
            features['mouse_distance'] *= 4
            features['app_switch_count'] *= 3

        return features

    def process_features(self, features):
        """Process features and make prediction"""
        # Convert to feature vector
        feature_vector = list(features.values())

        # Build baseline for anomaly detection
        if not self.anomaly_detector.is_baseline_built:
            baseline_ready = self.anomaly_detector.build_baseline(feature_vector)
            if baseline_ready:
                print("\n✓ User baseline established!")
                print("  Anomaly detection is now active\n")

        # Detect anomalies
        anomaly_result = self.anomaly_detector.detect_anomaly(feature_vector)

        # Add to buffer
        self.feature_buffer.append(feature_vector)

        # Need 5 windows for prediction
        if len(self.feature_buffer) < 5:
            remaining = 5 - len(self.feature_buffer)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting data... "
                  f"({remaining} more windows needed)")
            return

        # Make prediction
        prediction = self._predict()

        # Combine prediction with anomaly detection
        result = {
            **prediction,
            'anomaly': anomaly_result
        }

        self.prediction_count += 1
        self.state_history.append(result)

        if anomaly_result['is_anomaly']:
            self.anomalies_detected.append(anomaly_result)

        # Display results
        self._display_realtime_output(result)

    def _predict(self):
        """Make state prediction"""
        # Create sequence
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)

        # Normalize
        seq_flat = sequence.reshape(-1, 27)
        seq_normalized = self.scaler.transform(seq_flat)
        sequence = seq_normalized.reshape(1, 5, 27)

        # Predict
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        predicted_class = predicted.item()
        state = self.label_map[predicted_class]
        confidence_score = confidence.item()

        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_number': self.prediction_count + 1,
            'state': state,
            'confidence': float(confidence_score)
        }

    def _display_realtime_output(self, result):
        """Display real-time monitoring output"""
        print("\n" + "="*70)
        print(f"REAL-TIME UPDATE #{result['prediction_number']}")
        print("="*70)
        print(f"Time: {datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')}")

        # State prediction
        print(f"\n{'CURRENT STATE':^70}")
        print("-"*70)
        print(f"State: {result['state'].upper().replace('_', ' ')}")
        print(f"Confidence: {result['confidence']*100:.1f}%")

        # Anomaly detection
        anomaly = result['anomaly']
        print(f"\n{'ANOMALY DETECTION':^70}")
        print("-"*70)

        if anomaly['is_anomaly']:
            print(f"⚠️  ANOMALY DETECTED!")
            print(f"Type: {anomaly['anomaly_type'].upper().replace('_', ' ')}")
            print(f"Severity: {anomaly['anomaly_score']:.2f}")
            print(f"Details: {anomaly['details']}")

            if anomaly['anomaly_type'] == 'potential_intrusion':
                print("\n🚨 SECURITY ALERT: Possible unauthorized access detected!")
                print("   Recommended action: Verify user identity")
        else:
            print(f"✓ Normal behavior (Score: {anomaly['anomaly_score']:.2f})")
            print(f"Status: {anomaly['details']}")

        # Session stats
        duration = (datetime.now() - self.session_start).seconds // 60
        print(f"\n{'SESSION INFO':^70}")
        print("-"*70)
        print(f"Duration: {duration} minutes")
        print(f"Total predictions: {self.prediction_count + 1}")
        print(f"Anomalies detected: {len(self.anomalies_detected)}")

        print("="*70 + "\n")

    def print_summary(self):
        """Print monitoring session summary"""
        duration = (datetime.now() - self.session_start).seconds // 60

        print(f"\nSession Duration: {duration} minutes")
        print(f"Total Predictions: {self.prediction_count}")
        print(f"Anomalies Detected: {len(self.anomalies_detected)}")

        if self.anomalies_detected:
            print(f"\nAnomaly Summary:")
            anomaly_types = {}
            for anomaly in self.anomalies_detected:
                atype = anomaly['anomaly_type']
                anomaly_types[atype] = anomaly_types.get(atype, 0) + 1

            for atype, count in anomaly_types.items():
                print(f"  {atype.replace('_', ' ').title()}: {count}")

        print("="*70 + "\n")

    def run_demo(self, num_updates=10):
        """Run demo mode with simulated data"""
        print("\nRunning DEMO mode...")
        print(f"Will generate {num_updates} real-time updates\n")

        self.session_start = datetime.now()
        self.is_monitoring = True

        for i in range(num_updates):
            features = self._collect_window_data()
            self.process_features(features)

            if i < num_updates - 1:
                print("Waiting 3 minutes for next update...")
                # time.sleep(180)  # Uncomment for real timing
                time.sleep(2)  # Fast demo

        self.is_monitoring = False
        self.print_summary()


if __name__ == "__main__":
    try:
        monitor = RealTimeMonitor('trained_model.pth')

        if monitor.model is None:
            print("\nPlease train the model first:")
            print("  python training_pipeline.py")
        else:
            monitor.run_demo(num_updates=10)

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()