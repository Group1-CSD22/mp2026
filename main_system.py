"""
Main Work Monitoring System with Rich Metrics Output
Predictions every 3 minutes with detailed analytics
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from collections import deque
import sys

class WorkMonitoringSystem:
    """Complete monitoring system with rich metrics"""

    def __init__(self, model_path='trained_model.pth'):
        print("\n" + "="*70)
        print("WORK MONITORING SYSTEM - INITIALIZING")
        print("="*70)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {self.device}")

        # Load model
        self.model, self.scaler = self.load_model(model_path)

        # State definitions
        self.label_map = {
            0: 'focused_work',
            1: 'task_execution',
            2: 'research_reading',
            3: 'meeting_communication',
            4: 'unproductive'
        }

        self.state_descriptions = {
            'focused_work': 'Deep concentration on single task',
            'task_execution': 'Active work with moderate multitasking',
            'research_reading': 'Information gathering and reading',
            'meeting_communication': 'Meetings or collaborative work',
            'unproductive': 'Low productivity or off-task behavior'
        }

        # Feature buffer (need 5 windows = 15 minutes history)
        self.feature_buffer = deque(maxlen=5)

        # Session tracking
        self.session_data = {
            'start_time': None,
            'predictions': [],
            'state_durations': {state: 0 for state in self.label_map.values()},
            'metrics_history': []
        }

        # Running totals
        self.running_metrics = {
            'total_keystrokes': 0,
            'total_clicks': 0,
            'total_scrolls': 0,
            'total_distance': 0,
            'apps_used': set(),
            'app_switches': 0
        }

        self.is_running = False
        self.prediction_count = 0

        print("\n✓ System initialized successfully")

    def load_model(self, path):
        """Load trained model"""
        try:
            from htan_model import ImprovedHTAN

            # Fix for PyTorch 2.6+ weights_only issue
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Get input dimension from checkpoint (could be 24 or 27)
            # We'll infer it from the first layer weights
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
            sys.exit(1)

    def start_monitoring(self):
        """Start monitoring session"""
        if self.is_running:
            print("System already running")
            return

        self.is_running = True
        self.session_data['start_time'] = datetime.now()

        print("\n" + "="*70)
        print("MONITORING STARTED")
        print("="*70)
        print(f"Start time: {self.session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prediction interval: 3 minutes")
        print(f"Sequence length: 5 windows (15 minutes)")
        print("\nCollecting initial data...")
        print("(First prediction after 15 minutes of data collection)")
        print("="*70 + "\n")

    def stop_monitoring(self):
        """Stop monitoring and save session"""
        if not self.is_running:
            print("System not running")
            return

        self.is_running = False
        self.save_session()

        print("\n" + "="*70)
        print("MONITORING STOPPED")
        print("="*70)
        self.print_session_summary()

    def process_window(self, features):
        """Process 3-minute window and make prediction"""

        # Add to buffer
        self.feature_buffer.append(features)

        # Update running metrics
        self._update_running_metrics(features)

        # Need 5 windows before prediction
        if len(self.feature_buffer) < 5:
            remaining = 5 - len(self.feature_buffer)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Data collected. "
                  f"Need {remaining} more window(s) for prediction...")
            return None

        # Make prediction
        prediction_result = self._predict(features)

        # Store prediction
        self.session_data['predictions'].append(prediction_result)
        self.session_data['state_durations'][prediction_result['state']] += 180  # 3 minutes
        self.session_data['metrics_history'].append(prediction_result)

        self.prediction_count += 1

        # Print detailed output
        self._print_prediction_output(prediction_result)

        return prediction_result

    def _predict(self, current_features):
        """Make prediction from feature sequence"""

        # Feature order
        feature_names = [
            'keystroke_count', 'keystroke_rate', 'typing_speed_wpm',
            'keystroke_variance', 'keystroke_burst_ratio',
            'avg_inter_keystroke_time', 'keystroke_rhythm_std',
            'mouse_movement_count', 'mouse_distance', 'mouse_velocity',
            'mouse_velocity_std', 'mouse_acceleration',
            'mouse_direction_changes', 'mouse_idle_ratio',
            'click_count', 'click_rate', 'click_clustering',
            'avg_click_interval', 'click_pattern_regularity',
            'scroll_count', 'scroll_rate', 'scroll_intensity',
            'scroll_direction_changes',
            'app_switch_count', 'app_switch_rate',
            'unique_apps_count', 'app_focus_stability'
        ]

        # Extract feature vectors
        sequence = []
        for feat_dict in self.feature_buffer:
            vec = [feat_dict.get(name, 0.0) for name in feature_names]
            sequence.append(vec)

        sequence = np.array(sequence, dtype=np.float32)

        # Normalize
        seq_flat = sequence.reshape(-1, 27)  # Changed from 24 to 27
        seq_normalized = self.scaler.transform(seq_flat)
        sequence = seq_normalized.reshape(1, 5, 27)  # Changed from 24 to 27

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
        probabilities = probs.cpu().numpy()[0]

        # Build result
        result = {
            'timestamp': datetime.now().isoformat(),
            'prediction_number': self.prediction_count + 1,
            'state': state,
            'state_description': self.state_descriptions[state],
            'confidence': float(confidence_score),
            'all_probabilities': {
                self.label_map[i]: float(probabilities[i])
                for i in range(5)
            },
            'current_window_metrics': current_features,
            'session_totals': dict(self.running_metrics),
            'productivity_score': self._calculate_productivity_score()
        }

        # Convert set to list for JSON
        result['session_totals']['apps_used'] = list(result['session_totals']['apps_used'])

        return result

    def _update_running_metrics(self, features):
        """Update running totals"""
        self.running_metrics['total_keystrokes'] += int(features.get('keystroke_count', 0))
        self.running_metrics['total_clicks'] += int(features.get('click_count', 0))
        self.running_metrics['total_scrolls'] += int(features.get('scroll_count', 0))
        self.running_metrics['total_distance'] += features.get('mouse_distance', 0)
        self.running_metrics['app_switches'] += int(features.get('app_switch_count', 0))

    def _calculate_productivity_score(self):
        """Calculate overall productivity score (0-100)"""
        if not self.session_data['predictions']:
            return 0.0

        # Weight states by productivity
        weights = {
            'focused_work': 1.0,
            'task_execution': 0.85,
            'research_reading': 0.75,
            'meeting_communication': 0.7,
            'unproductive': 0.2
        }

        total_time = sum(self.session_data['state_durations'].values())
        if total_time == 0:
            return 0.0

        score = 0.0
        for state, duration in self.session_data['state_durations'].items():
            score += (duration / total_time) * weights[state]

        return round(score * 100, 1)

    def _print_prediction_output(self, result):
        """Print detailed prediction output"""
        print("\n" + "="*70)
        print(f"PREDICTION #{result['prediction_number']}")
        print("="*70)

        # Timestamp
        print(f"\nTime: {datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')}")

        # Main prediction
        print(f"\n{'DETECTED STATE':^70}")
        print("-"*70)
        print(f"State: {result['state'].upper().replace('_', ' ')}")
        print(f"Description: {result['state_description']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")

        # All probabilities
        print(f"\n{'ALL STATE PROBABILITIES':^70}")
        print("-"*70)
        for state, prob in sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"{state:25s} {bar} {prob*100:5.1f}%")

        # Current window metrics
        metrics = result['current_window_metrics']
        rich = metrics.get('rich_metrics', {})

        print(f"\n{'CURRENT WINDOW METRICS (3 MINUTES)':^70}")
        print("-"*70)
        print(f"Keyboard:")
        print(f"  Typing Speed: {metrics.get('typing_speed_wpm', 0):.1f} WPM")
        print(f"  Total Keystrokes: {metrics.get('keystroke_count', 0)}")
        print(f"  Keystroke Rate: {metrics.get('keystroke_rate', 0):.2f}/sec")

        print(f"\nMouse:")
        print(f"  Total Distance: {metrics.get('mouse_distance', 0):.0f} pixels")
        print(f"  Average Velocity: {metrics.get('mouse_velocity', 0):.1f} px/s")
        print(f"  Direction Changes: {metrics.get('mouse_direction_changes', 0)}")
        print(f"  Idle Ratio: {metrics.get('mouse_idle_ratio', 0)*100:.1f}%")

        print(f"\nClicks & Scrolls:")
        print(f"  Clicks: {metrics.get('click_count', 0)}")
        print(f"  Click Rate: {metrics.get('click_rate', 0):.2f}/sec")
        print(f"  Scrolls: {metrics.get('scroll_count', 0)}")
        print(f"  Scroll Intensity: {metrics.get('scroll_intensity', 0):.2f}")

        print(f"\nApplication Usage:")
        print(f"  App Switches: {metrics.get('app_switch_count', 0)}")
        print(f"  Unique Apps: {metrics.get('unique_apps_count', 0)}")
        print(f"  Focus Stability: {metrics.get('app_focus_stability', 0)*100:.1f}%")

        # Session totals
        totals = result['session_totals']
        duration = (datetime.now() - self.session_data['start_time']).seconds

        print(f"\n{'SESSION TOTALS':^70}")
        print("-"*70)
        print(f"Duration: {duration//60} minutes")
        print(f"Total Keystrokes: {totals['total_keystrokes']}")
        print(f"Total Clicks: {totals['total_clicks']}")
        print(f"Total Scrolls: {totals['total_scrolls']}")
        print(f"App Switches: {totals['app_switches']}")
        print(f"Productivity Score: {result['productivity_score']:.1f}/100")

        print("="*70 + "\n")

    def print_session_summary(self):
        """Print final session summary"""
        if not self.session_data['start_time']:
            return

        duration = (datetime.now() - self.session_data['start_time']).seconds

        print(f"\nSession Duration: {duration//60} minutes")
        print(f"Total Predictions: {self.prediction_count}")

        print(f"\nState Breakdown:")
        total_time = sum(self.session_data['state_durations'].values())
        for state, duration in sorted(self.session_data['state_durations'].items(),
                                     key=lambda x: x[1], reverse=True):
            if total_time > 0:
                percentage = (duration / total_time) * 100
                minutes = duration // 60
                print(f"  {state:25s}: {minutes:3d} min ({percentage:5.1f}%)")

        print(f"\nFinal Productivity Score: {self._calculate_productivity_score()}/100")
        print("="*70 + "\n")

    def save_session(self):
        """Save session data to file"""
        # Prepare session data
        session_export = {
            'start_time': self.session_data['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.session_data['start_time']).seconds,
            'total_predictions': self.prediction_count,
            'state_durations': self.session_data['state_durations'],
            'productivity_score': self._calculate_productivity_score(),
            'running_metrics': {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.running_metrics.items()
            },
            'predictions': self.session_data['predictions']
        }

        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_export, f, indent=2)

        print(f"✓ Session saved to {filename}")

    def simulate_demo(self, num_windows=6):
        """Simulate monitoring with synthetic data"""
        print("\nRunning DEMO mode with synthetic data...")
        print(f"Will generate {num_windows} predictions ({num_windows*3} minutes)\n")

        self.start_monitoring()

        for i in range(num_windows):
            # Generate synthetic features
            features = self._generate_demo_features()

            # Process window
            self.process_window(features)

            # Wait between predictions (or skip in demo)
            if i < num_windows - 1:
                print(f"\nWaiting 3 minutes for next window...")
                # time.sleep(180)  # Uncomment for real timing

        self.stop_monitoring()

    def _generate_demo_features(self):
        """Generate realistic demo features"""
        import random

        state = random.choice(list(self.label_map.values()))

        if state == 'focused_work':
            return {
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
        # Add similar for other states...
        else:
            return {
                'keystroke_count': random.uniform(100, 250),
                'keystroke_rate': random.uniform(0.5, 1.4),
                'typing_speed_wpm': random.uniform(20, 35),
                'keystroke_variance': random.uniform(0.02, 0.08),
                'keystroke_burst_ratio': random.uniform(0.3, 0.5),
                'avg_inter_keystroke_time': random.uniform(0.1, 0.2),
                'keystroke_rhythm_std': random.uniform(0.08, 0.2),
                'mouse_movement_count': random.uniform(500, 1200),
                'mouse_distance': random.uniform(8000, 25000),
                'mouse_velocity': random.uniform(80, 200),
                'mouse_velocity_std': random.uniform(20, 50),
                'mouse_acceleration': random.uniform(25, 60),
                'mouse_direction_changes': random.randint(30, 80),
                'mouse_idle_ratio': random.uniform(0.2, 0.4),
                'click_count': random.randint(25, 60),
                'click_rate': random.uniform(0.14, 0.33),
                'click_clustering': random.uniform(0.7, 1.5),
                'avg_click_interval': random.uniform(5, 12),
                'click_pattern_regularity': random.uniform(0.4, 0.7),
                'scroll_count': random.randint(15, 45),
                'scroll_rate': random.uniform(0.08, 0.25),
                'scroll_intensity': random.uniform(1.5, 4),
                'scroll_direction_changes': random.randint(4, 12),
                'app_switch_count': random.randint(3, 8),
                'app_switch_rate': random.uniform(1, 2.7),
                'unique_apps_count': random.randint(2, 4),
                'app_focus_stability': random.uniform(0.4, 0.7)
            }


if __name__ == "__main__":
    try:
        system = WorkMonitoringSystem('trained_model.pth')
        system.simulate_demo(num_windows=6)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()