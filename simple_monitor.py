"""
Simple Real-Time Work Monitor
Clear, easy to understand output with anomaly detection
"""

import torch
import numpy as np
import time
from datetime import datetime, timedelta
from collections import deque
import random

class SimpleWorkMonitor:
    """Simple, clear real-time work monitoring"""
    
    def __init__(self, model_path='trained_model.pth'):
        print("\n" + "="*80)
        print(" "*20 + "WORK MONITORING SYSTEM v1.0")
        print("="*80)
        
        # Load model
        self.device = 'cpu'
        self.model, self.scaler = self._load_model(model_path)
        
        # States
        self.states = {
            0: 'Focused Work',
            1: 'Active Task',
            2: 'Reading/Research',
            3: 'Meeting/Call',
            4: 'Unproductive'
        }
        
        # Monitoring data
        self.feature_buffer = deque(maxlen=5)
        self.user_baseline = None
        self.baseline_data = []
        
        # Session tracking
        self.session_start = None
        self.updates = []
        self.anomalies = []
        
        print("\n✓ System ready")
        print(f"✓ Model loaded: 27 features, 5 states")
        print(f"✓ Device: {self.device}\n")
    
    def _load_model(self, path):
        """Load model"""
        from htan_model import ImprovedHTAN
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Get input dimension
        first_layer = checkpoint['model_state_dict']['feature_embedding.0.weight']
        input_dim = first_layer.shape[1]
        
        model = ImprovedHTAN(input_dim=input_dim, hidden_dim=128, num_classes=5, seq_len=5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint['scaler']
    
    def start(self, duration_minutes=30, demo_mode=True):
        """Start monitoring session"""
        self.session_start = datetime.now()
        
        print("="*80)
        print(f"SESSION STARTED: {self.session_start.strftime('%I:%M %p')}")
        print("="*80)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Update interval: Every 3 minutes")
        print(f"Mode: {'DEMO (simulated data)' if demo_mode else 'LIVE'}")
        print("="*80)
        
        # Calculate number of updates
        num_updates = duration_minutes // 3
        
        print(f"\nCollecting initial baseline data...")
        print("(Need 5 windows = 15 minutes before first prediction)\n")
        
        # Run monitoring loop
        for i in range(num_updates):
            # Collect data for this 3-minute window
            if demo_mode:
                features = self._generate_demo_data()
            else:
                features = self._collect_real_data()
            
            # Add to buffer
            self.feature_buffer.append(features)
            
            # Check if we have enough data
            if len(self.feature_buffer) < 5:
                remaining = 5 - len(self.feature_buffer)
                elapsed = (i + 1) * 3
                print(f"[{elapsed:2d} min] Collecting data... ({remaining} more windows needed)")
                
                # Build baseline
                if len(self.baseline_data) < 20:
                    self.baseline_data.append(features)
                
                continue
            
            # Build baseline if needed
            if self.user_baseline is None and len(self.baseline_data) >= 5:
                self._build_baseline()
            
            # Make prediction and check for anomalies
            self._process_update(i + 1, features)
            
            # Simulate waiting (skip in demo)
            if not demo_mode:
                time.sleep(180)  # 3 minutes
        
        # Print final summary
        self._print_summary()
    
    def _build_baseline(self):
        """Build user behavior baseline"""
        baseline_array = np.array(self.baseline_data)
        self.user_baseline = {
            'mean': np.mean(baseline_array, axis=0),
            'std': np.std(baseline_array, axis=0) + 1e-6
        }
        print("\n" + "="*80)
        print("✓ USER BASELINE ESTABLISHED")
        print("="*80)
        print("Anomaly detection is now ACTIVE\n")
    
    def _process_update(self, update_num, current_features):
        """Process a monitoring update"""
        # Make prediction
        state, confidence, all_probs = self._predict()
        
        # Check for anomalies
        is_anomaly, anomaly_score, anomaly_details = self._check_anomaly(current_features)
        
        # Calculate metrics
        elapsed = (datetime.now() - self.session_start).seconds // 60
        
        # Store update
        update = {
            'number': update_num,
            'time': datetime.now(),
            'elapsed_minutes': elapsed,
            'state': state,
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'features': current_features
        }
        self.updates.append(update)
        
        if is_anomaly:
            self.anomalies.append(update)
        
        # Display update
        self._display_update(update, anomaly_details)
    
    def _predict(self):
        """Make state prediction"""
        # Convert buffer to sequence
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        
        # Normalize
        seq_flat = sequence.reshape(-1, 27)
        seq_norm = self.scaler.transform(seq_flat)
        sequence = seq_norm.reshape(1, 5, 27)
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        
        state_id = predicted.item()
        state_name = self.states[state_id]
        conf = confidence.item()
        all_probs = probs.cpu().numpy()[0]
        
        return state_name, conf, all_probs
    
    def _check_anomaly(self, features):
        """Check if current behavior is anomalous"""
        if self.user_baseline is None:
            return False, 0.0, "Baseline building"
        
        features_array = np.array(features)
        
        # Calculate z-scores
        z_scores = np.abs((features_array - self.user_baseline['mean']) / self.user_baseline['std'])
        anomaly_score = np.mean(z_scores)
        
        # Count anomalous features (z > 3)
        anomalous_count = np.sum(z_scores > 3)
        
        # Determine if anomaly
        is_anomaly = anomalous_count > 5
        
        # Details
        if is_anomaly:
            if anomalous_count > 15:
                details = "⚠️  CRITICAL: Possible intrusion detected"
            elif anomalous_count > 10:
                details = "⚠️  HIGH: Unusual behavior pattern"
            else:
                details = "⚠️  MODERATE: Deviation from normal"
        else:
            details = "✓ Normal behavior"
        
        return is_anomaly, anomaly_score, details
    
    def _display_update(self, update, anomaly_details):
        """Display monitoring update"""
        print("\n" + "="*80)
        print(f"UPDATE #{update['number']:02d} | {update['time'].strftime('%I:%M:%S %p')} | {update['elapsed_minutes']} min elapsed")
        print("="*80)
        
        # State
        print(f"\n📊 CURRENT STATE: {update['state'].upper()}")
        print(f"   Confidence: {update['confidence']*100:.1f}%")
        
        # Key metrics from this window
        features = update['features']
        print(f"\n📈 WINDOW METRICS (last 3 minutes):")
        print(f"   Typing: {features[2]:.0f} WPM | {int(features[0])} keystrokes")
        print(f"   Mouse: {features[8]:.0f} pixels moved | {int(features[14])} clicks")
        print(f"   Apps: {int(features[23])} switches | {features[26]*100:.0f}% focus")
        
        # Anomaly status
        if update['is_anomaly']:
            print(f"\n🚨 ANOMALY DETECTED!")
            print(f"   {anomaly_details}")
            print(f"   Severity Score: {update['anomaly_score']:.2f}")
        else:
            print(f"\n✓ {anomaly_details}")
            print(f"   Behavior Score: {update['anomaly_score']:.2f}")
        
        print("="*80)
    
    def _print_summary(self):
        """Print session summary"""
        duration = (datetime.now() - self.session_start).seconds // 60
        
        print("\n\n" + "="*80)
        print(" "*25 + "SESSION SUMMARY")
        print("="*80)
        
        print(f"\n⏱️  SESSION DETAILS:")
        print(f"   Start: {self.session_start.strftime('%I:%M %p')}")
        print(f"   End: {datetime.now().strftime('%I:%M %p')}")
        print(f"   Duration: {duration} minutes")
        print(f"   Total Updates: {len(self.updates)}")
        
        # State breakdown
        if self.updates:
            state_counts = {}
            for update in self.updates:
                state = update['state']
                state_counts[state] = state_counts.get(state, 0) + 1
            
            print(f"\n📊 STATE BREAKDOWN:")
            total = len(self.updates)
            for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                bar_length = int(percentage / 2)
                bar = '█' * bar_length
                print(f"   {state:20s} {bar:50s} {count:2d} ({percentage:5.1f}%)")
        
        # Productivity score
        if self.updates:
            weights = {
                'Focused Work': 100,
                'Active Task': 85,
                'Reading/Research': 75,
                'Meeting/Call': 70,
                'Unproductive': 20
            }
            total_score = sum(weights[u['state']] for u in self.updates)
            avg_score = total_score / len(self.updates)
            
            print(f"\n⭐ PRODUCTIVITY SCORE: {avg_score:.1f}/100")
        
        # Anomaly report
        print(f"\n🔒 SECURITY:")
        print(f"   Anomalies Detected: {len(self.anomalies)}")
        if self.anomalies:
            print(f"   ⚠️  WARNING: {len(self.anomalies)} unusual behavior events detected")
            print(f"   Review recommended for security purposes")
        else:
            print(f"   ✓ No anomalies detected - normal session")
        
        print("\n" + "="*80 + "\n")
    
    def _generate_demo_data(self):
        """Generate realistic demo data"""
        states = ['focused', 'active', 'reading', 'meeting', 'unproductive']
        state = random.choice(states)
        
        if state == 'focused':
            return [
                random.uniform(200, 400),      # keystroke_count
                random.uniform(1.1, 2.2),      # keystroke_rate
                random.uniform(30, 50),        # typing_speed_wpm
                random.uniform(0.01, 0.05),
                random.uniform(0.4, 0.7),
                random.uniform(0.06, 0.12),
                random.uniform(0.05, 0.15),
                random.uniform(300, 800),      # mouse_movement_count
                random.uniform(5000, 15000),   # mouse_distance
                random.uniform(50, 150),
                random.uniform(10, 30),
                random.uniform(15, 40),
                random.randint(20, 50),
                random.uniform(0.1, 0.3),
                random.randint(15, 40),        # click_count
                random.uniform(0.08, 0.22),
                random.uniform(0.5, 1.2),
                random.uniform(8, 20),
                random.uniform(0.6, 0.9),
                random.randint(10, 30),
                random.uniform(0.05, 0.17),
                random.uniform(1, 3),
                random.randint(2, 8),
                random.randint(0, 3),          # app_switch_count
                random.uniform(0, 1),
                random.randint(1, 2),
                random.uniform(0.7, 1.0)       # app_focus_stability
            ]
        else:
            # Other states - more variable behavior
            features = [
                random.uniform(50, 250),
                random.uniform(0.3, 1.4),
                random.uniform(10, 35),
                random.uniform(0.02, 0.1),
                random.uniform(0.2, 0.5),
                random.uniform(0.1, 0.3),
                random.uniform(0.08, 0.25),
                random.uniform(200, 1200),
                random.uniform(3000, 25000),
                random.uniform(40, 200),
                random.uniform(15, 55),
                random.uniform(10, 65),
                random.randint(15, 80),
                random.uniform(0.2, 0.6),
                random.randint(10, 60),
                random.uniform(0.05, 0.33),
                random.uniform(0.4, 1.6),
                random.uniform(5, 25),
                random.uniform(0.3, 0.8),
                random.randint(5, 70),
                random.uniform(0.03, 0.35),
                random.uniform(0.5, 5.5),
                random.randint(1, 18),
                random.randint(0, 12),
                random.uniform(0, 4),
                random.randint(1, 5),
                random.uniform(0.3, 0.9)
            ]
            
            # Occasionally inject anomaly
            if random.random() < 0.2:
                features[0] *= 4  # High keystrokes
                features[8] *= 3  # High mouse movement
                features[23] *= 2  # Many app switches
            
            return features
    
    def _collect_real_data(self):
        """Collect real data (placeholder)"""
        # In real implementation, use data_collector.py
        return self._generate_demo_data()


if __name__ == "__main__":
    try:
        monitor = SimpleWorkMonitor('trained_model.pth')
        
        # Run 30-minute demo session
        monitor.start(duration_minutes=30, demo_mode=True)
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user\n")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
