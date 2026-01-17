"""
LIVE REAL-TIME WORK MONITORING SYSTEM
Collects actual keyboard, mouse, and app data in real-time
With user profiling and anomaly detection
"""

import torch
import numpy as np
import json
import time
import threading
from datetime import datetime
from collections import deque
from pathlib import Path

# Data collection imports
try:
    from pynput import keyboard, mouse
    import psutil
    import win32gui
    import win32process
    LIVE_MODE_AVAILABLE = True
except ImportError:
    LIVE_MODE_AVAILABLE = False
    print("⚠️  Warning: Live mode dependencies not installed")
    print("   Install with: pip install pynput psutil pywin32")


class LiveDataCollector:
    """Collects real keyboard, mouse, and application data"""
    
    def __init__(self):
        self.is_collecting = False
        
        # Event storage for current 3-minute window
        self.keystroke_times = []
        self.mouse_positions = []
        self.mouse_clicks = []
        self.scroll_events = []
        
        # App tracking
        self.current_app = None
        self.app_switches = []
        self.app_start_time = time.time()
        
        # Listeners
        self.kb_listener = None
        self.mouse_listener = None
        self.app_monitor_thread = None
        
        # Window start time
        self.window_start = None
        
    def start_collection(self):
        """Start collecting data"""
        self.is_collecting = True
        self.window_start = time.time()
        
        # Clear previous window data
        self.keystroke_times = []
        self.mouse_positions = []
        self.mouse_clicks = []
        self.scroll_events = []
        self.app_switches = []
        
        # Start keyboard listener
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        
        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # Start app monitoring
        self.app_monitor_thread = threading.Thread(target=self._monitor_apps, daemon=True)
        self.app_monitor_thread.start()
        
    def stop_collection(self):
        """Stop collecting data"""
        self.is_collecting = False
        
        if self.kb_listener:
            self.kb_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
    
    def _on_key_press(self, key):
        """Record keystroke"""
        if self.is_collecting:
            self.keystroke_times.append(time.time())
    
    def _on_mouse_move(self, x, y):
        """Record mouse movement"""
        if self.is_collecting:
            self.mouse_positions.append((time.time(), x, y))
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Record mouse click"""
        if self.is_collecting and pressed:
            self.mouse_clicks.append((time.time(), x, y))
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Record scroll"""
        if self.is_collecting:
            self.scroll_events.append((time.time(), dy))
    
    def _get_active_app(self):
        """Get currently active application"""
        try:
            window = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(window)
            process = psutil.Process(pid)
            return process.name()
        except:
            return "Unknown"
    
    def _monitor_apps(self):
        """Monitor application switches"""
        while self.is_collecting:
            try:
                app = self._get_active_app()
                current_time = time.time()
                
                if app != self.current_app:
                    if self.current_app is not None:
                        self.app_switches.append({
                            'from': self.current_app,
                            'to': app,
                            'time': current_time
                        })
                    self.current_app = app
                    self.app_start_time = current_time
                
            except:
                pass
            
            time.sleep(1)
    
    def get_features(self):
        """Calculate features from collected data"""
        window_duration = time.time() - self.window_start
        
        # Keystroke features
        keystroke_count = len(self.keystroke_times)
        keystroke_rate = keystroke_count / window_duration if window_duration > 0 else 0
        typing_speed_wpm = (keystroke_count / 5) / (window_duration / 60) if window_duration > 0 else 0
        
        if len(self.keystroke_times) > 1:
            intervals = np.diff(self.keystroke_times)
            keystroke_variance = float(np.var(intervals))
            keystroke_burst_ratio = float(np.sum(intervals < 0.2) / len(intervals))
            avg_inter_keystroke = float(np.mean(intervals))
            keystroke_rhythm_std = float(np.std(intervals))
        else:
            keystroke_variance = 0.0
            keystroke_burst_ratio = 0.0
            avg_inter_keystroke = 0.0
            keystroke_rhythm_std = 0.0
        
        # Mouse features
        mouse_movement_count = len(self.mouse_positions)
        
        if len(self.mouse_positions) > 1:
            distances = []
            velocities = []
            
            for i in range(1, len(self.mouse_positions)):
                t1, x1, y1 = self.mouse_positions[i-1]
                t2, x2, y2 = self.mouse_positions[i]
                
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                time_diff = t2 - t1
                
                if time_diff > 0:
                    vel = dist / time_diff
                    distances.append(dist)
                    velocities.append(vel)
            
            mouse_distance = float(np.sum(distances))
            mouse_velocity = float(np.mean(velocities)) if velocities else 0.0
            mouse_velocity_std = float(np.std(velocities)) if len(velocities) > 1 else 0.0
            mouse_acceleration = mouse_velocity_std
            mouse_direction_changes = self._count_direction_changes()
            
            # Calculate idle time
            time_gaps = [self.mouse_positions[i][0] - self.mouse_positions[i-1][0] 
                        for i in range(1, len(self.mouse_positions))]
            idle_time = sum([gap for gap in time_gaps if gap > 5.0])
            mouse_idle_ratio = idle_time / window_duration if window_duration > 0 else 0
        else:
            mouse_distance = 0.0
            mouse_velocity = 0.0
            mouse_velocity_std = 0.0
            mouse_acceleration = 0.0
            mouse_direction_changes = 0
            mouse_idle_ratio = 1.0
        
        # Click features
        click_count = len(self.mouse_clicks)
        click_rate = click_count / window_duration if window_duration > 0 else 0
        
        if len(self.mouse_clicks) > 1:
            click_times = [t for t, _, _ in self.mouse_clicks]
            click_intervals = np.diff(click_times)
            click_clustering = float(np.std(click_intervals))
            avg_click_interval = float(np.mean(click_intervals))
            click_pattern_regularity = 1.0 / (1.0 + click_clustering) if click_clustering > 0 else 1.0
        else:
            click_clustering = 0.0
            avg_click_interval = 0.0
            click_pattern_regularity = 1.0
        
        # Scroll features
        scroll_count = len(self.scroll_events)
        scroll_rate = scroll_count / window_duration if window_duration > 0 else 0
        
        if scroll_count > 0:
            scroll_directions = [d for _, d in self.scroll_events]
            scroll_intensity = float(np.mean(np.abs(scroll_directions)))
            scroll_direction_changes = sum(1 for i in range(1, len(scroll_directions))
                                          if (scroll_directions[i] > 0) != (scroll_directions[i-1] > 0))
        else:
            scroll_intensity = 0.0
            scroll_direction_changes = 0
        
        # App features
        app_switch_count = len(self.app_switches)
        app_switch_rate = app_switch_count / (window_duration / 60) if window_duration > 0 else 0
        unique_apps = len(set([s['from'] for s in self.app_switches] + 
                              [s['to'] for s in self.app_switches])) if self.app_switches else 1
        app_focus_stability = 1.0 / (1.0 + app_switch_count)
        
        # Return all 27 features
        return [
            keystroke_count, keystroke_rate, typing_speed_wpm,
            keystroke_variance, keystroke_burst_ratio, avg_inter_keystroke, keystroke_rhythm_std,
            mouse_movement_count, mouse_distance, mouse_velocity,
            mouse_velocity_std, mouse_acceleration, mouse_direction_changes, mouse_idle_ratio,
            click_count, click_rate, click_clustering,
            avg_click_interval, click_pattern_regularity,
            scroll_count, scroll_rate, scroll_intensity, scroll_direction_changes,
            app_switch_count, app_switch_rate, unique_apps, app_focus_stability
        ]
    
    def _count_direction_changes(self):
        """Count mouse direction changes"""
        if len(self.mouse_positions) < 3:
            return 0
        
        changes = 0
        for i in range(2, len(self.mouse_positions)):
            _, x1, y1 = self.mouse_positions[i-2]
            _, x2, y2 = self.mouse_positions[i-1]
            _, x3, y3 = self.mouse_positions[i]
            
            v1 = (x2-x1, y2-y1)
            v2 = (x3-x2, y3-y2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                
                if angle > np.pi / 4:
                    changes += 1
        
        return changes


class UserProfile:
    """Manages user behavioral profile for anomaly detection"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.profile_path = Path(f"user_profiles/{user_id}.json")
        self.profile_path.parent.mkdir(exist_ok=True)
        
        # Load existing profile or create new
        self.baseline = self._load_profile()
        self.session_data = []
        
    def _load_profile(self):
        """Load user profile from disk"""
        if self.profile_path.exists():
            with open(self.profile_path, 'r') as f:
                data = json.load(f)
                print(f"✓ Loaded existing profile for user: {self.user_id}")
                return {
                    'mean': np.array(data['mean']),
                    'std': np.array(data['std']),
                    'sample_count': data['sample_count']
                }
        return None
    
    def update_profile(self, features):
        """Update user profile with new data"""
        self.session_data.append(features)
        
        if len(self.session_data) >= 20:
            # Calculate or update baseline
            data_array = np.array(self.session_data)
            
            if self.baseline is None:
                # Create new baseline
                self.baseline = {
                    'mean': np.mean(data_array, axis=0),
                    'std': np.std(data_array, axis=0) + 1e-6,
                    'sample_count': len(self.session_data)
                }
                print(f"\n✓ User baseline established ({len(self.session_data)} samples)")
            else:
                # Update existing baseline (incremental)
                old_count = self.baseline['sample_count']
                new_count = old_count + len(self.session_data)
                
                # Weighted average
                self.baseline['mean'] = (self.baseline['mean'] * old_count + 
                                        np.sum(data_array, axis=0)) / new_count
                self.baseline['std'] = np.sqrt(
                    (self.baseline['std']**2 * old_count + 
                     np.sum((data_array - self.baseline['mean'])**2, axis=0)) / new_count
                ) + 1e-6
                self.baseline['sample_count'] = new_count
            
            # Save updated profile
            self._save_profile()
            self.session_data = []
    
    def _save_profile(self):
        """Save profile to disk"""
        with open(self.profile_path, 'w') as f:
            json.dump({
                'mean': self.baseline['mean'].tolist(),
                'std': self.baseline['std'].tolist(),
                'sample_count': self.baseline['sample_count']
            }, f)
    
    def check_anomaly(self, features):
        """Check if features are anomalous"""
        if self.baseline is None:
            return False, 0.0, "Building profile..."
        
        features_array = np.array(features)
        
        # Calculate z-scores
        z_scores = np.abs((features_array - self.baseline['mean']) / self.baseline['std'])
        anomaly_score = float(np.mean(z_scores))
        
        # Count highly anomalous features
        critical_anomalies = np.sum(z_scores > 3)
        
        # Determine anomaly status
        is_anomaly = critical_anomalies > 5
        
        if is_anomaly:
            if critical_anomalies > 15:
                severity = "CRITICAL - Possible intrusion"
            elif critical_anomalies > 10:
                severity = "HIGH - Unusual behavior"
            else:
                severity = "MODERATE - Minor deviation"
        else:
            severity = "Normal"
        
        return is_anomaly, anomaly_score, severity


class LiveMonitor:
    """Main live monitoring system"""
    
    def __init__(self, user_id="default_user", model_path='trained_model.pth'):
        print("\n" + "="*80)
        print(" "*20 + "LIVE WORK MONITORING SYSTEM")
        print("="*80)
        
        if not LIVE_MODE_AVAILABLE:
            print("\n❌ ERROR: Live mode dependencies not installed!")
            print("   Install with: pip install pynput psutil pywin32")
            return
        
        # Load model
        self.device = 'cpu'
        self.model, self.scaler = self._load_model(model_path)
        
        if self.model is None:
            print("\n❌ ERROR: Model not found! Train model first:")
            print("   python training_pipeline.py")
            return
        
        # Initialize data collector
        self.collector = LiveDataCollector()
        
        # Initialize user profile
        self.user_profile = UserProfile(user_id)
        
        # States
        self.states = {
            0: 'Focused Work',
            1: 'Active Task',
            2: 'Reading/Research',
            3: 'Meeting/Call',
            4: 'Unproductive'
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.feature_buffer = deque(maxlen=5)
        self.session_start = None
        self.updates = []
        self.anomalies = []
        
        print(f"\n✓ System ready for user: {user_id}")
        print(f"✓ Model loaded: 27 features, 5 states")
        print(f"✓ Live data collection: ENABLED")
        
    def _load_model(self, path):
        """Load model"""
        try:
            from htan_model import ImprovedHTAN
            
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            first_layer = checkpoint['model_state_dict']['feature_embedding.0.weight']
            input_dim = first_layer.shape[1]
            
            model = ImprovedHTAN(input_dim=input_dim, hidden_dim=128, num_classes=5, seq_len=5)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model, checkpoint['scaler']
        except:
            return None, None
    
    def start(self):
        """Start live monitoring"""
        if not LIVE_MODE_AVAILABLE or self.model is None:
            return
        
        self.is_monitoring = True
        self.session_start = datetime.now()
        
        print("\n" + "="*80)
        print(f"LIVE MONITORING STARTED: {self.session_start.strftime('%I:%M %p')}")
        print("="*80)
        print("Monitoring your keyboard, mouse, and applications...")
        print("Press Ctrl+C to stop\n")
        print("Collecting initial data (need 15 minutes for first prediction)...")
        print("="*80 + "\n")
        
        update_num = 0
        
        try:
            while self.is_monitoring:
                # Collect data for 3 minutes
                print(f"[{datetime.now().strftime('%I:%M %p')}] Collecting 3-minute window...")
                
                self.collector.start_collection()
                time.sleep(180)  # 3 minutes
                self.collector.stop_collection()
                
                # Get features
                features = self.collector.get_features()
                
                # Add to buffer
                self.feature_buffer.append(features)
                update_num += 1
                
                # Update user profile
                self.user_profile.update_profile(features)
                
                # Check if we have enough data
                if len(self.feature_buffer) < 5:
                    remaining = 5 - len(self.feature_buffer)
                    elapsed = update_num * 3
                    print(f"   [{elapsed} min] Need {remaining} more windows...\n")
                    continue
                
                # Make prediction
                self._process_update(update_num, features)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.collector.stop_collection()
            self._print_summary()
    
    def _process_update(self, update_num, features):
        """Process monitoring update"""
        # Predict state
        state, confidence = self._predict()
        
        # Check anomaly
        is_anomaly, anomaly_score, severity = self.user_profile.check_anomaly(features)
        
        # Store update
        update = {
            'number': update_num,
            'time': datetime.now(),
            'state': state,
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'severity': severity,
            'features': features
        }
        
        self.updates.append(update)
        if is_anomaly:
            self.anomalies.append(update)
        
        # Display
        self._display_update(update)
    
    def _predict(self):
        """Make prediction"""
        sequence = np.array(list(self.feature_buffer), dtype=np.float32)
        
        seq_flat = sequence.reshape(-1, 27)
        seq_norm = self.scaler.transform(seq_flat)
        sequence = seq_norm.reshape(1, 5, 27)
        
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        
        return self.states[predicted.item()], confidence.item()
    
    def _display_update(self, update):
        """Display update"""
        print("\n" + "="*80)
        print(f"UPDATE #{update['number']:02d} | {update['time'].strftime('%I:%M:%S %p')}")
        print("="*80)
        
        print(f"\n📊 STATE: {update['state'].upper()}")
        print(f"   Confidence: {update['confidence']*100:.1f}%")
        
        features = update['features']
        print(f"\n📈 LIVE METRICS:")
        print(f"   Typing: {features[2]:.0f} WPM | {int(features[0])} keys | {features[1]:.2f} keys/sec")
        print(f"   Mouse: {features[8]:.0f}px moved | {int(features[14])} clicks | {int(features[12])} direction changes")
        print(f"   Apps: {int(features[23])} switches | {int(features[25])} unique | {features[26]*100:.0f}% focus")
        
        if update['is_anomaly']:
            print(f"\n🚨 ANOMALY: {update['severity']}")
            print(f"   Score: {update['anomaly_score']:.2f}")
        else:
            print(f"\n✓ {update['severity']}")
        
        print("="*80 + "\n")
    
    def _print_summary(self):
        """Print summary"""
        if not self.updates:
            return
        
        duration = (datetime.now() - self.session_start).seconds // 60
        
        print("\n" + "="*80)
        print(" "*30 + "SESSION SUMMARY")
        print("="*80)
        print(f"\nDuration: {duration} minutes | Updates: {len(self.updates)}")
        
        # State distribution
        state_counts = {}
        for u in self.updates:
            state_counts[u['state']] = state_counts.get(u['state'], 0) + 1
        
        print(f"\nState Breakdown:")
        for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(self.updates)) * 100
            print(f"  {state:20s} {count:2d} ({pct:5.1f}%)")
        
        print(f"\nAnomalies: {len(self.anomalies)}")
        if self.anomalies:
            print("  ⚠️  Review recommended")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    user_id = sys.argv[1] if len(sys.argv) > 1 else "default_user"
    
    monitor = LiveMonitor(user_id=user_id, model_path='trained_model.pth')
    
    if LIVE_MODE_AVAILABLE and monitor.model is not None:
        monitor.start()
