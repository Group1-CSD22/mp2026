"""
Enhanced Privacy-Preserving Data Collection Module
Includes app tracking, detailed metrics, and 3-minute windows
"""

import time
import numpy as np
from pynput import keyboard, mouse
from collections import deque
from datetime import datetime
import threading
import json
import psutil
import win32gui
import win32process

class EnhancedBehavioralCollector:
    """Collects behavioral features with app tracking and rich metrics"""
    
    def __init__(self, window_size=180):  # 3 minutes = 180 seconds
        self.window_size = window_size
        self.is_collecting = False
        
        # Raw event storage
        self.keystroke_times = deque(maxlen=5000)
        self.mouse_positions = deque(maxlen=5000)
        self.mouse_clicks = deque(maxlen=2000)
        self.scroll_events = deque(maxlen=2000)
        
        # Application tracking
        self.app_switches = deque(maxlen=500)
        self.current_app = None
        self.app_durations = {}
        self.last_app_check = time.time()
        
        # Listeners
        self.kb_listener = None
        self.mouse_listener = None
        
        # Feature computation thread
        self.compute_thread = None
        self.app_tracking_thread = None
        self.features_queue = deque(maxlen=100)
        
        # Session metrics
        self.session_metrics = {
            'total_keystrokes': 0,
            'total_clicks': 0,
            'total_scrolls': 0,
            'total_mouse_distance': 0,
            'apps_used': set(),
            'app_switches_count': 0
        }
        
    def start_collection(self):
        """Start collecting behavioral data"""
        self.is_collecting = True
        
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
        
        # Start feature computation thread
        self.compute_thread = threading.Thread(target=self._compute_features_loop, daemon=True)
        self.compute_thread.start()
        
        # Start app tracking thread
        self.app_tracking_thread = threading.Thread(target=self._track_apps_loop, daemon=True)
        self.app_tracking_thread.start()
        
        print("✓ Enhanced data collection started (3-minute windows)")
        
    def stop_collection(self):
        """Stop collecting data"""
        self.is_collecting = False
        
        if self.kb_listener:
            self.kb_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
            
        print("✓ Data collection stopped")
        
    def _on_key_press(self, key):
        """Record keystroke timing"""
        if self.is_collecting:
            self.keystroke_times.append(time.time())
            self.session_metrics['total_keystrokes'] += 1
            
    def _on_mouse_move(self, x, y):
        """Record mouse position"""
        if self.is_collecting:
            self.mouse_positions.append((time.time(), x, y))
            
    def _on_mouse_click(self, x, y, button, pressed):
        """Record mouse clicks"""
        if self.is_collecting and pressed:
            self.mouse_clicks.append((time.time(), x, y))
            self.session_metrics['total_clicks'] += 1
            
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Record scroll events"""
        if self.is_collecting:
            self.scroll_events.append((time.time(), dy))
            self.session_metrics['total_scrolls'] += 1
    
    def _get_active_window_info(self):
        """Get current active application name"""
        try:
            window = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(window)
            process = psutil.Process(pid)
            app_name = process.name()
            window_title = win32gui.GetWindowText(window)
            return app_name, window_title
        except:
            return "Unknown", "Unknown"
    
    def _track_apps_loop(self):
        """Track application usage continuously"""
        while self.is_collecting:
            try:
                app_name, window_title = self._get_active_window_info()
                current_time = time.time()
                
                if app_name != self.current_app:
                    # App switch detected
                    if self.current_app is not None:
                        self.app_switches.append({
                            'from': self.current_app,
                            'to': app_name,
                            'time': current_time
                        })
                        self.session_metrics['app_switches_count'] += 1
                    
                    # Update current app
                    self.current_app = app_name
                    self.session_metrics['apps_used'].add(app_name)
                
                # Track duration
                if app_name not in self.app_durations:
                    self.app_durations[app_name] = 0
                
                time_diff = current_time - self.last_app_check
                self.app_durations[app_name] += time_diff
                self.last_app_check = current_time
                
            except Exception as e:
                pass
            
            time.sleep(1)  # Check every second
    
    def _compute_features_loop(self):
        """Compute features every 3 minutes"""
        while self.is_collecting:
            time.sleep(self.window_size)
            features = self._extract_features()
            if features:
                self.features_queue.append(features)
            
    def _extract_features(self):
        """Extract comprehensive behavioral features"""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # Filter events in window
        recent_keystrokes = [t for t in self.keystroke_times if t >= window_start]
        recent_positions = [(t, x, y) for t, x, y in self.mouse_positions if t >= window_start]
        recent_clicks = [(t, x, y) for t, x, y in self.mouse_clicks if t >= window_start]
        recent_scrolls = [(t, d) for t, d in self.scroll_events if t >= window_start]
        recent_app_switches = [s for s in self.app_switches if s['time'] >= window_start]
        
        # Compute all feature categories
        keystroke_features = self._compute_keystroke_features(recent_keystrokes)
        mouse_features = self._compute_mouse_features(recent_positions)
        click_features = self._compute_click_features(recent_clicks)
        scroll_features = self._compute_scroll_features(recent_scrolls)
        app_features = self._compute_app_features(recent_app_switches)
        
        # Additional rich metrics for UI
        rich_metrics = self._compute_rich_metrics(
            recent_keystrokes, recent_positions, recent_clicks
        )
        
        features = {
            'timestamp': datetime.now().isoformat(),
            'window_duration': self.window_size,
            
            # Core features for ML
            **keystroke_features,
            **mouse_features,
            **click_features,
            **scroll_features,
            **app_features,
            
            # Rich metrics for UI/analysis
            'rich_metrics': rich_metrics
        }
        
        return features
    
    def _compute_keystroke_features(self, keystrokes):
        """Enhanced keystroke features"""
        if len(keystrokes) < 2:
            return {
                'keystroke_count': 0,
                'keystroke_rate': 0.0,
                'typing_speed_wpm': 0.0,
                'keystroke_variance': 0.0,
                'keystroke_burst_ratio': 0.0,
                'avg_inter_keystroke_time': 0.0,
                'keystroke_rhythm_std': 0.0
            }
        
        intervals = np.diff(keystrokes)
        
        # Calculate WPM (assuming 5 chars per word, 1 keystroke = 1 char)
        wpm = (len(keystrokes) / 5) / (self.window_size / 60)
        
        return {
            'keystroke_count': len(keystrokes),
            'keystroke_rate': len(keystrokes) / self.window_size,
            'typing_speed_wpm': float(wpm),
            'keystroke_variance': float(np.var(intervals)) if len(intervals) > 0 else 0.0,
            'keystroke_burst_ratio': float(np.sum(intervals < 0.2) / len(intervals)) if len(intervals) > 0 else 0.0,
            'avg_inter_keystroke_time': float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
            'keystroke_rhythm_std': float(np.std(intervals)) if len(intervals) > 1 else 0.0
        }
    
    def _compute_mouse_features(self, positions):
        """Enhanced mouse features"""
        if len(positions) < 2:
            return {
                'mouse_movement_count': 0,
                'mouse_distance': 0.0,
                'mouse_velocity': 0.0,
                'mouse_velocity_std': 0.0,
                'mouse_acceleration': 0.0,
                'mouse_direction_changes': 0,
                'mouse_idle_ratio': 1.0
            }
        
        distances = []
        velocities = []
        
        for i in range(1, len(positions)):
            t1, x1, y1 = positions[i-1]
            t2, x2, y2 = positions[i]
            
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            time_diff = t2 - t1
            
            if time_diff > 0:
                vel = dist / time_diff
                distances.append(dist)
                velocities.append(vel)
        
        # Calculate idle time (no mouse movement)
        idle_threshold = 5.0  # 5 seconds
        time_gaps = []
        for i in range(1, len(positions)):
            time_gaps.append(positions[i][0] - positions[i-1][0])
        
        idle_time = sum([gap for gap in time_gaps if gap > idle_threshold])
        idle_ratio = idle_time / self.window_size if time_gaps else 0
        
        return {
            'mouse_movement_count': len(positions),
            'mouse_distance': float(np.sum(distances)) if distances else 0.0,
            'mouse_velocity': float(np.mean(velocities)) if velocities else 0.0,
            'mouse_velocity_std': float(np.std(velocities)) if len(velocities) > 1 else 0.0,
            'mouse_acceleration': float(np.std(velocities)) if len(velocities) > 1 else 0.0,
            'mouse_direction_changes': self._count_direction_changes(positions),
            'mouse_idle_ratio': float(idle_ratio)
        }
    
    def _count_direction_changes(self, positions):
        """Count direction changes"""
        if len(positions) < 3:
            return 0
        
        changes = 0
        for i in range(2, len(positions)):
            _, x1, y1 = positions[i-2]
            _, x2, y2 = positions[i-1]
            _, x3, y3 = positions[i]
            
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
    
    def _compute_click_features(self, clicks):
        """Enhanced click features"""
        if len(clicks) < 2:
            return {
                'click_count': len(clicks),
                'click_rate': len(clicks) / self.window_size if self.window_size > 0 else 0,
                'click_clustering': 0.0,
                'avg_click_interval': 0.0,
                'click_pattern_regularity': 0.0
            }
        
        times = [t for t, _, _ in clicks]
        intervals = np.diff(times)
        
        clustering = float(np.std(intervals)) if len(intervals) > 1 else 0.0
        regularity = 1.0 / (1.0 + clustering) if clustering > 0 else 1.0
        
        return {
            'click_count': len(clicks),
            'click_rate': len(clicks) / self.window_size,
            'click_clustering': clustering,
            'avg_click_interval': float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
            'click_pattern_regularity': float(regularity)
        }
    
    def _compute_scroll_features(self, scrolls):
        """Enhanced scroll features"""
        if len(scrolls) == 0:
            return {
                'scroll_count': 0,
                'scroll_rate': 0.0,
                'scroll_intensity': 0.0,
                'scroll_direction_changes': 0
            }
        
        directions = [d for _, d in scrolls]
        
        # Count direction changes
        dir_changes = sum(1 for i in range(1, len(directions)) 
                         if (directions[i] > 0) != (directions[i-1] > 0))
        
        return {
            'scroll_count': len(scrolls),
            'scroll_rate': len(scrolls) / self.window_size,
            'scroll_intensity': float(np.mean(np.abs(directions))),
            'scroll_direction_changes': dir_changes
        }
    
    def _compute_app_features(self, app_switches):
        """Application usage features"""
        return {
            'app_switch_count': len(app_switches),
            'app_switch_rate': len(app_switches) / (self.window_size / 60),  # per minute
            'unique_apps_count': len(set([s['from'] for s in app_switches] + [s['to'] for s in app_switches])) if app_switches else 1,
            'app_focus_stability': 1.0 / (1.0 + len(app_switches)) if app_switches else 1.0
        }
    
    def _compute_rich_metrics(self, keystrokes, positions, clicks):
        """Compute rich metrics for UI display"""
        metrics = {
            'typing_speed_wpm': 0.0,
            'avg_keys_per_minute': 0.0,
            'mouse_activity_score': 0.0,
            'interaction_intensity': 0.0,
            'activity_level': 'idle'
        }
        
        # Typing speed
        if len(keystrokes) >= 5:
            wpm = (len(keystrokes) / 5) / (self.window_size / 60)
            metrics['typing_speed_wpm'] = round(wpm, 1)
            metrics['avg_keys_per_minute'] = round(len(keystrokes) / (self.window_size / 60), 1)
        
        # Mouse activity score (0-100)
        if len(positions) > 0:
            mouse_score = min(100, (len(positions) / (self.window_size * 2)) * 100)
            metrics['mouse_activity_score'] = round(mouse_score, 1)
        
        # Overall interaction intensity
        total_interactions = len(keystrokes) + len(clicks) + len(positions) / 10
        intensity = min(100, (total_interactions / self.window_size) * 10)
        metrics['interaction_intensity'] = round(intensity, 1)
        
        # Activity level classification
        if intensity > 70:
            metrics['activity_level'] = 'very_high'
        elif intensity > 50:
            metrics['activity_level'] = 'high'
        elif intensity > 30:
            metrics['activity_level'] = 'moderate'
        elif intensity > 10:
            metrics['activity_level'] = 'low'
        else:
            metrics['activity_level'] = 'idle'
        
        return metrics
    
    def get_latest_features(self):
        """Get most recent feature vector"""
        if len(self.features_queue) > 0:
            return self.features_queue[-1]
        return None
    
    def get_session_summary(self):
        """Get overall session summary"""
        return {
            'total_keystrokes': self.session_metrics['total_keystrokes'],
            'total_clicks': self.session_metrics['total_clicks'],
            'total_scrolls': self.session_metrics['total_scrolls'],
            'apps_used': list(self.session_metrics['apps_used']),
            'app_switches': self.session_metrics['app_switches_count'],
            'app_durations': {app: round(dur, 1) for app, dur in self.app_durations.items()}
        }
    
    def save_features(self, filename):
        """Save collected features"""
        features = list(self.features_queue)
        with open(filename, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"✓ Saved {len(features)} feature vectors to {filename}")


if __name__ == "__main__":
    collector = EnhancedBehavioralCollector(window_size=180)
    
    print("Starting data collection for 6 minutes (2 windows)...")
    collector.start_collection()
    
    try:
        time.sleep(360)  # 6 minutes
    except KeyboardInterrupt:
        pass
    
    collector.stop_collection()
    
    print("\nSession Summary:")
    print(json.dumps(collector.get_session_summary(), indent=2))
    
    collector.save_features("collected_features.json")
