"""
REVAMPED LIVE MONITORING SYSTEM v3.0
- 1-minute windows with model predictions each time
- Position-based mouse tracking with heatmaps
- App categorization system
- Proper user profile generation and updates
- Enhanced anomaly detection
- Manual correction for continuous learning
"""

import torch
import numpy as np
import json
import time
import threading
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
import pickle

try:
    from pynput import keyboard, mouse
    import psutil
    import win32gui
    import win32process
    LIVE_MODE_AVAILABLE = True
except ImportError:
    LIVE_MODE_AVAILABLE = False
    print("⚠️  Install dependencies: pip install pynput psutil pywin32")


class AppCategorizer:
    """Categorizes applications into productivity types"""
    
    def __init__(self):
        self.categories = {
            'code_development': [
                'pycharm', 'vscode', 'code', 'visual studio', 'intellij',
                'eclipse', 'atom', 'sublime', 'notepad++', 'vim'
            ],
            'productivity': [
                'word', 'excel', 'powerpoint', 'outlook', 'onenote',
                'notion', 'evernote', 'todoist', 'trello', 'asana'
            ],
            'communication': [
                'teams', 'slack', 'discord', 'zoom', 'skype',
                'whatsapp', 'telegram', 'messenger'
            ],
            'browser': [
                'chrome', 'firefox', 'edge', 'safari', 'brave', 'opera'
            ],
            'media': [
                'spotify', 'vlc', 'netflix', 'youtube', 'itunes',
                'media player', 'photos'
            ],
            'gaming': [
                'steam', 'epic', 'game', 'valorant', 'league',
                'minecraft', 'roblox', 'gta', 'fortnite'
            ],
            'system': [
                'explorer', 'finder', 'cmd', 'powershell', 'terminal',
                'taskmgr', 'settings'
            ]
        }
        
        # Load custom apps
        self.custom_apps_file = Path('app_categories.json')
        self.load_custom_apps()
    
    def load_custom_apps(self):
        """Load user-defined app categories"""
        if self.custom_apps_file.exists():
            with open(self.custom_apps_file, 'r') as f:
                custom = json.load(f)
                for category, apps in custom.items():
                    if category in self.categories:
                        self.categories[category].extend(apps)
    
    def add_app(self, app_name, category):
        """Add new app to category"""
        if category not in self.categories:
            self.categories[category] = []
        
        app_lower = app_name.lower()
        if app_lower not in self.categories[category]:
            self.categories[category].append(app_lower)
            self.save_custom_apps()
    
    def save_custom_apps(self):
        """Save custom categorizations"""
        with open(self.custom_apps_file, 'w') as f:
            json.dump(self.categories, f, indent=2)
    
    def categorize(self, app_name):
        """Get category for app"""
        app_lower = app_name.lower()
        
        for category, apps in self.categories.items():
            for app in apps:
                if app in app_lower:
                    return category
        
        return 'uncategorized'
    
    def get_productivity_score(self, category):
        """Get productivity weight for category"""
        weights = {
            'code_development': 1.0,
            'productivity': 0.9,
            'communication': 0.7,
            'browser': 0.5,
            'media': 0.2,
            'gaming': 0.1,
            'system': 0.6,
            'uncategorized': 0.5
        }
        return weights.get(category, 0.5)


class EnhancedDataCollector:
    """Enhanced collector with position-based tracking and app categorization"""
    
    def __init__(self, window_size=60):  # 1 minute windows
        self.window_size = window_size
        self.is_collecting = False
        
        # Position-based mouse tracking
        self.mouse_positions_raw = []  # Store actual positions
        self.mouse_heatmap = defaultdict(int)  # Position frequency
        self.screen_width = 1920  # Will detect actual
        self.screen_height = 1080
        
        # Events
        self.keystroke_times = []
        self.key_types = []  # Track key categories (alphanumeric, special, etc.)
        self.mouse_clicks = []
        self.scroll_events = []
        
        # App tracking with categories
        self.app_categorizer = AppCategorizer()
        self.current_app = None
        self.current_app_category = None
        self.app_durations = defaultdict(float)
        self.app_switches = []
        self.focused_app = None  # Currently clicked/focused app
        
        # Listeners
        self.kb_listener = None
        self.mouse_listener = None
        self.app_monitor_thread = None
        
        self.window_start = None
        self.last_activity_time = time.time()
        
    def start_collection(self):
        """Start collecting"""
        self.is_collecting = True
        self.window_start = time.time()
        
        # Clear data
        self.mouse_positions_raw = []
        self.mouse_heatmap.clear()
        self.keystroke_times = []
        self.key_types = []
        self.mouse_clicks = []
        self.scroll_events = []
        self.app_switches = []
        
        # Start listeners
        self.kb_listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        self.kb_listener.start()
        
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # App monitoring
        self.app_monitor_thread = threading.Thread(target=self._monitor_apps, daemon=True)
        self.app_monitor_thread.start()
    
    def stop_collection(self):
        """Stop collecting"""
        self.is_collecting = False
        
        if self.kb_listener:
            self.kb_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
    
    def _categorize_key(self, key):
        """Categorize key type"""
        try:
            if hasattr(key, 'char') and key.char:
                if key.char.isalpha():
                    return 'alpha'
                elif key.char.isdigit():
                    return 'digit'
                else:
                    return 'symbol'
            else:
                return 'special'
        except:
            return 'special'
    
    def _on_key_press(self, key):
        """Record keystroke with type"""
        if self.is_collecting:
            current_time = time.time()
            self.keystroke_times.append(current_time)
            self.key_types.append(self._categorize_key(key))
            self.last_activity_time = current_time
    
    def _on_mouse_move(self, x, y):
        """Record actual mouse positions"""
        if self.is_collecting:
            current_time = time.time()
            self.mouse_positions_raw.append((current_time, x, y))
            
            # Update heatmap (grid-based)
            grid_x = int(x / 100)  # 100px grid cells
            grid_y = int(y / 100)
            self.mouse_heatmap[(grid_x, grid_y)] += 1
            
            self.last_activity_time = current_time
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Record clicks and update focused app"""
        if self.is_collecting and pressed:
            current_time = time.time()
            self.mouse_clicks.append((current_time, x, y))
            self.last_activity_time = current_time
            
            # Update focused app on click
            self.focused_app = self.current_app
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Record scrolling"""
        if self.is_collecting:
            current_time = time.time()
            self.scroll_events.append((current_time, dy))
            self.last_activity_time = current_time
    
    def _get_active_app(self):
        """Get active application"""
        try:
            window = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(window)
            process = psutil.Process(pid)
            app_name = process.name()
            return app_name
        except:
            return "Unknown"
    
    def _monitor_apps(self):
        """Monitor app usage"""
        last_check = time.time()
        
        while self.is_collecting:
            try:
                app = self._get_active_app()
                current_time = time.time()
                
                if app != self.current_app:
                    # App switch
                    if self.current_app:
                        duration = current_time - last_check
                        self.app_durations[self.current_app] += duration
                        
                        self.app_switches.append({
                            'from': self.current_app,
                            'to': app,
                            'time': current_time
                        })
                    
                    self.current_app = app
                    self.current_app_category = self.app_categorizer.categorize(app)
                    last_check = current_time
                else:
                    # Same app, accumulate time
                    duration = current_time - last_check
                    self.app_durations[app] += duration
                    last_check = current_time
                
            except:
                pass
            
            time.sleep(0.5)
    
    def get_features(self):
        """Extract comprehensive features"""
        window_duration = time.time() - self.window_start
        
        # Keystroke features
        keystroke_count = len(self.keystroke_times)
        keystroke_rate = keystroke_count / window_duration if window_duration > 0 else 0
        
        # Key type distribution
        key_type_counts = defaultdict(int)
        for kt in self.key_types:
            key_type_counts[kt] += 1
        
        alpha_ratio = key_type_counts['alpha'] / max(keystroke_count, 1)
        digit_ratio = key_type_counts['digit'] / max(keystroke_count, 1)
        special_ratio = key_type_counts['special'] / max(keystroke_count, 1)
        
        # Typing speed
        typing_speed_wpm = (keystroke_count / 5) / (window_duration / 60) if window_duration > 0 else 0
        
        # Keystroke timing
        if len(self.keystroke_times) > 1:
            intervals = np.diff(self.keystroke_times)
            keystroke_variance = float(np.var(intervals))
            keystroke_burst_ratio = float(np.sum(intervals < 0.2) / len(intervals))
            avg_inter_keystroke = float(np.mean(intervals))
        else:
            keystroke_variance = 0.0
            keystroke_burst_ratio = 0.0
            avg_inter_keystroke = 0.0
        
        # Mouse features (position-based)
        mouse_movement_count = len(self.mouse_positions_raw)
        
        if len(self.mouse_positions_raw) > 1:
            positions = self.mouse_positions_raw
            
            # Calculate actual distances
            distances = []
            for i in range(1, len(positions)):
                _, x1, y1 = positions[i-1]
                _, x2, y2 = positions[i]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                distances.append(dist)
            
            total_distance = float(np.sum(distances))
            avg_distance_per_move = float(np.mean(distances)) if distances else 0
            
            # Mouse coverage (heatmap spread)
            unique_cells = len(self.mouse_heatmap)
            coverage_score = unique_cells / 100  # Normalized
            
            # Movement patterns
            x_coords = [x for _, x, y in positions]
            y_coords = [y for _, x, y in positions]
            
            x_variance = float(np.var(x_coords))
            y_variance = float(np.var(y_coords))
            
            # Direction changes
            direction_changes = 0
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
                        direction_changes += 1
        else:
            total_distance = 0.0
            avg_distance_per_move = 0.0
            coverage_score = 0.0
            x_variance = 0.0
            y_variance = 0.0
            direction_changes = 0
        
        # Click features
        click_count = len(self.mouse_clicks)
        click_rate = click_count / window_duration if window_duration > 0 else 0
        
        if len(self.mouse_clicks) > 1:
            click_times = [t for t, _, _ in self.mouse_clicks]
            click_intervals = np.diff(click_times)
            avg_click_interval = float(np.mean(click_intervals))
            click_variance = float(np.var(click_intervals))
        else:
            avg_click_interval = 0.0
            click_variance = 0.0
        
        # Scroll features
        scroll_count = len(self.scroll_events)
        scroll_rate = scroll_count / window_duration if window_duration > 0 else 0
        
        if scroll_count > 0:
            scroll_amounts = [abs(d) for _, d in self.scroll_events]
            total_scroll = float(np.sum(scroll_amounts))
            avg_scroll = float(np.mean(scroll_amounts))
        else:
            total_scroll = 0.0
            avg_scroll = 0.0
        
        # App features
        app_switch_count = len(self.app_switches)
        unique_apps = len(self.app_durations)
        
        # Current app productivity
        if self.current_app_category:
            app_productivity = self.app_categorizer.get_productivity_score(self.current_app_category)
        else:
            app_productivity = 0.5
        
        # Focused app stability
        focus_stability = 1.0 / (1.0 + app_switch_count)
        
        # Time in productive vs unproductive apps
        productive_time = 0
        total_app_time = sum(self.app_durations.values())
        
        for app, duration in self.app_durations.items():
            category = self.app_categorizer.categorize(app)
            score = self.app_categorizer.get_productivity_score(category)
            if score >= 0.7:
                productive_time += duration
        
        productive_ratio = productive_time / max(total_app_time, 1)
        
        # Idle detection
        time_since_activity = time.time() - self.last_activity_time
        is_idle = time_since_activity > 30  # 30 seconds = idle
        
        # Return 30 features
        features = [
            # Keyboard (7)
            keystroke_count, keystroke_rate, typing_speed_wpm,
            keystroke_variance, keystroke_burst_ratio, avg_inter_keystroke,
            alpha_ratio,
            
            # Mouse position-based (8)
            mouse_movement_count, total_distance, avg_distance_per_move,
            coverage_score, x_variance, y_variance, direction_changes,
            float(is_idle),
            
            # Click (3)
            click_count, click_rate, avg_click_interval,
            
            # Scroll (3)
            scroll_count, scroll_rate, total_scroll,
            
            # App (6)
            app_switch_count, unique_apps, app_productivity,
            focus_stability, productive_ratio, digit_ratio,
            
            # Additional (3)
            special_ratio, click_variance, avg_scroll
        ]
        
        metadata = {
            'is_idle': is_idle,
            'current_app': self.current_app,
            'app_category': self.current_app_category,
            'app_durations': dict(self.app_durations),
            'heatmap_cells': unique_cells if 'unique_cells' in locals() else 0
        }
        
        return features, metadata


# Continuing in next artifact due to length...
