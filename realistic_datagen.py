"""
REALISTIC DATA GENERATOR (Hard Mode) - FIXED
- Fixes TypeError in np.random.normal
- Overlapping feature distributions (no easy separations)
- High variance to force temporal learning
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

class RealisticDatasetGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

        # 6 States
        self.states = ['deep_work', 'active_work', 'research', 'communication', 'distracted', 'idle']

        # DEFINING "MESSY" PROFILES
        # Format: (mean, max_cap, std_dev)
        self.profiles = {
            'deep_work': {
                'keystroke_rate': (2.5, 5.0, 1.2),
                'typing_speed_wpm': (45, 90, 15),
                'mouse_movement_count': (150, 600, 100),
                'total_distance': (2000, 8000, 2000),
                'app_productivity': (0.7, 1.0, 0.25),
                'scroll_rate': (0.1, 0.5, 0.2),
                'is_idle': (0, 0, 0.05)
            },
            'active_work': {
                'keystroke_rate': (1.5, 4.0, 1.0),
                'typing_speed_wpm': (25, 60, 12),
                'mouse_movement_count': (400, 1000, 200),
                'total_distance': (5000, 15000, 3000),
                'app_productivity': (0.5, 0.9, 0.2),
                'scroll_rate': (0.3, 0.8, 0.3),
                'is_idle': (0, 0, 0)
            },
            'research': {
                'keystroke_rate': (0.5, 2.0, 0.8),
                'typing_speed_wpm': (10, 40, 10),
                'mouse_movement_count': (300, 800, 200),
                'total_distance': (4000, 12000, 2500),
                'app_productivity': (0.4, 0.8, 0.3),
                'scroll_rate': (0.8, 2.0, 0.6),
                'is_idle': (0, 0, 0)
            },
            'communication': {
                'keystroke_rate': (1.0, 3.0, 1.0),
                'typing_speed_wpm': (20, 55, 12),
                'mouse_movement_count': (200, 600, 150),
                'total_distance': (3000, 9000, 2000),
                'app_productivity': (0.5, 0.9, 0.2),
                'scroll_rate': (0.2, 0.6, 0.2),
                'is_idle': (0.1, 0.2, 0.1)
            },
            'distracted': {
                'keystroke_rate': (0.5, 2.5, 1.0),
                'typing_speed_wpm': (10, 50, 15),
                'mouse_movement_count': (400, 1200, 300),
                'total_distance': (6000, 20000, 5000),
                'app_productivity': (0.1, 0.5, 0.3),
                'scroll_rate': (0.5, 1.5, 0.5),
                'is_idle': (0, 0, 0)
            },
            'idle': {
                'keystroke_rate': (0, 0.5, 0.2),
                'typing_speed_wpm': (0, 5, 2),
                'mouse_movement_count': (0, 100, 50),
                'total_distance': (0, 1000, 400),
                'app_productivity': (0, 1.0, 0.5),
                'scroll_rate': (0, 0.2, 0.1),
                'is_idle': (0.9, 1.0, 0.1)
            }
        }

        self.feature_names = [
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

    def _generate_noisy_features(self, state):
        p = self.profiles[state]
        f = {}

        # Helper to generate normal distribution from (mean, cap, std)
        def get_val(key):
            mean, cap, std = p[key]
            val = np.random.normal(mean, std)
            return val  # We rely on specific logic below to clip/max

        # 1. Base Generators (The "Nerfed" ones)
        # Using max(0, val) to prevent negative rates
        f['keystroke_rate'] = max(0, get_val('keystroke_rate'))
        f['typing_speed_wpm'] = max(0, get_val('typing_speed_wpm'))
        f['mouse_movement_count'] = max(0, get_val('mouse_movement_count'))
        f['total_distance'] = max(0, get_val('total_distance'))
        f['app_productivity'] = np.clip(get_val('app_productivity'), 0, 1)
        f['scroll_rate'] = max(0, get_val('scroll_rate'))
        f['is_idle'] = np.clip(get_val('is_idle'), 0, 1)

        # 2. Derived Features (Adding noise to relationships)
        f['keystroke_count'] = f['keystroke_rate'] * 60
        f['keystroke_variance'] = np.random.uniform(0.01, 0.2)
        f['keystroke_burst_ratio'] = np.random.uniform(0.1, 0.9)
        f['avg_inter_keystroke'] = 1.0 / (f['keystroke_rate'] + 1e-5)
        f['alpha_ratio'] = np.random.uniform(0.5, 0.9)

        f['avg_distance_per_move'] = f['total_distance'] / (f['mouse_movement_count'] + 1)
        f['coverage_score'] = np.clip(f['mouse_movement_count'] / 1000, 0, 1) * np.random.uniform(0.5, 1.2)
        f['x_variance'] = f['total_distance'] * np.random.uniform(1, 3)
        f['y_variance'] = f['total_distance'] * np.random.uniform(0.5, 2)
        f['direction_changes'] = f['mouse_movement_count'] * np.random.uniform(0.05, 0.2)

        f['click_rate'] = f['mouse_movement_count'] * np.random.uniform(0.001, 0.05)
        f['click_count'] = f['click_rate'] * 60
        f['avg_click_interval'] = np.random.uniform(1, 20)
        f['click_variance'] = np.random.uniform(0.5, 5.0)

        f['scroll_count'] = f['scroll_rate'] * 60
        f['total_scroll'] = f['scroll_count'] * np.random.uniform(10, 50)
        f['avg_scroll'] = f['total_scroll'] / (f['scroll_count'] + 1)

        # Apps - MAKE THIS UNRELIABLE
        f['app_switch_count'] = np.random.poisson(2) if state in ['deep_work', 'idle'] else np.random.poisson(6)
        f['unique_apps'] = max(1, f['app_switch_count'] // 2)
        f['focus_stability'] = 1.0 / (f['app_switch_count'] + 1)
        f['productive_ratio'] = f['app_productivity'] * np.random.uniform(0.8, 1.2)

        f['digit_ratio'] = np.random.uniform(0, 0.2)
        f['special_ratio'] = np.random.uniform(0, 0.2)

        return [float(f[name]) for name in self.feature_names]

    def generate_dataset(self, num_users=30, sessions_per_user=20, windows_per_session=60):
        print(f"Generating Realistic (Hard) Data for {num_users} users...")
        data = []

        transitions = {
            'deep_work':     [0.85, 0.05, 0.05, 0.02, 0.02, 0.01],
            'active_work':   [0.10, 0.75, 0.05, 0.05, 0.03, 0.02],
            'research':      [0.05, 0.10, 0.75, 0.05, 0.03, 0.02],
            'communication': [0.05, 0.10, 0.05, 0.70, 0.05, 0.05],
            'distracted':    [0.02, 0.05, 0.05, 0.05, 0.80, 0.03],
            'idle':          [0.05, 0.05, 0.05, 0.05, 0.05, 0.75]
        }

        for user in range(num_users):
            for sess in range(sessions_per_user):
                curr_state = np.random.choice(self.states)
                for win in range(windows_per_session):
                    feats = self._generate_noisy_features(curr_state)
                    sample = {
                        'user_id': user,
                        'session_id': f"u{user}_s{sess}",
                        'window_id': win,
                        'state': curr_state,
                        'timestamp': datetime.now().isoformat()
                    }
                    for i, name in enumerate(self.feature_names):
                        sample[name] = feats[i]
                    data.append(sample)

                    probs = transitions[curr_state]
                    curr_state = np.random.choice(self.states, p=probs)

        return data

    def save(self, data):
        Path("datasets").mkdir(exist_ok=True)
        with open("datasets/improved_dataset.json", "w") as f:
            json.dump(data, f)
        print("✅ Saved realistic dataset to datasets/improved_dataset.json")

if __name__ == "__main__":
    gen = RealisticDatasetGenerator()
    data = gen.generate_dataset()
    gen.save(data)