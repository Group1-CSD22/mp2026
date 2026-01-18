"""
Improved Synthetic Dataset Generator
- 30 features (position-based mouse, app categories)
- 6 well-defined states (removed ambiguous "meeting")
- 30,000+ samples with balanced distribution
- Realistic transitions
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path


class ImprovedDatasetGenerator:
    """Generate high-quality synthetic data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # NEW 6-STATE SYSTEM (removed ambiguous "meeting")
        self.states = {
            'deep_work': {
                'description': 'Intense focused work - coding, writing, analysis',
                'characteristics': 'High typing, low app switching, productive apps'
            },
            'active_work': {
                'description': 'Regular productive work with some multitasking',
                'characteristics': 'Moderate typing/clicking, mixed apps'
            },
            'research': {
                'description': 'Reading, browsing, information gathering',
                'characteristics': 'High scrolling, browser usage, low typing'
            },
            'communication': {
                'description': 'Emails, chat, video calls (clear indicators)',
                'characteristics': 'Communication apps, moderate typing, stable focus'
            },
            'distracted': {
                'description': 'Off-task, social media, entertainment',
                'characteristics': 'Low productivity apps, erratic patterns'
            },
            'idle': {
                'description': 'Away from computer or very minimal activity',
                'characteristics': 'Near-zero activity across all metrics'
            }
        }
        
        # Profiles for 30 features
        self.profiles = {
            'deep_work': {
                # Keyboard (7)
                'keystroke_count': (120, 250, 30),
                'keystroke_rate': (2.0, 4.2, 0.5),
                'typing_speed_wpm': (40, 65, 8),
                'keystroke_variance': (0.01, 0.04, 0.01),
                'keystroke_burst_ratio': (0.5, 0.8, 0.1),
                'avg_inter_keystroke': (0.05, 0.10, 0.02),
                'alpha_ratio': (0.7, 0.9, 0.08),
                
                # Mouse position-based (8)
                'mouse_movement_count': (200, 500, 80),
                'total_distance': (3000, 8000, 1200),
                'avg_distance_per_move': (10, 30, 5),
                'coverage_score': (0.3, 0.7, 0.15),
                'x_variance': (5000, 15000, 2500),
                'y_variance': (3000, 10000, 1800),
                'direction_changes': (15, 40, 8),
                'is_idle': (0, 0, 0),
                
                # Click (3)
                'click_count': (10, 30, 6),
                'click_rate': (0.15, 0.50, 0.12),
                'avg_click_interval': (5, 15, 3),
                
                # Scroll (3)
                'scroll_count': (8, 25, 6),
                'scroll_rate': (0.12, 0.42, 0.10),
                'total_scroll': (50, 150, 30),
                
                # App (6)
                'app_switch_count': (0, 3, 1),
                'unique_apps': (1, 2, 0.5),
                'app_productivity': (0.85, 1.0, 0.08),  # High productivity apps
                'focus_stability': (0.8, 1.0, 0.08),
                'productive_ratio': (0.85, 1.0, 0.08),
                'digit_ratio': (0.15, 0.35, 0.08),
                
                # Additional (3)
                'special_ratio': (0.05, 0.15, 0.04),
                'click_variance': (0.5, 2.0, 0.5),
                'avg_scroll': (2, 6, 1.5)
            },
            
            'active_work': {
                'keystroke_count': (60, 150, 25),
                'keystroke_rate': (1.0, 2.5, 0.4),
                'typing_speed_wpm': (25, 45, 7),
                'keystroke_variance': (0.02, 0.06, 0.015),
                'keystroke_burst_ratio': (0.3, 0.6, 0.12),
                'avg_inter_keystroke': (0.08, 0.15, 0.025),
                'alpha_ratio': (0.6, 0.85, 0.10),
                
                'mouse_movement_count': (300, 700, 100),
                'total_distance': (5000, 15000, 2500),
                'avg_distance_per_move': (12, 35, 7),
                'coverage_score': (0.4, 0.9, 0.18),
                'x_variance': (8000, 25000, 4000),
                'y_variance': (5000, 18000, 3000),
                'direction_changes': (25, 65, 12),
                'is_idle': (0, 0, 0),
                
                'click_count': (15, 45, 10),
                'click_rate': (0.25, 0.75, 0.15),
                'avg_click_interval': (3, 10, 2.5),
                
                'scroll_count': (12, 40, 9),
                'scroll_rate': (0.20, 0.67, 0.15),
                'total_scroll': (70, 220, 45),
                
                'app_switch_count': (2, 7, 2),
                'unique_apps': (2, 4, 0.8),
                'app_productivity': (0.65, 0.90, 0.12),
                'focus_stability': (0.5, 0.8, 0.12),
                'productive_ratio': (0.65, 0.90, 0.12),
                'digit_ratio': (0.12, 0.30, 0.08),
                
                'special_ratio': (0.08, 0.20, 0.05),
                'click_variance': (1.0, 3.5, 0.8),
                'avg_scroll': (3, 8, 2)
            },
            
            'research': {
                'keystroke_count': (15, 60, 15),
                'keystroke_rate': (0.25, 1.0, 0.25),
                'typing_speed_wpm': (8, 20, 5),
                'keystroke_variance': (0.04, 0.12, 0.03),
                'keystroke_burst_ratio': (0.1, 0.3, 0.08),
                'avg_inter_keystroke': (0.15, 0.40, 0.08),
                'alpha_ratio': (0.5, 0.75, 0.12),
                
                'mouse_movement_count': (250, 600, 100),
                'total_distance': (4000, 12000, 2200),
                'avg_distance_per_move': (10, 28, 6),
                'coverage_score': (0.35, 0.85, 0.18),
                'x_variance': (6000, 20000, 3500),
                'y_variance': (4000, 15000, 2800),
                'direction_changes': (20, 55, 12),
                'is_idle': (0, 0, 0),
                
                'click_count': (8, 22, 5),
                'click_rate': (0.12, 0.37, 0.10),
                'avg_click_interval': (6, 18, 4),
                
                'scroll_count': (30, 80, 18),  # High scrolling!
                'scroll_rate': (0.50, 1.33, 0.28),
                'total_scroll': (180, 500, 100),
                
                'app_switch_count': (1, 5, 1.5),
                'unique_apps': (1, 3, 0.7),
                'app_productivity': (0.45, 0.70, 0.12),  # Browser-heavy
                'focus_stability': (0.6, 0.85, 0.12),
                'productive_ratio': (0.50, 0.75, 0.12),
                'digit_ratio': (0.05, 0.18, 0.06),
                
                'special_ratio': (0.12, 0.28, 0.08),
                'click_variance': (1.5, 4.0, 1.0),
                'avg_scroll': (5, 12, 3)
            },
            
            'communication': {
                'keystroke_count': (40, 110, 22),
                'keystroke_rate': (0.67, 1.83, 0.37),
                'typing_speed_wpm': (15, 35, 8),
                'keystroke_variance': (0.03, 0.09, 0.025),
                'keystroke_burst_ratio': (0.2, 0.45, 0.10),
                'avg_inter_keystroke': (0.10, 0.25, 0.05),
                'alpha_ratio': (0.65, 0.85, 0.10),
                
                'mouse_movement_count': (150, 400, 75),
                'total_distance': (2500, 8000, 1600),
                'avg_distance_per_move': (8, 25, 5),
                'coverage_score': (0.25, 0.65, 0.15),
                'x_variance': (4000, 12000, 2200),
                'y_variance': (3000, 9000, 1800),
                'direction_changes': (12, 35, 8),
                'is_idle': (0, 0.05, 0.02),  # Slight idle during calls
                
                'click_count': (8, 25, 6),
                'click_rate': (0.12, 0.42, 0.12),
                'avg_click_interval': (5, 18, 4),
                
                'scroll_count': (5, 20, 6),
                'scroll_rate': (0.08, 0.33, 0.10),
                'total_scroll': (30, 120, 35),
                
                'app_switch_count': (1, 4, 1.2),
                'unique_apps': (1, 2, 0.5),
                'app_productivity': (0.65, 0.80, 0.10),  # Communication apps
                'focus_stability': (0.7, 0.95, 0.12),
                'productive_ratio': (0.70, 0.85, 0.10),
                'digit_ratio': (0.08, 0.22, 0.07),
                
                'special_ratio': (0.10, 0.25, 0.07),
                'click_variance': (1.2, 3.2, 0.8),
                'avg_scroll': (2, 7, 2)
            },
            
            'distracted': {
                'keystroke_count': (20, 80, 20),
                'keystroke_rate': (0.33, 1.33, 0.33),
                'typing_speed_wpm': (10, 25, 6),
                'keystroke_variance': (0.05, 0.15, 0.04),
                'keystroke_burst_ratio': (0.15, 0.40, 0.10),
                'avg_inter_keystroke': (0.12, 0.35, 0.08),
                'alpha_ratio': (0.45, 0.70, 0.12),
                
                'mouse_movement_count': (350, 850, 150),
                'total_distance': (6000, 18000, 3500),
                'avg_distance_per_move': (15, 40, 8),
                'coverage_score': (0.5, 1.2, 0.25),
                'x_variance': (10000, 30000, 5500),
                'y_variance': (7000, 22000, 4200),
                'direction_changes': (35, 95, 20),
                'is_idle': (0, 0, 0),
                
                'click_count': (20, 60, 14),
                'click_rate': (0.33, 1.0, 0.23),
                'avg_click_interval': (2, 7, 2),
                
                'scroll_count': (25, 70, 16),
                'scroll_rate': (0.42, 1.17, 0.27),
                'total_scroll': (140, 400, 90),
                
                'app_switch_count': (5, 15, 3),  # High switching!
                'unique_apps': (3, 7, 1.5),
                'app_productivity': (0.15, 0.40, 0.12),  # Low productivity apps
                'focus_stability': (0.2, 0.5, 0.12),
                'productive_ratio': (0.15, 0.40, 0.12),
                'digit_ratio': (0.10, 0.28, 0.08),
                
                'special_ratio': (0.15, 0.35, 0.10),
                'click_variance': (0.8, 2.5, 0.7),
                'avg_scroll': (4, 10, 2.5)
            },
            
            'idle': {
                'keystroke_count': (0, 8, 3),
                'keystroke_rate': (0, 0.13, 0.05),
                'typing_speed_wpm': (0, 3, 1),
                'keystroke_variance': (0, 0.02, 0.01),
                'keystroke_burst_ratio': (0, 0.08, 0.03),
                'avg_inter_keystroke': (0.5, 3.0, 0.8),
                'alpha_ratio': (0, 0.5, 0.2),
                
                'mouse_movement_count': (0, 50, 20),
                'total_distance': (0, 500, 200),
                'avg_distance_per_move': (0, 15, 8),
                'coverage_score': (0, 0.15, 0.08),
                'x_variance': (0, 2000, 1000),
                'y_variance': (0, 1500, 800),
                'direction_changes': (0, 8, 4),
                'is_idle': (0.8, 1.0, 0.08),  # Marked as idle
                
                'click_count': (0, 5, 2),
                'click_rate': (0, 0.08, 0.03),
                'avg_click_interval': (15, 60, 15),
                
                'scroll_count': (0, 6, 3),
                'scroll_rate': (0, 0.10, 0.05),
                'total_scroll': (0, 40, 18),
                
                'app_switch_count': (0, 2, 1),
                'unique_apps': (0, 1, 0.5),
                'app_productivity': (0, 0.5, 0.25),
                'focus_stability': (0.7, 1.0, 0.15),
                'productive_ratio': (0, 0.5, 0.25),
                'digit_ratio': (0, 0.15, 0.08),
                
                'special_ratio': (0, 0.20, 0.10),
                'click_variance': (0, 1.5, 0.8),
                'avg_scroll': (0, 4, 2)
            }
        }
    
    def generate_sample(self, state):
        """Generate single sample"""
        profile = self.profiles[state]
        sample = {'state': state}
        
        for feature, (mean, max_val, std) in profile.items():
            value = np.random.normal(mean, std)
            value = np.clip(value, 0, max_val * 1.2)
            sample[feature] = round(float(value), 4)
        
        return sample
    
    def generate_session(self, num_windows=60):
        """Generate realistic session with state transitions"""
        # Improved transition matrix (removed meeting bias)
        transitions = {
            'deep_work': {
                'deep_work': 0.70,
                'active_work': 0.15,
                'research': 0.08,
                'communication': 0.03,
                'distracted': 0.02,
                'idle': 0.02
            },
            'active_work': {
                'active_work': 0.55,
                'deep_work': 0.20,
                'research': 0.12,
                'communication': 0.05,
                'distracted': 0.05,
                'idle': 0.03
            },
            'research': {
                'research': 0.60,
                'active_work': 0.18,
                'deep_work': 0.10,
                'communication': 0.04,
                'distracted': 0.05,
                'idle': 0.03
            },
            'communication': {
                'communication': 0.65,
                'active_work': 0.20,
                'deep_work': 0.05,
                'research': 0.05,
                'distracted': 0.02,
                'idle': 0.03
            },
            'distracted': {
                'distracted': 0.50,
                'active_work': 0.20,
                'research': 0.15,
                'idle': 0.08,
                'deep_work': 0.05,
                'communication': 0.02
            },
            'idle': {
                'idle': 0.60,
                'active_work': 0.18,
                'deep_work': 0.10,
                'research': 0.08,
                'distracted': 0.03,
                'communication': 0.01
            }
        }
        
        # Start with active_work (most common)
        current_state = 'active_work'
        samples = []
        
        for _ in range(num_windows):
            sample = self.generate_sample(current_state)
            samples.append(sample)
            
            # Transition
            probs = transitions[current_state]
            current_state = np.random.choice(
                list(probs.keys()),
                p=list(probs.values())
            )
        
        return samples
    
    def generate_dataset(self, num_users=25, sessions_per_user=20, windows_per_session=60):
        """Generate large balanced dataset"""
        print(f"\nGenerating Improved Dataset...")
        print(f"Users: {num_users}")
        print(f"Sessions per user: {sessions_per_user}")
        print(f"Windows per session: {windows_per_session}")
        
        total_samples = num_users * sessions_per_user * windows_per_session
        print(f"Total samples: {total_samples:,}\n")
        
        all_samples = []
        
        for user_id in range(num_users):
            print(f"User {user_id+1}/{num_users}...", end=' ')
            
            for session in range(sessions_per_user):
                samples = self.generate_session(windows_per_session)
                
                for sample in samples:
                    sample['user_id'] = user_id
                    sample['session_id'] = f"u{user_id}_s{session}"
                    sample['timestamp'] = datetime.now().isoformat()
                
                all_samples.extend(samples)
            
            print("✓")
        
        print(f"\n✓ Generated {len(all_samples):,} samples")
        return all_samples
    
    def save_dataset(self, data, base_name="improved_dataset"):
        """Save dataset"""
        output_dir = Path("datasets")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        # Save CSV
        csv_path = output_dir / f"{base_name}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Get sizes
        json_size = json_path.stat().st_size / (1024**2)
        csv_size = csv_path.stat().st_size / (1024**2)
        
        print(f"\n✓ Saved:")
        print(f"  JSON: {json_path} ({json_size:.1f} MB)")
        print(f"  CSV: {csv_path} ({csv_size:.1f} MB)")
        
        # Statistics
        print(f"\nDataset Statistics:")
        print(f"  Total: {len(data):,}")
        print(f"  Features: 30")
        print(f"  States: 6")
        
        state_counts = df['state'].value_counts()
        print(f"\nState Distribution:")
        for state, count in state_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {state:20s}: {count:6,} ({pct:5.1f}%)")
        
        return json_path, csv_path


if __name__ == "__main__":
    gen = ImprovedDatasetGenerator()
    
    # Generate 30,000 samples
    dataset = gen.generate_dataset(
        num_users=25,
        sessions_per_user=20,
        windows_per_session=60
    )
    
    # Save
    gen.save_dataset(dataset)
    
    print("\n✓ Dataset generation complete!")
