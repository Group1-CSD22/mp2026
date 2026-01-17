"""
Optimized Synthetic Dataset Generator
Better state definitions, realistic patterns, optimized file size
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json

class OptimizedSyntheticGenerator:
    """Generate high-quality synthetic data with clear states"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # NEW STATE DEFINITIONS - Clear, measurable, non-ambiguous
        self.state_definitions = {
            'focused_work': {
                'description': 'Sustained concentration on a single task',
                'indicators': 'High typing, low app switching, stable mouse patterns'
            },
            'task_execution': {
                'description': 'Active work with moderate multitasking',
                'indicators': 'Moderate typing/clicking, some app switching'
            },
            'research_reading': {
                'description': 'Information gathering and reading',
                'indicators': 'Low typing, high scrolling, minimal clicking'
            },
            'meeting_communication': {
                'description': 'Meetings, calls, or collaborative work',
                'indicators': 'Low keyboard, minimal mouse, app stays constant'
            },
            'unproductive': {
                'description': 'Low productivity or off-task behavior',
                'indicators': 'Random patterns, frequent app switching, irregular activity'
            }
        }
        
        # Behavioral profiles for each state (all values per 3-minute window)
        self.profiles = {
            'focused_work': {
                # Keyboard features
                'keystroke_count': (200, 400, 50),
                'keystroke_rate': (1.1, 2.2, 0.3),
                'typing_speed_wpm': (30, 50, 8),
                'keystroke_variance': (0.01, 0.05, 0.01),
                'keystroke_burst_ratio': (0.4, 0.7, 0.1),
                'avg_inter_keystroke_time': (0.06, 0.12, 0.02),
                'keystroke_rhythm_std': (0.05, 0.15, 0.03),
                
                # Mouse features
                'mouse_movement_count': (300, 800, 100),
                'mouse_distance': (5000, 15000, 2000),
                'mouse_velocity': (50, 150, 30),
                'mouse_velocity_std': (10, 30, 5),
                'mouse_acceleration': (15, 40, 8),
                'mouse_direction_changes': (20, 50, 10),
                'mouse_idle_ratio': (0.1, 0.3, 0.08),
                
                # Click features
                'click_count': (15, 40, 8),
                'click_rate': (0.08, 0.22, 0.04),
                'click_clustering': (0.5, 1.2, 0.2),
                'avg_click_interval': (8, 20, 3),
                'click_pattern_regularity': (0.6, 0.9, 0.1),
                
                # Scroll features
                'scroll_count': (10, 30, 5),
                'scroll_rate': (0.05, 0.17, 0.03),
                'scroll_intensity': (1, 3, 0.5),
                'scroll_direction_changes': (2, 8, 2),
                
                # App features
                'app_switch_count': (0, 3, 1),
                'app_switch_rate': (0, 1, 0.3),
                'unique_apps_count': (1, 2, 0.5),
                'app_focus_stability': (0.7, 1.0, 0.1)
            },
            
            'task_execution': {
                'keystroke_count': (100, 250, 40),
                'keystroke_rate': (0.5, 1.4, 0.2),
                'typing_speed_wpm': (20, 35, 5),
                'keystroke_variance': (0.02, 0.08, 0.02),
                'keystroke_burst_ratio': (0.3, 0.5, 0.1),
                'avg_inter_keystroke_time': (0.1, 0.2, 0.03),
                'keystroke_rhythm_std': (0.08, 0.2, 0.04),
                
                'mouse_movement_count': (500, 1200, 150),
                'mouse_distance': (8000, 25000, 3000),
                'mouse_velocity': (80, 200, 40),
                'mouse_velocity_std': (20, 50, 10),
                'mouse_acceleration': (25, 60, 12),
                'mouse_direction_changes': (30, 80, 15),
                'mouse_idle_ratio': (0.2, 0.4, 0.1),
                
                'click_count': (25, 60, 10),
                'click_rate': (0.14, 0.33, 0.05),
                'click_clustering': (0.7, 1.5, 0.25),
                'avg_click_interval': (5, 12, 2),
                'click_pattern_regularity': (0.4, 0.7, 0.1),
                
                'scroll_count': (15, 45, 8),
                'scroll_rate': (0.08, 0.25, 0.04),
                'scroll_intensity': (1.5, 4, 0.7),
                'scroll_direction_changes': (4, 12, 3),
                
                'app_switch_count': (3, 8, 2),
                'app_switch_rate': (1, 2.7, 0.7),
                'unique_apps_count': (2, 4, 1),
                'app_focus_stability': (0.4, 0.7, 0.1)
            },
            
            'research_reading': {
                'keystroke_count': (20, 80, 20),
                'keystroke_rate': (0.1, 0.4, 0.1),
                'typing_speed_wpm': (5, 15, 3),
                'keystroke_variance': (0.05, 0.15, 0.03),
                'keystroke_burst_ratio': (0.1, 0.3, 0.08),
                'avg_inter_keystroke_time': (0.3, 0.8, 0.15),
                'keystroke_rhythm_std': (0.15, 0.4, 0.08),
                
                'mouse_movement_count': (400, 1000, 150),
                'mouse_distance': (6000, 18000, 2500),
                'mouse_velocity': (40, 120, 25),
                'mouse_velocity_std': (15, 40, 8),
                'mouse_acceleration': (10, 35, 8),
                'mouse_direction_changes': (25, 60, 12),
                'mouse_idle_ratio': (0.3, 0.6, 0.1),
                
                'click_count': (10, 25, 5),
                'click_rate': (0.05, 0.14, 0.03),
                'click_clustering': (0.3, 0.8, 0.15),
                'avg_click_interval': (12, 30, 5),
                'click_pattern_regularity': (0.5, 0.8, 0.12),
                
                'scroll_count': (40, 100, 15),
                'scroll_rate': (0.22, 0.55, 0.08),
                'scroll_intensity': (2, 5, 0.8),
                'scroll_direction_changes': (8, 20, 4),
                
                'app_switch_count': (1, 4, 1),
                'app_switch_rate': (0.3, 1.3, 0.4),
                'unique_apps_count': (1, 3, 0.8),
                'app_focus_stability': (0.6, 0.9, 0.12)
            },
            
            'meeting_communication': {
                'keystroke_count': (10, 50, 15),
                'keystroke_rate': (0.05, 0.28, 0.08),
                'typing_speed_wpm': (3, 12, 3),
                'keystroke_variance': (0.08, 0.2, 0.04),
                'keystroke_burst_ratio': (0.05, 0.2, 0.06),
                'avg_inter_keystroke_time': (0.5, 1.5, 0.3),
                'keystroke_rhythm_std': (0.2, 0.5, 0.1),
                
                'mouse_movement_count': (100, 400, 80),
                'mouse_distance': (2000, 8000, 1500),
                'mouse_velocity': (20, 80, 20),
                'mouse_velocity_std': (8, 25, 6),
                'mouse_acceleration': (5, 20, 5),
                'mouse_direction_changes': (10, 30, 8),
                'mouse_idle_ratio': (0.5, 0.8, 0.12),
                
                'click_count': (5, 15, 4),
                'click_rate': (0.03, 0.08, 0.02),
                'click_clustering': (0.2, 0.6, 0.12),
                'avg_click_interval': (20, 50, 8),
                'click_pattern_regularity': (0.7, 0.95, 0.1),
                
                'scroll_count': (5, 20, 5),
                'scroll_rate': (0.03, 0.11, 0.03),
                'scroll_intensity': (0.5, 2, 0.4),
                'scroll_direction_changes': (1, 5, 2),
                
                'app_switch_count': (0, 2, 1),
                'app_switch_rate': (0, 0.7, 0.3),
                'unique_apps_count': (1, 2, 0.4),
                'app_focus_stability': (0.8, 1.0, 0.08)
            },
            
            'unproductive': {
                'keystroke_count': (30, 120, 30),
                'keystroke_rate': (0.17, 0.67, 0.17),
                'typing_speed_wpm': (8, 22, 5),
                'keystroke_variance': (0.1, 0.25, 0.05),
                'keystroke_burst_ratio': (0.08, 0.25, 0.08),
                'avg_inter_keystroke_time': (0.4, 1.2, 0.25),
                'keystroke_rhythm_std': (0.18, 0.45, 0.1),
                
                'mouse_movement_count': (600, 1500, 200),
                'mouse_distance': (10000, 30000, 4000),
                'mouse_velocity': (100, 250, 50),
                'mouse_velocity_std': (30, 70, 15),
                'mouse_acceleration': (35, 80, 18),
                'mouse_direction_changes': (40, 100, 20),
                'mouse_idle_ratio': (0.15, 0.4, 0.1),
                
                'click_count': (20, 50, 10),
                'click_rate': (0.11, 0.28, 0.05),
                'click_clustering': (0.8, 1.8, 0.3),
                'avg_click_interval': (4, 10, 2),
                'click_pattern_regularity': (0.2, 0.5, 0.1),
                
                'scroll_count': (25, 70, 12),
                'scroll_rate': (0.14, 0.39, 0.07),
                'scroll_intensity': (2, 6, 1),
                'scroll_direction_changes': (6, 18, 4),
                
                'app_switch_count': (5, 15, 3),
                'app_switch_rate': (1.7, 5, 1),
                'unique_apps_count': (3, 6, 1.5),
                'app_focus_stability': (0.2, 0.5, 0.1)
            }
        }
    
    def generate_sample(self, state):
        """Generate single sample for given state"""
        profile = self.profiles[state]
        sample = {
            'state': state,
            'timestamp': datetime.now().isoformat()
        }
        
        for feature, (mean, max_val, std) in profile.items():
            # Truncated normal distribution
            value = np.random.normal(mean, std)
            value = np.clip(value, 0, max_val * 1.2)
            sample[feature] = round(float(value), 3)
        
        return sample
    
    def generate_realistic_session(self, duration_windows=20):
        """Generate realistic work session with state transitions"""
        # State transition probabilities (realistic work patterns)
        transitions = {
            'focused_work': {
                'focused_work': 0.65,
                'task_execution': 0.20,
                'research_reading': 0.08,
                'meeting_communication': 0.02,
                'unproductive': 0.05
            },
            'task_execution': {
                'task_execution': 0.50,
                'focused_work': 0.25,
                'research_reading': 0.12,
                'meeting_communication': 0.05,
                'unproductive': 0.08
            },
            'research_reading': {
                'research_reading': 0.55,
                'task_execution': 0.20,
                'focused_work': 0.15,
                'meeting_communication': 0.03,
                'unproductive': 0.07
            },
            'meeting_communication': {
                'meeting_communication': 0.70,
                'task_execution': 0.15,
                'focused_work': 0.08,
                'research_reading': 0.05,
                'unproductive': 0.02
            },
            'unproductive': {
                'unproductive': 0.45,
                'task_execution': 0.25,
                'research_reading': 0.15,
                'focused_work': 0.10,
                'meeting_communication': 0.05
            }
        }
        
        # Start with task_execution (most common starting state)
        current_state = 'task_execution'
        samples = []
        
        for _ in range(duration_windows):
            # Generate sample
            sample = self.generate_sample(current_state)
            samples.append(sample)
            
            # Transition to next state
            next_probs = transitions[current_state]
            current_state = np.random.choice(
                list(next_probs.keys()),
                p=list(next_probs.values())
            )
        
        return samples
    
    def generate_dataset(self, num_users=5, sessions_per_user=10, windows_per_session=20):
        """Generate complete dataset - optimized for smaller size"""
        print(f"\nGenerating optimized dataset...")
        print(f"  Users: {num_users}")
        print(f"  Sessions per user: {sessions_per_user}")
        print(f"  Windows per session: {windows_per_session}")
        print(f"  Total samples: {num_users * sessions_per_user * windows_per_session}")
        
        all_samples = []
        
        for user_id in range(num_users):
            print(f"\n  Generating data for User {user_id + 1}/{num_users}...")
            
            # Add slight user variation
            self._add_user_variation(variation_factor=0.15)
            
            for session in range(sessions_per_user):
                session_data = self.generate_realistic_session(windows_per_session)
                
                # Add user_id and session_id
                for sample in session_data:
                    sample['user_id'] = user_id
                    sample['session_id'] = f"u{user_id}_s{session}"
                
                all_samples.extend(session_data)
                
                if (session + 1) % 5 == 0:
                    print(f"    Session {session + 1}/{sessions_per_user} complete")
        
        print(f"\n✓ Generated {len(all_samples)} total samples")
        return all_samples
    
    def _add_user_variation(self, variation_factor=0.15):
        """Add individual user patterns"""
        for state in self.profiles:
            for feature in self.profiles[state]:
                mean, max_val, std = self.profiles[state][feature]
                variation = np.random.uniform(1 - variation_factor, 1 + variation_factor)
                new_mean = mean * variation
                self.profiles[state][feature] = (new_mean, max_val, std)
    
    def save_dataset(self, data, filename_base="work_monitoring_data"):
        """Save dataset - optimized for size"""
        print(f"\nSaving dataset to disk...")

        # Save as JSON (compressed)
        json_file = f"{filename_base}.json"
        print(f"  Writing {json_file}...")
        with open(json_file, 'w') as f:
            json.dump(data, f)  # No indent to save space

        # Save as CSV
        csv_file = f"{filename_base}.csv"
        print(f"  Writing {csv_file}...")
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        # Get file sizes
        import os
        json_size = os.path.getsize(json_file) / (1024 * 1024)  # MB
        csv_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB

        print(f"\n✓ Dataset saved successfully:")
        print(f"  JSON: {json_file} ({json_size:.2f} MB)")
        print(f"  CSV: {csv_file} ({csv_size:.2f} MB)")

        # Print statistics
        self._print_dataset_stats(df)

        return json_file, csv_file

    def _print_dataset_stats(self, df):
        """Print dataset statistics"""
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Features: {len(df.columns) - 3}")  # -3 for state, user_id, session_id
        print(f"  Users: {df['user_id'].nunique()}")
        print(f"  Sessions: {df['session_id'].nunique()}")

        print(f"\n  State Distribution:")
        state_counts = df['state'].value_counts()
        for state, count in state_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    {state}: {count} ({percentage:.1f}%)")

        # Feature statistics
        print(f"\n  Sample Feature Ranges:")
        key_features = ['typing_speed_wpm', 'click_rate', 'app_switch_rate']
        for feat in key_features:
            if feat in df.columns:
                print(f"    {feat}: {df[feat].min():.2f} - {df[feat].max():.2f}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("OPTIMIZED SYNTHETIC DATASET GENERATOR")
    print("="*60)

    generator = OptimizedSyntheticGenerator(seed=42)

    # Print state definitions
    print("\nState Definitions:")
    print("-" * 60)
    for state, info in generator.state_definitions.items():
        print(f"\n{state.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Indicators: {info['indicators']}")

    print("\n" + "="*60)

    # Generate dataset (larger size: ~20,000 samples = ~8-10MB)
    dataset = generator.generate_dataset(
        num_users=20,          # 20 users (good variety)
        sessions_per_user=50,  # 50 sessions each
        windows_per_session=20 # 20 3-minute windows per session
    )
    # Total: 20 * 50 * 20 = 20,000 samples

    # Save dataset
    print("\nSaving dataset...")
    json_file, csv_file = generator.save_dataset(dataset, filename_base="work_monitoring_data")

    print(f"\n✓ Files created successfully!")
    print(f"  {json_file}")
    print(f"  {csv_file}")

    print("\n" + "="*60)
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"\nYou can now train the model using:")
    print(f"  python training_pipeline.py")