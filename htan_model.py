"""
Updated Hybrid Temporal Attention Network (HTAN)
For 5 new states with 24 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAttentionLayer(nn.Module):
    """Attention mechanism for temporal sequences"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        return context, attn_weights


class ImprovedHTAN(nn.Module):
    """Improved HTAN for 5 states with variable input features"""

    def __init__(self, input_dim=27, hidden_dim=128, num_classes=5, seq_len=5):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        print(f"Initializing ImprovedHTAN with input_dim={input_dim}")

        # Feature embedding (removed batch norm to avoid reshape issues)
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Multi-head attention layers
        self.attention1 = TemporalAttentionLayer(hidden_dim, hidden_dim)
        self.attention2 = TemporalAttentionLayer(hidden_dim, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # User adaptation layer for online learning
        self.adaptation_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape

        # Embed each time step independently
        embedded_list = []
        for t in range(seq_len):
            emb = self.feature_embedding(x[:, t, :])
            embedded_list.append(emb.unsqueeze(1))
        embedded = torch.cat(embedded_list, dim=1)

        # Apply attention layers
        attn_out1, attn_weights1 = self.attention1(embedded)
        attn_out2, attn_weights2 = self.attention2(attn_out1)

        # LSTM processing
        lstm_out, _ = self.lstm(attn_out2)

        # Take last time step
        final_hidden = lstm_out[:, -1, :]

        # Adaptive transformation
        adapted = self.adaptation_layer(final_hidden)

        # Classification
        logits = self.classifier(adapted)

        if return_attention:
            return logits, (attn_weights1, attn_weights2)
        return logits


class OnlineLearner:
    """Enhanced online learning with experience replay"""

    def __init__(self, model, learning_rate=0.0005, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()

        # Experience replay buffer
        self.memory_buffer = []
        self.max_buffer_size = 500

        # Performance tracking
        self.update_history = []

    def update(self, x, y, epochs=2):
        """Perform online update with new data"""
        self.model.train()

        # Convert to tensors
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor([y]).to(self.device) if isinstance(y, int) else torch.LongTensor(y).to(self.device)

        # Add to memory buffer
        self.memory_buffer.append((x.cpu(), y.cpu()))
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer.pop(0)

        # Train on current sample
        total_loss = 0
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

            # Experience replay
            if len(self.memory_buffer) > 10:
                replay_indices = np.random.choice(
                    len(self.memory_buffer),
                    min(3, len(self.memory_buffer)),
                    replace=False
                )
                for idx in replay_indices:
                    rx, ry = self.memory_buffer[idx]
                    rx, ry = rx.to(self.device), ry.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(rx)
                    loss = self.criterion(outputs, ry)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

        avg_loss = total_loss / epochs
        self.update_history.append(avg_loss)

        return avg_loss

    def predict(self, x):
        """Make prediction with confidence scores"""
        self.model.eval()

        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        return predicted.item(), probs.cpu().numpy()[0], confidence.item()

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_buffer': self.memory_buffer,
            'update_history': self.update_history
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory_buffer = checkpoint.get('memory_buffer', [])
        self.update_history = checkpoint.get('update_history', [])


def prepare_data_for_training(data, seq_len=5):
    """Prepare sequential data from dataset"""

    # State mapping
    label_map = {
        'focused_work': 0,
        'task_execution': 1,
        'research_reading': 2,
        'meeting_communication': 3,
        'unproductive': 4
    }

    # Feature order (27 features - ALL features from synthetic data)
    feature_names = [
        # Keyboard features (7)
        'keystroke_count', 'keystroke_rate', 'typing_speed_wpm',
        'keystroke_variance', 'keystroke_burst_ratio',
        'avg_inter_keystroke_time', 'keystroke_rhythm_std',

        # Mouse features (7)
        'mouse_movement_count', 'mouse_distance', 'mouse_velocity',
        'mouse_velocity_std', 'mouse_acceleration',
        'mouse_direction_changes', 'mouse_idle_ratio',

        # Click features (5)
        'click_count', 'click_rate', 'click_clustering',
        'avg_click_interval', 'click_pattern_regularity',

        # Scroll features (4)
        'scroll_count', 'scroll_rate', 'scroll_intensity',
        'scroll_direction_changes',

        # App features (4)
        'app_switch_count', 'app_switch_rate',
        'unique_apps_count', 'app_focus_stability'
    ]

    print(f"Using {len(feature_names)} features for training")

    sequences = []
    labels = []

    # Group by session
    sessions = {}
    for sample in data:
        session_id = sample.get('session_id', f"u{sample.get('user_id', 0)}_s0")
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(sample)

    print(f"Processing {len(sessions)} sessions...")

    # Create sequences from each session
    for session_id, samples in sessions.items():
        session_features = []
        session_labels = []

        for sample in samples:
            # Extract features in correct order
            features = [sample.get(feat, 0.0) for feat in feature_names]
            session_features.append(features)
            session_labels.append(label_map[sample['state']])

        # Create overlapping sequences
        for i in range(len(session_features) - seq_len + 1):
            seq = session_features[i:i + seq_len]
            label = session_labels[i + seq_len - 1]  # Label from last window
            sequences.append(seq)
            labels.append(label)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"Created {len(sequences)} sequences of length {seq_len}")
    print(f"Feature dimension: {sequences.shape[2]}")

    return sequences, labels


# Example usage
if __name__ == "__main__":
    print("Testing Improved HTAN Model...")

    # Create model
    model = ImprovedHTAN(
        input_dim=24,
        hidden_dim=128,
        num_classes=5,
        seq_len=5
    )

    print(f"\nModel Architecture:")
    print(f"  Input: 24 features")
    print(f"  Hidden: 128 dimensions")
    print(f"  Output: 5 classes")
    print(f"  Sequence Length: 5 windows (15 minutes)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 5
    input_dim = 24

    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(x)

    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {outputs.shape}")

    # Test online learner
    learner = OnlineLearner(model, learning_rate=0.0005)

    test_x = torch.randn(seq_len, input_dim)
    test_y = 0

    loss = learner.update(test_x, test_y)
    print(f"\nOnline Learning Test:")
    print(f"  Update loss: {loss:.4f}")

    predicted, probs, confidence = learner.predict(test_x)
    print(f"  Prediction: Class {predicted}")
    print(f"  Confidence: {confidence:.4f}")

    print("\n✓ Model test complete!")