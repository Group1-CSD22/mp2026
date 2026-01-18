"""
Updated Hybrid Temporal Attention Network (HTAN)
- 30 input features (enhanced with position-based mouse, app categories)
- 6 output classes (improved state definitions)
- Optimized for online learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalAttentionLayer(nn.Module):
    """Multi-head attention for temporal sequences"""
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Multi-head projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.out(context)
        
        return output, attn_weights.mean(dim=1)  # Average across heads


class UpdatedHTAN(nn.Module):
    """Updated HTAN for 6-state classification with 30 features"""
    
    def __init__(self, input_dim=30, hidden_dim=128, num_classes=6, seq_len=5, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        
        print(f"Initializing HTAN: {input_dim} features → {num_classes} states")
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal attention layers (multi-head)
        self.attention1 = TemporalAttentionLayer(hidden_dim, hidden_dim, num_heads=4)
        self.attention2 = TemporalAttentionLayer(hidden_dim, hidden_dim, num_heads=4)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Embed each time step
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
        
        # Classification
        logits = self.classifier(final_hidden)
        
        if return_attention:
            return logits, (attn_weights1, attn_weights2)
        return logits
    
    def extract_features(self, x):
        """Extract learned representations"""
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            embedded_list = []
            for t in range(seq_len):
                emb = self.feature_embedding(x[:, t, :])
                embedded_list.append(emb.unsqueeze(1))
            embedded = torch.cat(embedded_list, dim=1)
            
            attn_out1, _ = self.attention1(embedded)
            attn_out2, _ = self.attention2(attn_out1)
            lstm_out, _ = self.lstm(attn_out2)
            
            features = lstm_out[:, -1, :]
        
        return features


class OnlineLearningOptimizer:
    """Optimizer for online learning with experience replay"""
    
    def __init__(self, model, learning_rate=0.0003, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Experience replay buffer
        self.memory = []
        self.max_memory = 500
        
        # Tracking
        self.update_count = 0
        self.loss_history = []
    
    def add_experience(self, x, y):
        """Add experience to memory"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor([y]) if isinstance(y, int) else torch.LongTensor(y)
        
        self.memory.append((x, y))
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def update(self, batch_size=8, epochs=3):
        """Perform online update with experience replay"""
        if len(self.memory) < batch_size:
            return None
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            # Sample batch from memory
            indices = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), replace=False)
            
            batch_x = []
            batch_y = []
            
            for idx in indices:
                x, y = self.memory[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            # Convert to tensors
            batch_x = torch.stack(batch_x).to(self.device)
            batch_y = torch.cat(batch_y).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / epochs
        self.loss_history.append(avg_loss)
        self.update_count += 1
        
        # Learning rate scheduling
        self.scheduler.step(avg_loss)
        
        self.model.eval()
        return avg_loss
    
    def predict(self, x):
        """Make prediction"""
        self.model.eval()
        
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        
        return predicted.item(), probs.cpu().numpy()[0], confidence.item()


def prepare_data_for_training(data, seq_len=5):
    """Prepare sequential data from dataset"""
    
    # Updated state mapping (6 states)
    label_map = {
        'deep_work': 0,
        'active_work': 1,
        'research': 2,
        'communication': 3,
        'distracted': 4,
        'idle': 5
    }
    
    # 30 features in order
    feature_names = [
        # Keyboard (7)
        'keystroke_count', 'keystroke_rate', 'typing_speed_wpm',
        'keystroke_variance', 'keystroke_burst_ratio', 'avg_inter_keystroke',
        'alpha_ratio',
        
        # Mouse position-based (8)
        'mouse_movement_count', 'total_distance', 'avg_distance_per_move',
        'coverage_score', 'x_variance', 'y_variance', 'direction_changes',
        'is_idle',
        
        # Click (3)
        'click_count', 'click_rate', 'avg_click_interval',
        
        # Scroll (3)
        'scroll_count', 'scroll_rate', 'total_scroll',
        
        # App (6)
        'app_switch_count', 'unique_apps', 'app_productivity',
        'focus_stability', 'productive_ratio', 'digit_ratio',
        
        # Additional (3)
        'special_ratio', 'click_variance', 'avg_scroll'
    ]
    
    print(f"Preparing data with {len(feature_names)} features")
    print(f"States: {list(label_map.keys())}")
    
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
    
    # Create sequences
    for session_id, samples in sessions.items():
        session_features = []
        session_labels = []
        
        for sample in samples:
            features = [sample.get(feat, 0.0) for feat in feature_names]
            session_features.append(features)
            session_labels.append(label_map[sample['state']])
        
        # Create overlapping sequences
        for i in range(len(session_features) - seq_len + 1):
            seq = session_features[i:i + seq_len]
            label = session_labels[i + seq_len - 1]
            sequences.append(seq)
            labels.append(label)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Shape: {sequences.shape}")
    
    return sequences, labels


if __name__ == "__main__":
    # Test model
    model = UpdatedHTAN(
        input_dim=30,
        hidden_dim=128,
        num_classes=6,
        seq_len=5
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 5, 30)
    outputs = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {outputs.shape}")
    
    # Test online learner
    learner = OnlineLearningOptimizer(model)
    learner.add_experience(torch.randn(5, 30), 0)
    learner.add_experience(torch.randn(5, 30), 1)
    
    loss = learner.update()
    print(f"\nOnline update loss: {loss}")
    
    print("\n✓ Model test complete!")
