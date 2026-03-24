import numpy as np
import torch
from dqn_agent import DQNAgent
from sklearn.metrics import roc_auc_score
import random


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# -----------------------------
# 1. Load TRAIN data (normal only)
# -----------------------------
train_features = np.load("/UCSDped2/features_PED2_train.npy")
train_labels   = np.load("/UCSDped2/labels_PED2_train.npy")

print("Train Features:", train_features.shape)
print("Train Labels:", train_labels.shape)

# -----------------------------
# 2. Load TEST data
# -----------------------------
test_features = np.load("/UCSDped2/features_PED2_test.npy")
test_labels   = np.load("/UCSDped2/labels_PED2_test.npy")

print("Test Features:", test_features.shape)
print("Test Labels:", test_labels.shape)
print("Test Anomalies:", test_labels.sum())

# -----------------------------
# 3. RL Agent (DQN)
# -----------------------------
input_dim = 3 * 1152
agent = DQNAgent(input_dim=input_dim, epsilon=0.2)

print("Device:", next(agent.model.parameters()).device)

# -----------------------------
# 4. State function (temporal)
# -----------------------------
def get_state(features, t):
    prev_f = features[t-1] if t > 0 else features[t]
    next_f = features[t+1] if t < len(features)-1 else features[t]
    return np.concatenate([prev_f, features[t], next_f])

# -----------------------------
# 5. TRAINING (normal-only learning)
# -----------------------------
epochs = 10

for epoch in range(epochs):
    total_reward = 0

    for t in range(len(train_features) - 1):
        state = get_state(train_features, t)
        next_state = get_state(train_features, t + 1)

        action = agent.choose_action(state)

        # ?? Unsupervised reward (learn normal)
        if action == 0:
            reward = +1
        else:
            reward = -0.5

        agent.update(state, action, reward, next_state)
        total_reward += reward

    # decay exploration
    agent.epsilon = max(0.01, agent.epsilon * 0.95)

    print(f"Epoch {epoch}: Train Reward = {total_reward}")

# -----------------------------
# 6. TEST: compute anomaly scores
# -----------------------------
scores = []

for t in range(len(test_features)):
    state = get_state(test_features, t)
    score = agent.anomaly_score(state)   # low confidence ? anomaly
    scores.append(score)

scores = np.array(scores)

# -----------------------------
# 7. Normalize scores (important)
# -----------------------------
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# -----------------------------
# 8. Threshold-based prediction
# -----------------------------
# threshold = np.mean(scores)
threshold = np.percentile(scores, 85)
preds = (scores > threshold).astype(int)

accuracy = (preds == test_labels).mean()
print(f"RL Test Accuracy (threshold): {accuracy:.4f}")

# -----------------------------
# 9. AUC (MAIN METRIC)
# -----------------------------
auc = roc_auc_score(test_labels, scores)
print(f"RL AUC: {auc:.4f}")