# Reinforcement Learning for Video Anomaly Detection

## Overview
This project investigates the use of reinforcement learning for video anomaly detection using the UCSD Ped2 dataset.

Instead of treating anomaly detection as a supervised classification problem, this work models it as a sequential decision-making task. The agent is trained only on normal data and learns to detect anomalies as deviations from learned behavior.

---

## Problem Setting
- Input: video sequences (UCSD Ped2)
- Training data: only normal frames
- Testing data: normal + anomalous frames
- Goal: detect abnormal events without explicit anomaly supervision

---

## Approach

Pipeline:

Video Frames → Hiera-L Features → Temporal State → DQN → Anomaly Score

---

## Methodology

### Feature Extraction
- Hiera-L is used to extract spatio-temporal representations from video
- Each feature encodes a temporal window of frames

### State Representation
Each state incorporates temporal context:

state = [previous frame, current frame, next frame]

This allows the model to capture motion and temporal dependencies.

---

### Reinforcement Learning Formulation

- Actions:
  - 0 → normal
  - 1 → anomaly

- Training setup:
  - Only normal data is used
  - The agent is rewarded for predicting normal behavior

- Reward design:
  +1   → normal behavior  
  -0.5 → deviation from normal  

The agent learns a policy that models normal patterns over time.

---

### Anomaly Detection

- The model outputs Q-values for each state
- Anomaly score is computed using Q-value behavior
- Lower confidence or weaker separation indicates anomaly

---

## Evaluation

- Dataset: UCSD Ped2
- Training: normal data only
- Testing: mixed (normal + anomaly)

Metrics:
- ROC-AUC (primary)
- Threshold-based accuracy (secondary)

Results:
AUC ≈ 0.73  
Accuracy ≈ 0.62  

---

## Key Insight

This project demonstrates that reinforcement learning can be used to learn normal behavior in video sequences, and anomalies can be detected as deviations from learned decision patterns.

---

## Limitations

- Reinforcement learning is not the standard approach for video anomaly detection
- Performance depends on reward design
- DQN without replay buffer or target network is limited
- Learned representations depend on feature quality

---

## Future Work

- Improve reward design
- Add experience replay and target networks
- Explore hybrid approaches combining RL and reconstruction methods
- Evaluate on additional datasets
- Compare against standard VAD baselines

---

## Project Structure

src/
- data_loader.py
- feature_extraction.py
- dqn_agent.py
- train-test.py

notebooks/

README.md  
requirements.txt  

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Run training and evaluation:

python src/train-test.py

---

## Author
Omeshamisu Anigala