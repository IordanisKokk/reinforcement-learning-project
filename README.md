# Computational Intelligence - Reinforcement Learning Project
## Evaluating Reinforcement Learning Strategies in Multi-Agent Environments
This project explores the performance of various multi-agent reinforcement learning (MARL) algorithms in the Knights/Archers/Zombies (KAZ) environment from PettingZoo Butterfly environments, a cooperative multi-agent setting where different agent types (Knights and Archers) must work together to defeat zombies. The goal is to compare different RL strategies, evaluate independent vs. coordinated learning, and analyze the impact of observation space selection.

### Objectives & Research Questions:
- **Which RL algorithm works best in KAZ for different agents? (Knights vs Archers)**
- **Does independent learning (IQL, IPPO) work well in cooperative settings?**
- **Do mixed-algorithm teams perform better than homogeneous teams (all agents using the same RL algorithm)?**
- **Does multi-agent coordination (QMIX) outperform independent RL?**

### Methodology
- Implement and train multi-agent RL models using the PettingZoo framework.
- Compare policy-based (PPO, A2C) and value-based (IQL, QMIX) multi-agent RL algorithms.
- Train agents in single-agent vs. multi-agent settings to measure learning efficiency.
- Evaluate homogeneous teams (same RL algorithm for all agents) vs. mixed teams (different algorithms for different roles).