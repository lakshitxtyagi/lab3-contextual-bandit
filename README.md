# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Student Name:** Lakshit Tyagi  
**Roll Number:** U20230090  
**Course:** Reinforcement Learning Fundamentals  
**Date:** February 2026  

---

# 1. Introduction

This project implements a Contextual Multi-Armed Bandit (CMAB) framework for personalized news recommendation. Unlike standard multi-armed bandits, the contextual bandit setting incorporates user-specific side information before selecting an action.

Here:
- **Contexts** represent user categories.
- **Arms** represent news categories.
- **Reward** represents user engagement returned by the `rlcmab-sampler`.

The objective is to learn a policy that maximizes cumulative reward over a time horizon of **T = 10,000** steps.

---

# 2. Problem Formulation

- **Contexts (3):** user_1, user_2, user_3  
- **News Categories (4 per context):** Entertainment, Education, Tech, Crime  
- **Total Arms:** 12 (3 × 4 combinations)  

Each context maintains an independent bandit over its 4 associated arms.

The system pipeline is:

1. Classify user → determine context  
2. Select optimal arm using contextual bandit  
3. Sample article from selected category  
4. Receive reward from environment  

---

# 3. Data Preprocessing

## Dataset Summary

- Labeled Training Users: **2000 samples, 33 columns**
- Unlabeled Test Users: **2000 samples**
- News Articles: **209,527 articles**
- Effective features used: **31**

## Preprocessing Steps

1. Missing value handling  
   - Numerical: Median imputation  
   - Categorical: Mode imputation  

2. Feature transformations  
   - Boolean `subscriber` → binary  
   - Categorical encoding for `browser_version`, `region_code`  

3. Standardization  
   - Applied `StandardScaler` to ensure comparable feature magnitudes  

4. Train-validation split  
   - 80% training (1600 samples)  
   - 20% validation (400 samples)  

All features were converted into numerical format before training.

---

# 4. User Classification

The classifier acts as the **context detector** for the bandit system.

## Model Comparison

| Model | Validation Accuracy |
|-------|--------------------|
| Decision Tree | 58.75% |
| Logistic Regression | 86.25% |
| Random Forest | **90.75%** |

## Best Model: Random Forest

The Random Forest achieved **90.75% accuracy**, significantly outperforming the Decision Tree and Logistic Regression.

### Interpretation

- Logistic Regression captured linear decision boundaries effectively.
- Random Forest further improved performance by modeling nonlinear feature interactions.
- The relatively low Decision Tree accuracy indicates that single-tree variance is high.

The high validation accuracy ensures reliable context identification before bandit decision-making.

---

# 5. Contextual Bandit Algorithms

All simulations were run for **T = 10,000 steps**.

Each context maintained a separate bandit over 4 arms.

---

## 5.1 Epsilon-Greedy

Exploration probability: ε  
Exploitation probability: (1 − ε)

| ε | Average Reward | Total Reward |
|----|----------------|--------------|
| 0.01 | **5.6105** | 56104.58 |
| 0.10 | 5.0616 | 50615.59 |
| 0.30 | 3.9294 | 39293.90 |

### Observations

- ε = 0.01 performed best.
- Larger ε values caused excessive exploration.
- Performance degraded significantly at ε = 0.30.
- The algorithm is highly sensitive to exploration probability.

This confirms that in structured environments with distinguishable optimal arms, aggressive exploitation performs better.

---

## 5.2 Upper Confidence Bound (UCB)

Arm selection rule:

\[
\mu_a + c \sqrt{\frac{\ln t}{N_a}}
\]

| c | Average Reward | Total Reward |
|---|----------------|--------------|
| 0.5 | **5.6688** | 56688.27 |
| 1.0 | 5.6573 | 56572.75 |
| 2.0 | 5.6546 | 56545.57 |

### Observations

- UCB achieved the highest overall reward.
- Performance was stable across c values.
- Lower c slightly favored exploitation and produced marginally better results.
- Confidence-bound driven exploration reduces need for manual tuning.

UCB demonstrated strong theoretical robustness in this contextual setting.

---

## 5.3 SoftMax (Boltzmann Exploration)

Temperature parameter: τ = 1.0

- Average Reward: 5.5249  
- Total Reward: 55248.68  

### Observations

- Probabilistic exploration smooths transitions between arms.
- Performance competitive but slightly below UCB.
- Less sensitive to early noise compared to ε-greedy.

---

# 6. Comparative Analysis

## Overall Ranking

| Algorithm | Best Configuration | Average Reward |
|-----------|-------------------|----------------|
| UCB | c = 0.5 | **5.6688** |
| Epsilon-Greedy | ε = 0.01 | 5.6105 |
| SoftMax | τ = 1.0 | 5.5249 |

### Best Performing Algorithm: UCB

UCB achieved the highest cumulative and average reward.

### Interpretation

- UCB balances exploration automatically through confidence intervals.
- Epsilon-greedy depends heavily on manually chosen ε.
- SoftMax provides stable stochastic exploration but does not outperform UCB in this environment.

---

# 7. Per-Context Reward Analysis

### UCB (Best Performer)

- user_1: 7.7065  
- user_2: 4.4656  
- user_3: 4.8436  

### Insights

- user_1 consistently yields higher reward.
- Reward distributions differ significantly across contexts.
- Context separation improves learning efficiency (4 arms per context vs 12 global arms).

The contextual structure meaningfully reduces regret compared to non-contextual approaches.

---

# 8. Recommendation Engine

## Final System Configuration

- User Classifier: RandomForestClassifier (90.75%)
- Bandit Strategy: UCB (c = 0.5)
- Simulation Horizon: 10,000

## End-to-End Workflow

1. Preprocess incoming user
2. Predict user category
3. Select optimal news category via contextual bandit
4. Randomly sample article from chosen category
5. Receive reward

System validation confirmed coherent category recommendations.

---

# 9. Key Insights & Learning Outcomes

1. Contextual bandits significantly outperform naive global bandits in structured environments.
2. UCB provides robust performance with minimal hyperparameter sensitivity.
3. Exploration–exploitation trade-off is highly sensitive in ε-greedy methods.
4. Proper context classification is critical to downstream reward optimization.
5. All algorithms converged within 10,000 steps, indicating stable learning dynamics.

---

# 10. Conclusion

This project demonstrates the effectiveness of Contextual Multi-Armed Bandits for personalized recommendation systems.

Among the three strategies:

- **UCB emerged as the most stable and highest-performing algorithm.**
- Epsilon-greedy required careful tuning.
- SoftMax provided competitive but slightly lower performance.

The integration of a high-accuracy user classifier with contextual bandits enabled efficient and adaptive recommendation behavior.

---

# 11. Repository Structure

```
.
├── lab3_results_U20230090.ipynb
├── README.md
└── data/
    ├── train_users.csv
    ├── test_users.csv
    └── news_articles.csv
```

---

# 12. How to Run

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn rlcmab-sampler jupyter
```

Run:

```
jupyter notebook lab3_results_U20230090.ipynb
```

Execute all cells to reproduce results.

---

## Author

Lakshit Tyagi  
Roll Number: U20230090  
February 2026
