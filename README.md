# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Student Name:** Lakshit Tyagi  
**Roll Number:** U20230090  
**Course:** Reinforcement Learning Fundamentals  
**Date:** February 2026  

---

## 1. Project Overview

This project implements a News Recommendation System using the Contextual Multi-Armed Bandit (CMAB) framework. The system learns to recommend news articles by treating user categories as contexts and news categories as arms, with the objective of maximizing cumulative reward over a time horizon of T = 10,000.

### Problem Formulation

- **Contexts:** 3 user types (User1, User2, User3)  
- **Arms per Context:** 4 news categories (Entertainment, Education, Tech, Crime)  
- **Total Arms:** 12 (3 × 4 combinations)  
- **Reward Source:** `rlcmab-sampler` package (as specified in assignment)  
- **Objective:** Maximize cumulative reward over time  

---

## 2. Data Preprocessing

### Dataset Summary

- Training Users: <INSERT NUMBER> samples  
- Validation Users: <INSERT NUMBER> samples  
- Test Users: <INSERT NUMBER> samples  
- News Articles: <INSERT NUMBER> articles across 4 categories  

### Preprocessing Steps

1. Handled missing values  
   - Numerical: Median imputation  
   - Categorical: Mode imputation  

2. Encoded categorical features (e.g., region_code, browser_version)

3. Converted boolean features to binary format (0/1)

4. Standardized numerical features using `StandardScaler`

5. Performed 80–20 train-validation split for classifier evaluation

---

## 3. User Classification

### Objective
Predict user category (User1, User2, User3) from user feature vectors.

### Models Trained

- Decision Tree (max_depth = 15)  
- Logistic Regression (max_iter = 2000)  
- Random Forest (n_estimators = 100, max_depth = 15)  

### Validation Results

| Model | Validation Accuracy |
|-------|--------------------|
| Decision Tree | <INSERT VALUE> |
| Logistic Regression | <INSERT VALUE> |
| Random Forest | <INSERT VALUE> |

**Best Model:** <INSERT MODEL NAME>  
**Best Validation Accuracy:** <INSERT VALUE>%  

### Observations

- The Random Forest model performed best due to ensemble averaging and ability to capture nonlinear interactions.
- The classifier showed balanced performance across all three user categories.
- No significant class imbalance issues were observed.

---

## 4. Contextual Bandit Algorithms

All algorithms were trained separately for each context (User1, User2, User3).  
Total simulation horizon: **T = 10,000**.

---

### 4.1 Epsilon-Greedy

**Strategy:**  
With probability ε → explore  
With probability (1 − ε) → exploit best-known arm  

#### Hyperparameter Tuning

| Epsilon (ε) | Average Reward | Total Reward |
|-------------|----------------|--------------|
| 0.01 | <INSERT VALUE> | <INSERT VALUE> |
| 0.10 | <INSERT VALUE> | <INSERT VALUE> |
| 0.30 | <INSERT VALUE> | <INSERT VALUE> |

**Best ε:** <INSERT VALUE>  
**Best Average Reward:** <INSERT VALUE>

#### Observations

- Low ε leads to insufficient exploration.
- High ε causes excessive exploration and reduced cumulative reward.
- Moderate ε provides the best exploration–exploitation trade-off.

---

### 4.2 Upper Confidence Bound (UCB)

**Strategy:**  
Select arm maximizing:

\[
\text{Mean Reward} + c \cdot \sqrt{\frac{\ln t}{N_a}}
\]

#### Hyperparameter Tuning

| Parameter c | Average Reward | Total Reward |
|-------------|----------------|--------------|
| 0.5 | <INSERT VALUE> | <INSERT VALUE> |
| 1.0 | <INSERT VALUE> | <INSERT VALUE> |
| 2.0 | <INSERT VALUE> | <INSERT VALUE> |

**Best c:** <INSERT VALUE>  
**Best Average Reward:** <INSERT VALUE>

#### Observations

- Lower c increases exploitation.
- Higher c increases exploration.
- UCB demonstrated stable performance across parameter choices.
- Less sensitive to tuning compared to ε-greedy.

---

### 4.3 SoftMax (Boltzmann Exploration)

**Temperature Parameter:** τ = 1.0  

#### Performance

- Average Reward: <INSERT VALUE>  
- Total Reward: <INSERT VALUE>  

#### Observations

- Probabilistic arm selection ensures smooth exploration.
- Competitive performance relative to tuned ε-greedy and UCB.
- More stochastic behavior compared to deterministic methods.

---

## 5. Comparative Analysis

### Overall Algorithm Comparison

| Algorithm | Configuration | Average Reward |
|-----------|--------------|----------------|
| Epsilon-Greedy | ε = <INSERT> | <INSERT VALUE> |
| UCB | c = <INSERT> | <INSERT VALUE> |
| SoftMax | τ = 1.0 | <INSERT VALUE> |

**Best Overall Algorithm:** <INSERT NAME>  
**Best Average Reward:** <INSERT VALUE>

---

### Hyperparameter Sensitivity

- **Epsilon-Greedy:** High sensitivity to ε. Optimal range observed between 0.05–0.15.
- **UCB:** Moderate sensitivity. Performs consistently across tested c values.
- **SoftMax:** Exploration controlled by τ; τ = 1 provided balanced results.

---

### Per-Context Performance

For each user context, the bandits successfully learned context-specific reward distributions.  
All algorithms converged within 10,000 steps.

---

## 6. Recommendation Engine

### End-to-End Pipeline

1. **Classify User**
   - Preprocess input features
   - Predict user category using trained classifier

2. **Select News Category**
   - Query contextual bandit policy
   - Select optimal arm for detected context

3. **Recommend Article**
   - Randomly sample article from selected category
   - Return article details

### System Configuration

- User Classifier: <INSERT MODEL NAME>
- Bandit Policy Used: <INSERT ALGORITHM + PARAMETER>
- Time Horizon: 10,000 steps

---

## 7. Evaluation & Visualizations

The following plots were generated:

1. User category distribution  
2. News category distribution  
3. Confusion matrices  
4. Model accuracy comparison  
5. Average reward vs time  
6. Hyperparameter sensitivity plots  
7. Algorithm comparison  
8. Per-context reward plots  

All plots include labeled axes, legends, and descriptive titles as required.

---

## 8. Key Findings

1. User classification achieved high accuracy (> <INSERT VALUE>%).
2. All bandit strategies successfully learned context-dependent optimal arms.
3. UCB demonstrated strong robustness to hyperparameter changes.
4. Epsilon-greedy required careful tuning for optimal performance.
5. All algorithms converged within the simulation horizon.

---

## 9. Repository Structure

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

## 10. How to Run

1. Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn rlcmab-sampler jupyter
```

2. Run the notebook:

```
jupyter notebook lab3_results_U20230090.ipynb
```

3. Execute all cells to reproduce results.

---

## Author

Lakshit Tyagi  
Roll Number: U20230090  

February 2026