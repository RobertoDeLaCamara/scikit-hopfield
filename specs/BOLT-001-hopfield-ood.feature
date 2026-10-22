Feature: Modern Hopfield Energy Out-of-Distribution Detector (BOLT-001)
  As an ML / SecOps Engineer
  I want a scikit-learn/PyOD compatible HopfieldOOD estimator supporting the continuous beta-dial, SHE, Hopfield Boosting, and Rectified Lagrangian (RecLag)
  So that I can detect novel OOD patterns in embeddings and attribute each alert to its closest stored memories.

  Scenario: High-beta asymptotic convergence to 1-Nearest Neighbor cosine similarity
    Given an L2-normalized memory matrix X of in-distribution patterns
    When HopfieldOOD is evaluated with beta = 200.0 in "full" mode on query vectors Q
    Then the negated Hopfield energy score must match -max_i cos(x_i, q) within 0.02 absolute tolerance

  Scenario: Simplified Hopfield Energy (SHE) class prototype compression
    Given labeled in-distribution training samples (X, y) across K classes
    When HopfieldOOD is fitted with mode = "she"
    Then the stored memory bank must contain exactly K normalized class prototypes
    And in-distribution queries must score strictly lower anomaly energy than orthogonal OOD queries

  Scenario: Rectified Lagrangian (RecLag) trivial origin attractor for low-density OOD
    Given an in-distribution cluster and distant OOD samples whose max cosine similarity is below tau_rect
    When HopfieldOOD is fitted with mode = "reclag" and tau_rect = 0.5
    Then OOD queries with all similarities below tau_rect must collapse to zero rectified activation and receive maximum anomaly score

  Scenario: Hopfield Boosting with auxiliary outlier exposure
    Given in-distribution patterns X_id and auxiliary outlier patterns X_ood
    When HopfieldOOD is fitted with X_outliers = X_ood
    Then the boosting energy score must penalize proximity to X_ood while rewarding proximity to X_id

  Scenario: Explainable associative memory attribution
    Given a fitted HopfieldOOD detector
    When explain(Q, top_k=3) is called
    Then it must return top-3 memory indices and attention weights summing to 1.0 along the full memory bank
