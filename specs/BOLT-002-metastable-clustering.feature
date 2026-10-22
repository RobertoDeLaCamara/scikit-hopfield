Feature: Metastable Bifurcation Clustering via Continuous Hopfield Dynamics (BOLT-002)
  As a Knowledge Engineer and Data Scientist
  I want a scikit-learn ClusterMixin estimator that evolves queries along the Modern Hopfield update rule
  So that correlated embeddings naturally coalesce into metastable attractors across inverse temperature regimes.

  Scenario: Strict Lyapunov monotonic energy descent during state evolution
    Given a set of stored patterns X and an arbitrary initial query state xi_0
    When the continuous Hopfield update xi_{t+1} = X^T softmax(beta * X * xi_t) is applied iteratively
    Then the Hopfield energy E_beta(xi_t) must decrease or stay constant at every step (Delta E <= 1e-6)

  Scenario: Inverse temperature beta bifurcation from global mean to distinct clusters
    Given three well-separated semantic clusters in embedding space
    When HopfieldMetastableCluster is run with beta -> 0.01 (high temperature)
    Then all points must coalesce into a single global metastable state (n_clusters == 1)
    And when run with intermediate beta = 8.0, points must separate into the 3 true metastable cluster attractors
