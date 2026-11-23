# Changelog

All notable changes to `scikit-hopfield` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- `HopfieldOOD` (`OutlierMixin`, `BaseEstimator`): Modern Continuous Hopfield OOD/novelty detector supporting `full`, `she`, `kmeans` (`k-means++` with `random_state`), and `reclag` (Rectified Lagrangian, AAAI 2025), plus optional contrastive outlier exposure (`X_outliers`) and `explain()` attention attribution.
- `HopfieldOOD.energy_score(X)` exposing raw anomaly energy alongside standard `scikit-learn` `score_samples(X)`, `decision_function(X)`, `predict(X)`, and `fit_predict(X)`.
- `HopfieldMetastableCluster` (`ClusterMixin`, `BaseEstimator`): iterative CCCP metastable clustering with order-independent hierarchical attractor merging (`AgglomerativeClustering`) and `bifurcation_spectrum()`.
- `HopfieldBAM` (`TransformerMixin`, `BaseEstimator`): continuous bidirectional cross-domain associative memory with `resonate()`, `transform()`, `detect_knowledge_gap()`, and explicit `gap_margin` parameter.
- `sklearn.utils.estimator_checks.check_estimator` compatibility test suite and leave-one-out threshold calibration regression tests.
- Core energy primitives (`hopfield_energy`, `hopfield_update_step`, `attention_weights`, `logsumexp_scaled`, `reclag_score`, `reclag_score_from_sim`, `l2_normalize`).

### Fixed
- Eliminated self-similarity threshold bias in `HopfieldOOD` (`full` and `reclag`) via exact leave-one-out calibration in `fit()`.
- Removed the no-op `mode="boosting"` alias and raised explicit `ValueError` when `X_outliers` is passed with `mode="reclag"`.
- Replaced the hardcoded, memory-heavy `O(n*k*d)` KMeans loop in `HopfieldOOD(mode="kmeans")` with `sklearn.cluster.KMeans` (`k-means++`, configurable `random_state`).
- Replaced the order-dependent greedy attractor merge loop in `HopfieldMetastableCluster` with order-independent hierarchical clustering.
