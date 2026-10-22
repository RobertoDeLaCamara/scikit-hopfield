"""BADSFA Behavioral & Mathematical Contract Test Suite for scikit-hopfield."""

from __future__ import annotations

import unittest

import numpy as np

from scikit_hopfield import (
    HopfieldBAM,
    HopfieldMetastableCluster,
    HopfieldOOD,
    hopfield_energy,
    hopfield_update_step,
)


class TestBOLT001HopfieldOOD(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=[1.0, 0.0, 0.0] + [0.0] * 13, scale=0.05, size=(30, 16))
        c2 = rng.normal(loc=[0.0, 1.0, 0.0] + [0.0] * 13, scale=0.05, size=(30, 16))
        c3 = rng.normal(loc=[0.0, 0.0, 1.0] + [0.0] * 13, scale=0.05, size=(30, 16))
        self.X_train = np.vstack([c1, c2, c3])
        self.y_train = np.array([0] * 30 + [1] * 30 + [2] * 30)
        self.X_ood = rng.normal(loc=[0.0] * 10 + [1.0] * 6, scale=0.05, size=(15, 16))

    def test_high_beta_asymptotic_1nn_limit(self) -> None:
        """BOLT-001 Scenario 1: As beta -> infinity, score converges to -max_i cos(x_i, q)."""
        det = HopfieldOOD(beta=250.0, mode="full", normalize=True).fit(self.X_train)
        scores = det.decision_function(self.X_ood)

        X_norm = self.X_train / np.linalg.norm(self.X_train, axis=1, keepdims=True)
        Q_norm = self.X_ood / np.linalg.norm(self.X_ood, axis=1, keepdims=True)
        exact_neg_1nn = -np.max(Q_norm @ X_norm.T, axis=1)

        np.testing.assert_allclose(scores, exact_neg_1nn, atol=0.025)

    def test_she_mode_compresses_to_class_centroids_and_separates_ood(self) -> None:
        """BOLT-001 Scenario 2: SHE stores K class centroids and separates ID from OOD."""
        det = HopfieldOOD(beta=8.0, mode="she", contamination=0.1).fit(self.X_train, self.y_train)
        self.assertEqual(det.memories_.shape, (3, 16))

        id_scores = det.decision_function(self.X_train)
        ood_scores = det.decision_function(self.X_ood)
        self.assertGreater(float(np.min(ood_scores)), float(np.max(id_scores)))

        preds_ood = det.predict(self.X_ood)
        self.assertTrue(np.all(preds_ood == -1))

    def test_reclag_mode_collapses_low_similarity_ood_to_origin_attractor(self) -> None:
        """BOLT-001 Scenario 3: Rectified Lagrangian zeroes out similarities below tau_rect."""
        det = HopfieldOOD(beta=10.0, mode="reclag", tau_rect=0.6).fit(self.X_train)
        id_scores = det.decision_function(self.X_train[:10])
        ood_scores = det.decision_function(self.X_ood)

        np.testing.assert_allclose(ood_scores, np.zeros_like(ood_scores), atol=1e-7)
        self.assertTrue(np.all(id_scores < -0.25))

    def test_hopfield_boosting_with_auxiliary_outliers(self) -> None:
        """BOLT-001 Scenario 4: Hopfield Boosting increases separation when X_outliers is passed."""
        det_base = HopfieldOOD(beta=8.0, mode="full").fit(self.X_train)
        det_boost = HopfieldOOD(beta=8.0, mode="boosting").fit(
            self.X_train, X_outliers=self.X_ood[:5]
        )
        mean_ood_base = float(np.mean(det_base.decision_function(self.X_ood[5:])))
        mean_id_base = float(np.mean(det_base.decision_function(self.X_train)))
        gap_base = mean_ood_base - mean_id_base

        mean_ood_boost = float(np.mean(det_boost.decision_function(self.X_ood[5:])))
        mean_id_boost = float(np.mean(det_boost.decision_function(self.X_train)))
        gap_boost = mean_ood_boost - mean_id_boost
        self.assertGreater(gap_boost, gap_base)

    def test_explain_returns_normalized_attention_attribution(self) -> None:
        """BOLT-001 Scenario 5: Explainability returns top_k indices and weights."""
        det = HopfieldOOD(beta=12.0, mode="full").fit(self.X_train)
        explanation = det.explain(self.X_train[:4], top_k=3)
        self.assertEqual(explanation["top_indices"].shape, (4, 3))
        self.assertEqual(explanation["top_weights"].shape, (4, 3))
        self.assertTrue(np.all(explanation["top_indices"][:, 0] < 30))


class TestBOLT002MetastableClustering(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(loc=[1.0, 0.0, 0.0, 0.0], scale=0.04, size=(15, 4))
        b = rng.normal(loc=[0.0, 1.0, 0.0, 0.0], scale=0.04, size=(15, 4))
        c = rng.normal(loc=[0.0, 0.0, 1.0, 0.0], scale=0.04, size=(15, 4))
        self.X = np.vstack([a, b, c])

    def test_lyapunov_monotonic_energy_descent(self) -> None:
        """BOLT-002 Scenario 1: Energy E_beta(xi_t) decreases monotonically at every update step."""
        X_norm = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
        xi = np.array([0.5, 0.4, 0.2, 0.1], dtype=float)
        xi = xi / np.linalg.norm(xi)
        beta = 6.0

        prev_e = float(hopfield_energy(X_norm, xi[None, :], beta=beta)[0])
        for _ in range(15):
            xi = hopfield_update_step(X_norm, xi[None, :], beta=beta)[0]
            curr_e = float(hopfield_energy(X_norm, xi[None, :], beta=beta)[0])
            self.assertLessEqual(curr_e, prev_e + 1e-6)
            prev_e = curr_e

    def test_beta_bifurcation_from_global_average_to_three_clusters(self) -> None:
        """BOLT-002 Scenario 2: Low beta yields 1 global state; intermediate beta yields 3."""
        c_global = HopfieldMetastableCluster(beta=0.01, merge_tol=1e-2).fit(self.X)
        self.assertEqual(c_global.n_clusters_, 1)

        c_three = HopfieldMetastableCluster(beta=10.0, merge_tol=1e-2).fit(self.X)
        self.assertEqual(c_three.n_clusters_, 3)
        self.assertEqual(len(np.unique(c_three.labels_)), 3)


class TestBOLT003HeteroAssociativeBAM(unittest.TestCase):
    def setUp(self) -> None:
        self.X_notes = np.eye(4, 8, dtype=float)
        self.Y_books = np.eye(4, 8, k=2, dtype=float)

    def test_cross_domain_resonance_retrieves_paired_concept(self) -> None:
        """BOLT-003 Scenario 1: Noisy query in Domain A resonates with paired concept in B."""
        bam = HopfieldBAM(beta=15.0, max_iter=10).fit(self.X_notes, self.Y_books)
        noisy_note_2 = self.X_notes[2] + np.array([0.05, -0.04, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
        res = bam.resonate(noisy_note_2[None, :], top_k=2)

        self.assertEqual(int(res["top_b_indices"][0, 0]), 2)
        self.assertTrue(bool(res["converged"][0]))

    def test_knowledge_gap_detection_flags_unindexed_topic(self) -> None:
        """BOLT-003 Scenario 2: Orthogonal unindexed topic is flagged as a knowledge gap."""
        bam = HopfieldBAM(beta=15.0, gap_quantile=0.99).fit(self.X_notes, self.Y_books)
        known_query = self.X_notes[0:1]
        unknown_query = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        gap_known = bam.detect_knowledge_gap(known_query)
        gap_unknown = bam.detect_knowledge_gap(unknown_query)

        self.assertFalse(bool(gap_known["is_gap"][0]))
        self.assertTrue(bool(gap_unknown["is_gap"][0]))


if __name__ == "__main__":
    unittest.main()
