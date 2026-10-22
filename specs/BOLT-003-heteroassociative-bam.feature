Feature: Continuous Hetero-Associative Bidirectional Associative Memory (BOLT-003)
  As a Personal Knowledge Management (Obsidian + Books) Architect
  I want a continuous HopfieldBAM estimator bridging two distinct embedding domains (Notes <-> Books)
  So that queries reverberate to cross-domain resonance while detecting unindexed knowledge gaps.

  Scenario: Cross-domain reverberation converges to resonant paired attractors
    Given paired or cross-linked concept embeddings between Domain A (Vault Notes) and Domain B (Book Chunks)
    When a noisy query in Domain A is passed to HopfieldBAM.resonate()
    Then the bidirectional trajectory A -> B -> A must converge within max_iter and retrieve the matching Domain B concept as top-1

  Scenario: Knowledge gap detection on unindexed topics
    Given a fitted HopfieldBAM over existing Vault Notes and Book Chunks
    When an orthogonal query on an unindexed topic is evaluated
    Then detect_knowledge_gap() must flag is_gap = True with energy exceeding the calibrated in-distribution threshold
