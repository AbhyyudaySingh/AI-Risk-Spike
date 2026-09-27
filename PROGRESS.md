# Progress Tracking: AI Fraud-Spike Detector

## Phase 1 - Data Pipeline & Strict Chronological Split
- [x] Task 1.1 - Raw file ingestion
- [x] Task 1.2 - Integrity and target audit
- [x] Task 1.3 - Identity join
- [x] Task 1.4 - Feature-type audit
- [x] Task 1.5 - Missingness audit
- [x] Task 1.6 - Chronological ordering
- [x] Task 1.7 - Same-timestamp policy
- [x] Task 1.8 - Freeze 70/15/15 outer split
- [x] Phase 1 Validation Gate

## Phase 2 - Leakage-Safe Feature Engineering & Preprocessing
- [ ] Task 2.1 - Feature registry
- [ ] Task 2.2 - Numerical imputation
- [ ] Task 2.3 - Categorical missing values and vocabulary
- [ ] Task 2.4 - Logistic baseline encoding
- [ ] Task 2.5 - Missingness indicators
- [ ] Task 2.6 - Raw transaction feature transformations
- [ ] Task 2.7 - Stateful past-only entity features
- [ ] Task 2.8 - CV-specific preprocessing artifacts
- [ ] Phase 2 Validation Gate

## Phase 3 - Transaction-Risk Model (Layer 1)
- [ ] Task 3.1 - Expanding temporal CV folds
- [ ] Task 3.2 - Simple supervised baseline
- [ ] Task 3.3 - XGBoost temporal-CV training
- [ ] Task 3.4 - Class-imbalance experiment
- [ ] Task 3.5 - Calibration assessment
- [ ] Task 3.6 - Fit final v1 Layer-1 model on first 70%
- [ ] Task 3.7 - Validation inference
- [ ] Task 3.8 - Select `tau_tx`
- [ ] Task 3.9 - Freeze Layer-1 bundle
- [ ] Phase 3 Validation Gate

## Phase 4 - Time Aggregation & Spike Detection (Layer 2)
- [ ] Task 4.1 - Layer-2 input contract
- [ ] Task 4.2 - Tumbling 3,600-second aggregator
- [ ] Task 4.3 - Empty-window emission
- [ ] Task 4.4 - Rolling baseline
- [ ] Task 4.5 - Z-score detector
- [ ] Task 4.6 - Stateful EWMA
- [ ] Task 4.7 - Stateful CUSUM
- [ ] Task 4.8 - Isolation Forest comparator
- [ ] Task 4.9 - Layer-2 validation comparison
- [ ] Task 4.10 - Freeze Layer-2 configuration
- [ ] Phase 4 Validation Gate

## Phase 5 - Replay Simulator, Alerts, Proxy Truth, and Cost Framework
- [ ] Task 5.1 - Replay engine
- [ ] Task 5.2 - Replay state serialization
- [ ] Task 5.3 - Offline actual-fraud-rate windows
- [ ] Task 5.4 - Minimum-volume eligibility
- [ ] Task 5.5 - Proxy spike thresholds
- [ ] Task 5.6 - Event merging
- [ ] Task 5.7 - Event detection matching
- [ ] Task 5.8 - Severity engine
- [ ] Task 5.9 - Alert payload builder
- [ ] Task 5.10 - Alert dispatcher abstraction
- [ ] Task 5.11 - Parameterized cost model
- [ ] Task 5.12 - Cost sensitivity curves
- [ ] Phase 5 Validation Gate

## Phase 6 - Streamlit Dashboard and Final Held-Out Evaluation
- [ ] Task 6.1 - Test-seal audit
- [ ] Task 6.2 - Final chronological replay
- [ ] Task 6.3 - Transaction-level final metrics
- [ ] Task 6.4 - Test proxy-event construction
- [ ] Task 6.5 - Spike-level final metrics
- [ ] Task 6.6 - Detection latency report
- [ ] Task 6.7 - Dashboard data contract
- [ ] Task 6.8 - Streamlit dashboard
- [ ] Task 6.9 - Final reproducibility bundle
