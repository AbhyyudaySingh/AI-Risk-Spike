# AI Fraud-Spike Detector
## AI Agent Implementation Blueprint v1.0

**Purpose:** Self-contained implementation handoff for an AI coding agent  
**Canonical source:** Technical Specification v1.0  
**Project scope:** Defense-only e-commerce/payment fraud-spike monitoring  
**Primary dataset:** IEEE-CIS Fraud Detection  
**Primary evaluation:** Strict chronological out-of-time replay  
**Code status:** No implementation code is contained in this document. This is the build plan and architecture contract.

---

# 0. Agent Mission

Build a reproducible, leakage-safe, two-layer fraud-risk monitoring system:

1. **Layer 1 - Transaction Risk**
   - XGBoost binary classifier.
   - Produces continuous `fraud_probability`.
   - Produces `suspicious_flag` using validation-selected `tau_tx`.

2. **Layer 2 - Fraud Spike Detection**
   - Aggregates Layer-1 risk signals into fixed relative-time windows.
   - Estimates a past-only rolling baseline.
   - Compares rolling z-score, EWMA, CUSUM, and an Isolation Forest benchmark.
   - Produces structured risk states: `NORMAL`, `ELEVATED`, `HIGH`.

The system must never use future information at prediction time, must never accuse customers automatically, and must never optimize anything using the final held-out test labels.

---

# 1. Non-Negotiable Engineering Guardrails

## 1.1 Chronological discipline

- Sort by `TransactionDT` ascending before any split or temporal feature construction.
- `TransactionDT` is relative elapsed seconds, not a real calendar timestamp.
- Do not infer real hour-of-day or calendar-day semantics from `TransactionDT`.
- Outer split is chronological: earliest 70% train, next 15% validation, latest 15% test.
- Never shuffle before outer split.
- Never fit preprocessing on validation or test.
- Never use future entity counts, future rolling statistics, future labels, or global dataset statistics.
- For transactions sharing the same `TransactionDT`, features for all rows at that timestamp must be computed before state is updated with any row from that same timestamp.

## 1.2 Label isolation

`isFraud` is allowed only in:
- supervised Layer-1 training;
- validation/test evaluation;
- offline spike proxy-ground-truth generation.

`isFraud` is forbidden from:
- prediction-time feature generation;
- replay state;
- aggregation inputs;
- spike detector inputs;
- alert payload generation.

## 1.3 Two-layer separation

Layer 1 and Layer 2 are separate modules with explicit contracts.

Layer 2 consumes:
- transaction timestamp;
- transaction amount where needed for exposure telemetry;
- `fraud_probability`;
- `suspicious_flag`;
- non-label contextual fields that are explicitly approved.

Layer 2 must not require `isFraud`.

## 1.4 Cost discipline

Business costs remain configurable parameters.

Do not present development test values as merchant economics.

Required cost parameters:
- `c_review`
- `c_friction`
- `LGF`
- `c_fraud_ops`
- `intervention_effectiveness`
- `c_alert_review`

Final reporting must support `LOW`, `BASE`, and `HIGH` scenarios.

## 1.5 Defense-only output

Allowed:
- anomaly alerts;
- review recommendations;
- increased verification recommendations;
- monitoring telemetry.

Disallowed:
- automatic accusation;
- offensive fraud simulation;
- fraud evasion tools;
- permanent blocking solely from spike output.

---

# 2. Locked Decisions for the 15 Former Open Questions

These choices are now part of v1.0 and should not be re-opened during implementation unless a hard technical failure is documented.

## Decision 1 - Window type

**Use non-overlapping tumbling windows for v1.**

Primary window duration:
- 3,600 relative seconds.

Reason:
- deterministic event boundaries;
- clean alert counting;
- clean proxy-event construction;
- simpler latency interpretation;
- no duplicate alert inflation from overlapping windows.

Sliding windows may be explored only after v1 is complete.

## Decision 2 - Window anchor

Use a fixed relative-time grid:

`window_id = floor(TransactionDT / window_size_seconds)`

For the primary 3,600-second setting:
- window 0: `[0, 3600)`
- window 1: `[3600, 7200)`
- etc.

Do not anchor windows to the first observed transaction.

## Decision 3 - Empty windows

**Emit empty windows for temporal continuity.**

Policy:
- `transaction_count = 0`;
- risk-derived signals are `NA`, not zero;
- z-score/EWMA/CUSUM state is not updated from an empty observation;
- alert status is `NO_DATA` internally and maps to no external fraud alert;
- elapsed window time remains preserved for latency calculations.

If consecutive empty windows reach or exceed the selected rolling-history length, reset the detector to warm-up state.

## Decision 4 - Categorical encoding

**Primary XGBoost path: XGBoost native categorical handling.**

Pipeline:
- categorical missing values -> explicit `Unknown`;
- convert approved categorical columns to a stable categorical dtype;
- freeze the category vocabulary from the training period;
- unseen validation/test values -> `Unknown`.

For the simple logistic-regression baseline only:
- use bounded one-hot encoding with `handle_unknown=ignore`.

Do not use target encoding in v1.

Do not use naive one-hot encoding for high-cardinality features in the primary XGBoost path.

## Decision 5 - Probability calibration

**Evaluate calibration first; apply calibration only if validation shows material improvement.**

Evaluation:
- reliability curve;
- Brier score;
- calibration slope/intercept;
- ECE or equivalent.

Preferred calibrator if needed:
1. sigmoid/Platt first;
2. isotonic only if it clearly performs better and has adequate support.

Calibration must use chronological out-of-fold/pre-test predictions only.

If calibration does not improve validation calibration quality, preserve raw XGBoost probabilities.

## Decision 6 - Recall@Precision reporting

Report multiple operating points:

- Recall@Precision >= 50%
- Recall@Precision >= 70%
- Recall@Precision >= 80%
- Recall@Precision >= 90%

If a target precision is unattainable, report `N/A` rather than forcing a threshold.

The actual `tau_tx` remains cost/validation selected and is not automatically one of these operating points.

## Decision 7 - Definition of high risk

For v1:

`high_risk == suspicious`

Therefore:
- `high_risk_count = suspicious_count`;
- `high_risk_amount = amount associated with suspicious_flag = 1`.

Do not introduce a third transaction threshold in v1.

## Decision 8 - Final model refit

**Do not refit Layer 1 on train + validation before final testing.**

Final held-out evaluation uses:
- preprocessing fitted on earliest 70%;
- XGBoost trained on earliest 70%;
- optional calibrator selected/fitted according to the frozen pre-test protocol;
- `tau_tx` selected on the next 15%;
- Layer-2 parameters selected on the next 15%;
- latest 15% untouched until final replay.

Reason:
- preserves calibration validity;
- preserves transaction threshold validity;
- produces the cleanest out-of-time test.

An 85%-refit model may be a later production experiment, not the headline v1 result.

## Decision 9 - Baseline warm-up

Let validation select rolling-history length `N`.

**No spike alert may be emitted until N eligible non-empty historical windows exist.**

EWMA/CUSUM are initialized from the same pre-alert historical baseline.

The dashboard may display `WARMING_UP`, but this is not an anomaly state.

## Decision 10 - Minimum volume for proxy spike labels

Primary rule:

`min_eligible_volume = 10th percentile of transaction_count among non-empty 3,600-second training windows`

Rules:
- computed from training only;
- frozen before test;
- validation confirms that the rule is not pathological;
- test does not modify it.

Windows below the threshold are excluded from proxy spike ground-truth labeling.

## Decision 11 - Event detection tolerance

Headline metric:

**Strict during-event detection.**

A true proxy event is detected only if at least one alert occurs between event onset and event end.

Secondary operational metric:
- allow one subsequent window as a late-detection tolerance.

Both are reported, but strict during-event detection is the headline.

## Decision 12 - EWMA/CUSUM input

Primary signal:
- `suspicious_rate`.

Secondary experiment:
- `mean_fraud_probability`.

Do not combine the two into an opaque composite score until their separate behavior is understood.

## Decision 13 - Isolation Forest inputs

Use a compact window-level feature vector:

- `transaction_count`
- `suspicious_rate`
- `mean_fraud_probability`
- `sum_fraud_probability`
- `high_risk_amount_ratio = high_risk_amount / total_transaction_amount` when denominator > 0

Do not initially add large product/device mix vectors.

The comparator is intended to answer whether a compact unsupervised window model adds value over transparent statistical detectors.

## Decision 14 - Anomaly drivers

Primary dashboard explanation:
- observable window-level contributors;
- changes in suspicious volume;
- high-risk amount;
- card/product/device/address group shifts when supported.

Do not make SHAP a v1 dashboard dependency.

Transaction-level SHAP may be added later as an optional Layer-1 diagnostic, but not as a causal explanation.

## Decision 15 - Dashboard framework

Use **Streamlit** for v1.

Reason:
- Python-native;
- low implementation overhead;
- appropriate for research/demo tooling;
- easy consumption of frozen parquet/JSON artifacts;
- no need for a separate frontend/backend stack in v1.

The dashboard is read-only with respect to model state and must never recompute or retune predictions.

---

# 3. Repository Layout

```text
fraud-spike-detector/
|
|-- README.md
|-- pyproject.toml
|-- .gitignore
|
|-- configs/
|   |-- data.yaml
|   |-- splits.yaml
|   |-- features.yaml
|   |-- transaction_model.yaml
|   |-- calibration.yaml
|   |-- spike_detection.yaml
|   |-- costs.yaml
|   `-- dashboard.yaml
|
|-- data/
|   |-- raw/
|   |   |-- train_transaction.csv
|   |   `-- train_identity.csv
|   |-- interim/
|   `-- processed/
|       |-- train/
|       |-- validation/
|       `-- test/
|
|-- artifacts/
|   |-- manifests/
|   |-- preprocessing/
|   |-- models/
|   |   |-- transaction/
|   |   `-- spike/
|   |-- thresholds/
|   |-- predictions/
|   |-- windows/
|   |-- alerts/
|   `-- metrics/
|
|-- src/
|   `-- fraud_spike/
|       |-- common/
|       |   |-- contracts.py
|       |   |-- config.py
|       |   |-- logging.py
|       |   `-- manifest.py
|       |
|       |-- data/
|       |   |-- schemas.py
|       |   |-- ingest.py
|       |   |-- join.py
|       |   |-- audit.py
|       |   `-- split.py
|       |
|       |-- features/
|       |   |-- registry.py
|       |   |-- missing.py
|       |   |-- categorical.py
|       |   |-- transaction.py
|       |   |-- history.py
|       |   `-- pipeline.py
|       |
|       |-- models/
|       |   |-- baseline.py
|       |   |-- xgboost_model.py
|       |   |-- temporal_cv.py
|       |   |-- calibration.py
|       |   `-- thresholds.py
|       |
|       |-- spike/
|       |   |-- aggregate.py
|       |   |-- rolling_baseline.py
|       |   |-- zscore.py
|       |   |-- ewma.py
|       |   |-- cusum.py
|       |   |-- isolation_forest.py
|       |   |-- ground_truth.py
|       |   `-- events.py
|       |
|       |-- replay/
|       |   |-- engine.py
|       |   `-- state.py
|       |
|       |-- alerts/
|       |   |-- policy.py
|       |   |-- payload.py
|       |   `-- dispatcher.py
|       |
|       |-- costs/
|       |   |-- scenarios.py
|       |   `-- calculator.py
|       |
|       |-- evaluation/
|       |   |-- transaction_metrics.py
|       |   |-- spike_metrics.py
|       |   `-- report.py
|       |
|       `-- dashboard/
|           |-- app.py
|           |-- data_contract.py
|           `-- views.py
|
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- leakage/
|       |-- test_temporal_features.py
|       |-- test_preprocessing_fit_scope.py
|       |-- test_split_boundaries.py
|       `-- test_future_label_access.py
|
|-- notebooks/
|   `-- 01_eda.ipynb
|
`-- reports/
    |-- eda/
    |-- validation/
    `-- final/
```

Rule:
- notebooks are exploratory;
- production/reproducible logic belongs in `src/`;
- frozen data/model outputs belong in `artifacts/`;
- no dashboard logic belongs in training modules.

---

# 4. Module Dependency Contract

```text
common
  |
  v
data
  |
  v
features
  |
  v
models
  |
  v
spike
  |
  v
replay
  |------> alerts
  |------> costs
  `------> evaluation
              |
              v
           dashboard
```

Forbidden dependency directions:
- `models` must not import `evaluation`;
- `spike` must not import offline ground-truth labels during live scoring;
- `replay` must not import test labels;
- `dashboard` must not call fit/tune functions.

---

# 5. Core Artifact Contracts

## 5.1 Transaction source record

Required:
- `TransactionID`
- `TransactionDT`
- `TransactionAmt`
- approved raw feature fields

Offline-only:
- `isFraud`

## 5.2 Transaction prediction artifact

Fields:
- `TransactionID`
- `TransactionDT`
- `fraud_probability`
- `suspicious_flag`
- `model_version`

No label required for replay consumption.

## 5.3 Window telemetry artifact

Fields:
- `window_id`
- `window_start_relative`
- `window_end_relative`
- `transaction_count`
- `suspicious_count`
- `suspicious_rate`
- `mean_fraud_probability`
- `sum_fraud_probability`
- `high_risk_count`
- `total_transaction_amount`
- `high_risk_amount`
- `high_risk_amount_ratio`
- `data_quality_status`

## 5.4 Spike telemetry artifact

Fields:
- `window_id`
- `baseline_value`
- `baseline_std`
- `baseline_history_count`
- `z_score`
- `ewma_score`
- `cusum_score`
- `isolation_score`
- `selected_spike_score`
- `detector_state`
- `severity`

## 5.5 Alert artifact

Fields:
- `alert_id`
- `window_start_relative`
- `window_end_relative`
- `severity`
- `current_signal`
- `expected_baseline`
- `deviation`
- `spike_score`
- `transaction_count`
- `high_risk_count`
- `high_risk_amount`
- `model_version`
- `reason`
- `recommended_defensive_action`

---

# 6. Phase 1 - Data Pipeline & Strict Chronological Split

## Phase objective

Produce an audited, joined, chronologically ordered dataset and frozen 70/15/15 split boundaries without fitting any model transformation.

---

## Task 1.1 - Raw file ingestion

**Objective**
- Load IEEE-CIS labeled transaction and identity files exactly as provided.
- Record immutable source metadata.

**Input Artifact**
- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

**Output Artifact**
- canonical raw in-memory frames;
- `artifacts/manifests/dataset_manifest.json`.

**Verification Method**
- file presence;
- expected key columns;
- row/column counts;
- hashes/checksums;
- deterministic reload.

**Gate**
- fail if `TransactionID`, `TransactionDT`, or `isFraud` is unavailable in transaction data.

---

## Task 1.2 - Integrity and target audit

**Objective**
- Validate basic transaction-table integrity.

**Input Artifact**
- raw transaction frame.

**Output Artifact**
- `reports/eda/raw_integrity.json`.

**Verification Method**
- `TransactionID` uniqueness;
- target values only `{0,1}`;
- duplicate-row count;
- `TransactionDT` missingness;
- `TransactionAmt` validity summary.

---

## Task 1.3 - Identity join

**Objective**
- Add available identity/device fields while preserving every transaction.

**Input Artifact**
- transaction table;
- identity table.

**Output Artifact**
- `data/interim/joined_transactions.parquet`.

**Verification Method**
- left join on `TransactionID`;
- output row count equals transaction row count;
- no duplicate transaction rows introduced;
- matched identity percentage recorded.

---

## Task 1.4 - Feature-type audit

**Objective**
- Assign each source field to a semantic modeling class.

**Input Artifact**
- joined table.

**Output Artifact**
- `artifacts/manifests/schema_manifest.json`;
- `reports/eda/feature_type_audit.csv`.

**Required classes**
- identifier;
- target;
- numeric;
- categorical;
- relative time;
- masked numeric;
- masked categorical;
- excluded.

**Verification Method**
- every column has exactly one class;
- `TransactionID` = identifier;
- `isFraud` = target;
- `TransactionDT` = relative time.

---

## Task 1.5 - Missingness audit

**Objective**
- Quantify missingness before imputation decisions.

**Input Artifact**
- joined chronological-source table.

**Output Artifact**
- `reports/eda/missingness.csv`.

**Required statistics**
- missing count;
- missing percentage;
- missingness by label;
- missingness by broad chronological segment.

**Verification Method**
- compare report totals against direct null counts.

---

## Task 1.6 - Chronological ordering

**Objective**
- Establish the canonical temporal sequence.

**Input Artifact**
- joined table.

**Output Artifact**
- `data/interim/chronological_transactions.parquet`.

**Method**
- sort `TransactionDT` ascending;
- preserve deterministic tie handling.

**Verification Method**
- assert monotonic non-decreasing `TransactionDT`;
- assert row count unchanged.

---

## Task 1.7 - Same-timestamp policy

**Objective**
- Prevent within-timestamp pseudo-future leakage.

**Input Artifact**
- sorted transactions.

**Output Artifact**
- grouping metadata or deterministic timestamp batches.

**Policy**
For each unique timestamp `T`:
1. read state from `< T`;
2. generate features for every row at `T`;
3. score rows at `T`;
4. update history only after all rows at `T` have been processed.

**Verification Method**
- synthetic unit case with multiple rows sharing timestamp;
- prove order permutation inside timestamp does not change features.

---

## Task 1.8 - Freeze 70/15/15 outer split

**Objective**
- Create immutable out-of-time partitions.

**Input Artifact**
- chronologically sorted transactions.

**Output Artifact**
- `artifacts/manifests/split_manifest.json`;
- processed split references/files.

**Nominal split**
- earliest 70% = train;
- next 15% = validation;
- latest 15% = test.

**Boundary rule**
- never split identical `TransactionDT` values across partitions;
- shift boundary to a unique timestamp boundary.

**Verification Method**
- `max(train_time) < min(validation_time)`;
- `max(validation_time) < min(test_time)`;
- no transaction-ID overlap;
- split hashes recorded.

---

## Phase 1 Validation Gate

Do not proceed unless:
- source integrity passes;
- join preserves row count;
- chronology passes;
- timestamp ties are safe;
- test is sealed;
- `isFraud` and `TransactionID` are marked non-predictive.

---

# 7. Phase 2 - Leakage-Safe Feature Engineering & Preprocessing

## Phase objective

Create train-only preprocessing and streaming-safe behavioral features.

---

## Task 2.1 - Feature registry

**Objective**
- Declare every candidate input before training.

**Input Artifact**
- schema manifest;
- EDA outputs.

**Output Artifact**
- `artifacts/manifests/feature_manifest.json`.

**Required metadata per feature**
- name;
- source columns;
- feature type;
- prediction-time availability;
- history requirement;
- entity key;
- lookback definition;
- missing strategy;
- enabled/disabled state.

**Verification Method**
- no enabled feature references `isFraud`;
- no feature uses future records;
- no predictive feature is derived directly from `TransactionID`.

---

## Task 2.2 - Numerical imputation

**Objective**
- Fill selected numerical missing values without leakage.

**Input Artifact**
- numerical train features.

**Output Artifact**
- train-fitted median imputer artifact.

**Policy**
- fit median on current training fold only during CV;
- fit on full 70% train for final Layer-1 model.

**Verification Method**
- mutate validation/test distributions and confirm fitted medians do not change.

---

## Task 2.3 - Categorical missing values and vocabulary

**Objective**
- Stabilize categorical inputs for native XGBoost categoricals.

**Input Artifact**
- categorical training data.

**Output Artifact**
- frozen category-vocabulary artifact.

**Policy**
- missing -> `Unknown`;
- category vocabulary learned from training only;
- unseen future values -> `Unknown`.

**Verification Method**
- unseen categories do not crash;
- category map is unchanged by validation/test.

---

## Task 2.4 - Logistic baseline encoding

**Objective**
- Support the simple classifier benchmark.

**Input Artifact**
- selected categorical/numeric training fields.

**Output Artifact**
- bounded one-hot baseline preprocessor.

**Policy**
- `handle_unknown=ignore`;
- no high-cardinality expansion that exceeds configured safety limits.

**Verification Method**
- transformed shape deterministic;
- no validation categories create refit.

---

## Task 2.5 - Missingness indicators

**Objective**
- Preserve potentially predictive absence patterns.

**Input Artifact**
- original pre-imputation null masks.

**Output Artifact**
- selected `_missing` binary columns.

**Verification Method**
- each flag exactly matches original null state.

---

## Task 2.6 - Raw transaction feature transformations

**Objective**
- Create safe deterministic transformations.

**Candidate Input**
- `TransactionAmt`;
- `TransactionDT`;
- approved source features.

**Output Artifact**
- raw transformed feature block.

**Examples**
- log amount if validation supports;
- relative window ID;
- elapsed-day index.

**Constraint**
- do not construct fake real-world hour-of-day.

**Verification Method**
- deterministic equality across reruns.

---

## Task 2.7 - Stateful past-only entity features

**Objective**
- Build behavioral/velocity features without future leakage.

**Candidate entities**
- approved card representation;
- device;
- address;
- product;
- selected combinations.

**Candidate outputs**
- prior transaction count;
- prior mean/median amount;
- amount deviation from prior entity behavior;
- prior unique devices per card;
- prior unique cards per device;
- time since prior entity transaction;
- transaction velocity.

**Input Artifact**
- timestamp-batched chronological stream.

**Output Artifact**
- history-feature block.

**Verification Method**
Synthetic sequence:
- first entity occurrence -> prior count 0;
- second -> 1;
- third -> 2.

Future-deletion invariant:
- removing every row after time `T` must not change any feature at or before `T`.

---

## Task 2.8 - CV-specific preprocessing artifacts

**Objective**
- Fit transformations independently for every expanding fold.

**Input Artifact**
- fold boundaries.

**Output Artifact**
- one preprocessing artifact per fold;
- one final 70%-train preprocessing artifact.

**Verification Method**
- artifact metadata records fit-time max `TransactionDT`;
- validation times must be strictly greater.

---

## Phase 2 Validation Gate

Must pass:
- global-statistics leakage tests;
- future-category leakage tests;
- future entity-count tests;
- same-timestamp invariance;
- target leakage scan;
- deterministic feature manifest.

---

# 8. Phase 3 - Transaction-Risk Model (Layer 1)

## Phase objective

Train, validate, optionally calibrate, and freeze the transaction-level fraud-risk model.

---

## Task 3.1 - Expanding temporal CV folds

**Objective**
- Tune Layer 1 without random validation.

**Fold plan**
- Fold 1: train 0-30%, validate 30-40%;
- Fold 2: train 0-40%, validate 40-50%;
- Fold 3: train 0-50%, validate 50-60%;
- Fold 4: train 0-60%, validate 60-70%.

**Output Artifact**
- `artifacts/manifests/cv_folds.json`.

**Verification Method**
- all validation timestamps occur strictly after corresponding training timestamps.

---

## Task 3.2 - Simple supervised baseline

**Objective**
- Establish minimum benchmark.

**Input Artifact**
- leakage-safe fold matrices.

**Output Artifact**
- fold probabilities;
- CV metrics.

**Metrics**
- PR-AUC;
- precision;
- recall;
- ROC-AUC secondary.

**Verification Method**
- all predictions are out-of-fold and chronological.

---

## Task 3.3 - XGBoost temporal-CV training

**Objective**
- Select Layer-1 hyperparameters.

**Input Artifact**
- fold-specific preprocessing outputs.

**Output Artifact**
- fold models;
- OOF probabilities;
- experiment table.

**Verification Method**
- each model records max train timestamp;
- no fold uses validation labels for feature construction.

---

## Task 3.4 - Class-imbalance experiment

**Objective**
- Evaluate appropriate positive-class weighting.

**Input Artifact**
- training folds.

**Output Artifact**
- comparison of unweighted vs selected `scale_pos_weight` settings.

**Constraint**
- no sampling/weight choice based on final test.

**Verification Method**
- compare PR-AUC and precision/recall trade-offs.

---

## Task 3.5 - Calibration assessment

**Objective**
- Determine whether probability calibration is needed for aggregation.

**Input Artifact**
- chronological OOF probabilities.

**Output Artifact**
- calibration report.

**Metrics**
- reliability plot data;
- Brier score;
- calibration slope/intercept;
- ECE or equivalent.

**Decision**
- if calibration improves validation calibration, use it;
- otherwise retain raw XGBoost probabilities.

**Preferred calibrator**
- sigmoid/Platt;
- isotonic only if clearly superior and sufficiently supported.

**Verification Method**
- calibrator sees pre-test OOF predictions only.

---

## Task 3.6 - Fit final v1 Layer-1 model on first 70%

**Objective**
- Freeze the model used for both validation and final test.

**Input Artifact**
- complete 70% training period.

**Output Artifact**
- XGBoost model;
- preprocessing artifact;
- optional calibrator;
- model manifest.

**Verification Method**
- training max timestamp equals frozen 70% boundary;
- no 15% validation records are used for fit.

---

## Task 3.7 - Validation inference

**Objective**
- Produce the risk stream used for threshold and Layer-2 tuning.

**Input Artifact**
- next 15% validation period.

**Output Artifact**
- validation transaction predictions.

**Required fields**
- `TransactionID`;
- `TransactionDT`;
- `fraud_probability`;
- offline `isFraud` stored separately for evaluation.

**Verification Method**
- model artifact version matches frozen 70% model.

---

## Task 3.8 - Select `tau_tx`

**Objective**
- Convert probability into `suspicious_flag`.

**Input Artifact**
- validation predictions;
- parameterized cost configurations.

**Output Artifact**
- `artifacts/thresholds/tau_tx.json`.

**Required threshold table**
For each candidate threshold:
- precision;
- recall;
- false positives;
- false negatives;
- alert volume;
- parameterized cost.

**Also report**
- Recall@Precision >= 50%;
- >= 70%;
- >= 80%;
- >= 90%.

**Verification Method**
- threshold selection uses validation only.

---

## Task 3.9 - Freeze Layer-1 bundle

**Objective**
- Make Layer 1 immutable before final test.

**Output Artifact**
Layer-1 manifest containing:
- feature-manifest hash;
- preprocessing version;
- category vocabulary;
- XGBoost hyperparameters;
- calibration config;
- `tau_tx`;
- model version;
- train boundary.

**Verification Method**
- bundle checksum;
- reproducible validation predictions.

---

## Phase 3 Validation Gate

Layer 1 passes only if:
- temporal CV is clean;
- simple baseline comparison exists;
- XGBoost OOF metrics exist;
- calibration was evaluated;
- `tau_tx` is validation-selected;
- no test data influenced any decision.

---

# 9. Phase 4 - Time Aggregation & Spike Detection (Layer 2)

## Phase objective

Convert the transaction-risk stream into stable time-series signals and compare transparent/statistical spike detectors.

---

## Task 4.1 - Layer-2 input contract

**Objective**
- Enforce label isolation.

**Input Artifact**
- Layer-1 transaction predictions.

**Allowed fields**
- `TransactionID`;
- `TransactionDT`;
- `TransactionAmt`;
- `fraud_probability`;
- `suspicious_flag`;
- approved non-label contextual fields.

**Output Artifact**
- validated risk stream.

**Verification Method**
- schema must reject `isFraud` in live Layer-2 scoring API.

---

## Task 4.2 - Tumbling 3,600-second aggregator

**Objective**
- Create the primary relative-time windows.

**Window rule**
- `window_id = floor(TransactionDT / 3600)`.

**Output Artifact**
- validation-window telemetry.

**Required metrics**
- transaction count;
- suspicious count;
- suspicious rate;
- mean probability;
- sum probability;
- high-risk count;
- total amount;
- high-risk amount;
- high-risk amount ratio.

**Verification Method**
Boundary unit tests:
- time 3599 -> window 0;
- time 3600 -> window 1;
- time 3601 -> window 1.

---

## Task 4.3 - Empty-window emission

**Objective**
- Preserve time continuity.

**Input Artifact**
- observed window IDs.

**Output Artifact**
- dense relative-time window sequence.

**Policy**
Empty:
- transaction count 0;
- risk signals NA;
- no detector update;
- no fraud alert;
- internal status `NO_DATA`.

**Verification Method**
- intentionally remove a window and confirm gap is emitted.

---

## Task 4.4 - Rolling baseline

**Objective**
- Estimate expected suspicious behavior from past windows only.

**Primary signal**
- suspicious rate.

**Input Artifact**
- eligible previous non-empty windows.

**Output Artifact**
- rolling mean;
- rolling std;
- history count.

**Policy**
- current window excluded;
- rolling-history length `N` tuned on validation;
- no external alert until `N` eligible historical windows exist.

**Verification Method**
- future-window perturbation must not change current baseline.

---

## Task 4.5 - Z-score detector

**Objective**
- Implement transparent primary anomaly benchmark.

**Output Artifact**
- per-window z-score;
- state;
- candidate alert flag.

**Edge rules**
- insufficient history -> `WARMING_UP`;
- zero/near-zero std -> apply explicit epsilon-safe policy and record diagnostic;
- empty window -> no update.

**Verification Method**
- synthetic constant baseline then injected jump.

---

## Task 4.6 - Stateful EWMA

**Objective**
- Detect gradual sustained increases.

**Primary input**
- suspicious rate.

**Secondary experiment**
- mean fraud probability.

**Output Artifact**
- EWMA level/deviation/score.

**Verification Method**
- synthetic gradual drift;
- state serialization round-trip.

---

## Task 4.7 - Stateful CUSUM

**Objective**
- Detect accumulated positive shifts.

**Primary input**
- suspicious rate.

**Secondary experiment**
- mean fraud probability.

**Output Artifact**
- positive CUSUM state;
- candidate score/alert.

**Verification Method**
- sustained small shift accumulates;
- baseline behavior does not drift indefinitely;
- state reload preserves result.

---

## Task 4.8 - Isolation Forest comparator

**Objective**
- Benchmark compact unsupervised window anomaly detection.

**Input features**
- transaction count;
- suspicious rate;
- mean fraud probability;
- sum fraud probability;
- high-risk amount ratio.

**Training data**
- pre-validation historical windows only.

**Output Artifact**
- anomaly score per validation/test window.

**Verification Method**
- model fit-time manifest ends before scored period.

---

## Task 4.9 - Layer-2 validation comparison

**Objective**
- Compare detectors fairly.

**Input Artifact**
- validation-window predictions;
- validation proxy events from Phase 5.

**Output Artifact**
- detector comparison table.

**Metrics**
- event precision;
- event recall;
- strict detection latency;
- false-alert rate;
- parameterized cost.

**Verification Method**
- common ground-truth event definition across models.

---

## Task 4.10 - Freeze Layer-2 configuration

**Objective**
- Lock final spike detector before test.

**Output Artifact**
- spike model manifest containing:
  - 3,600s primary window;
  - chosen baseline length;
  - z parameters;
  - EWMA parameters;
  - CUSUM parameters;
  - Isolation Forest parameters;
  - chosen primary detector;
  - `tau_spike`;
  - severity boundaries.

**Verification Method**
- manifest checksum;
- deterministic validation alerts.

---

## Phase 4 Validation Gate

Must pass:
- current window excluded from baseline;
- empty windows handled;
- warm-up enforced;
- detector state serializable;
- test unseen.

---

# 10. Phase 5 - Replay Simulator, Alerts, Proxy Truth, and Cost Framework

## Phase objective

Build a deterministic live-style replay and offline evaluation apparatus without allowing labels into prediction.

---

## Task 5.1 - Replay engine

**Objective**
- Reproduce production-like chronological processing.

**Input Artifact**
- chronologically ordered validation/test records.

**Per-timestamp sequence**
1. read state from prior timestamps;
2. create transaction features;
3. score Layer 1;
4. assign suspicious flag;
5. add to active tumbling window;
6. close windows when time boundary is crossed;
7. calculate Layer-2 score;
8. evaluate alert policy;
9. persist telemetry;
10. update entity history after all rows at timestamp are scored.

**Output Artifact**
- transaction predictions;
- window telemetry;
- state snapshots;
- alerts.

**Verification Method**
- same input + same config -> same outputs.

---

## Task 5.2 - Replay state serialization

**Objective**
- Allow deterministic pause/resume.

**State includes**
- entity histories;
- active window;
- rolling baseline;
- warm-up state;
- EWMA;
- CUSUM;
- detector metadata;
- model versions.

**Output Artifact**
- replay checkpoint.

**Verification Method**
- full uninterrupted replay equals paused/resumed replay.

---

## Task 5.3 - Offline actual-fraud-rate windows

**Objective**
- Construct evaluation-only window truth.

**Input Artifact**
- final transaction labels after replay.

**Output Artifact**
- offline window truth table.

**Formula**
`actual_fraud_rate = fraud_count / transaction_count`

**Verification Method**
- module is not imported into live prediction path.

---

## Task 5.4 - Minimum-volume eligibility

**Objective**
- Remove unstable very-low-volume windows from proxy event truth.

**Policy**
- calculate non-empty training 3,600-second window transaction counts;
- set primary minimum to 10th percentile;
- freeze;
- validation sanity-check only;
- test cannot alter.

**Output Artifact**
- `min_eligible_volume` in ground-truth manifest.

**Verification Method**
- recomputation from test must not occur.

---

## Task 5.5 - Proxy spike thresholds

**Objective**
- Define event truth independently of test predictions.

**Input Artifact**
- eligible validation windows with actual fraud rate.

**Output Artifact**
- thresholds:
  - 95th percentile;
  - 97.5th percentile primary;
  - 99th percentile.

**Test application**
- apply frozen values to test actual fraud rates after replay.

**Verification Method**
- threshold artifact timestamp precedes final test evaluation.

---

## Task 5.6 - Event merging

**Objective**
- Convert consecutive positive windows into event units.

**Output Artifact**
Per event:
- event ID;
- onset window;
- end window;
- peak actual fraud rate;
- total transactions;
- fraud count.

**Verification Method**
- deterministic consecutive-window merging tests.

---

## Task 5.7 - Event detection matching

**Objective**
- Match alerts to proxy true events.

**Headline rule**
- alert occurs during event.

**Secondary rule**
- alert may occur in the immediately following window.

**Output Artifact**
- matched/unmatched events and alerts.

**Verification Method**
- synthetic timing cases around onset/end boundaries.

---

## Task 5.8 - Severity engine

**Objective**
- Map frozen spike score policy to operational state.

**External states**
- `NORMAL`;
- `ELEVATED`;
- `HIGH`.

**Internal non-alert states**
- `WARMING_UP`;
- `NO_DATA`.

**Output Artifact**
- severity per eligible window.

**Verification Method**
- threshold-boundary tests.

---

## Task 5.9 - Alert payload builder

**Objective**
- Generate structured defense-only alert records.

**Output Artifact**
Required:
- alert ID;
- relative window start/end;
- severity;
- current signal;
- expected baseline;
- deviation;
- spike score;
- transaction count;
- high-risk count;
- high-risk amount;
- model version;
- reason;
- recommended defensive action.

**Verification Method**
- schema validation;
- no customer accusation language;
- no `isFraud` field.

---

## Task 5.10 - Alert dispatcher abstraction

**Objective**
- Separate detection from presentation/delivery.

**v1 sink**
- persistent JSON/parquet alert store;
- Streamlit-readable event feed.

**Output Artifact**
- alert log.

**Verification Method**
- dispatcher failure cannot alter model state or predictions.

---

## Task 5.11 - Parameterized cost model

**Objective**
- Implement business evaluation without fabricated economics.

**Input Artifact**
- metric outcomes;
- configured scenario parameters.

**Output Artifact**
- per-scenario cost table.

**Equations**
False positive:
`Cost_FP = c_review + c_friction`

False negative:
`Cost_FN = TransactionAmt * LGF + c_fraud_ops`

Detected fraud estimated avoided loss:
`AvoidedLoss_TP = TransactionAmt * LGF * intervention_effectiveness - c_review`

False spike alert:
`FalseSpikeAlertCost = false_spike_alert_count * c_alert_review`

**Verification Method**
- unit tests with toy parameter values;
- calculator accepts arbitrary config;
- no business value hardcoded inside functions.

---

## Task 5.12 - Cost sensitivity curves

**Objective**
- Show threshold/model sensitivity to economics.

**Scenarios**
- LOW;
- BASE;
- HIGH.

**Output Artifact**
For each model/threshold:
- false-positive cost;
- missed-fraud cost;
- avoided fraud loss;
- spike-alert cost;
- residual loss;
- net modeled benefit.

**Verification Method**
- monotonic sanity tests when individual cost parameters are increased.

---

## Phase 5 Validation Gate

Freeze before final test:
- `tau_tx`;
- window size;
- baseline length;
- selected spike detector;
- `tau_spike`;
- severity thresholds;
- minimum proxy volume;
- 95/97.5/99 proxy thresholds;
- event matching tolerance;
- cost-scenario schema;
- alert payload schema.

---

# 11. Phase 6 - Streamlit Dashboard and Final Held-Out Evaluation

## Phase objective

Run the untouched final 15% exactly once under the frozen protocol and present reproducible results.

---

## Task 6.1 - Test-seal audit

**Objective**
- Prove final test did not participate in development.

**Input Artifact**
- test partition;
- all frozen manifests.

**Output Artifact**
- `reports/final/test_seal_manifest.json`.

**Verification Method**
Confirm no test label dependency in:
- preprocessing;
- feature selection;
- calibration;
- `tau_tx`;
- baseline length;
- spike detector;
- `tau_spike`;
- ground-truth threshold creation;
- cost-policy tuning.

---

## Task 6.2 - Final chronological replay

**Objective**
- Generate test predictions without labels.

**Input Artifact**
- latest 15%;
- frozen Layer-1 and Layer-2 bundles.

**Output Artifact**
- test transaction predictions;
- window telemetry;
- alerts.

**Verification Method**
- replay engine runs with target column hidden/unavailable.

---

## Task 6.3 - Transaction-level final metrics

**Objective**
- Reveal labels offline and evaluate Layer 1.

**Output Artifact**
- `reports/final/transaction_metrics.json`.

**Headline metrics**
- PR-AUC;
- precision;
- recall;
- Recall@Precision 50/70/80/90 where attainable.

**Secondary**
- ROC-AUC;
- confusion matrix;
- false-positive rate;
- alert volume;
- parameterized cost.

**Verification Method**
- metrics recompute exactly from frozen predictions.

---

## Task 6.4 - Test proxy-event construction

**Objective**
- Apply frozen validation-derived event thresholds.

**Input Artifact**
- offline test labels;
- frozen min volume;
- frozen 95/97.5/99 thresholds.

**Output Artifact**
- test proxy event files for all sensitivity tiers.

**Verification Method**
- no percentile fitting on test.

---

## Task 6.5 - Spike-level final metrics

**Objective**
- Evaluate Layer 2.

**Output Artifact**
- `reports/final/spike_metrics.json`.

**Headline**
- event precision;
- event recall;
- false-alert rate;
- strict during-event detection latency;
- parameterized business cost.

**Secondary**
- one-window-late tolerance metrics;
- window-level precision/recall;
- alert count;
- event duration;
- peak severity;
- sensitivity across 95/97.5/99 truth tiers.

---

## Task 6.6 - Detection latency report

**Objective**
- Quantify speed of detection.

**Formula**
`latency = first_alert_time - event_onset_time`

**Output Artifact**
- latency distribution.

**Report**
- median;
- mean;
- P90;
- event-by-event values.

**Verification Method**
- only matched events included;
- relative-time units documented.

---

## Task 6.7 - Dashboard data contract

**Objective**
- Make UI read frozen outputs only.

**Input Artifact**
- window telemetry;
- alerts;
- final metrics;
- optional contributor tables.

**Output Artifact**
- dashboard-ready immutable files.

**Verification Method**
- dashboard has no model-fit or threshold-tune imports.

---

## Task 6.8 - Streamlit dashboard

**Objective**
- Provide minimal operational view.

**Required panels**

### A. Current status
- NORMAL / ELEVATED / HIGH;
- internal WARMING_UP / NO_DATA where relevant.

### B. Suspicious activity timeline
- suspicious rate;
- rolling baseline;
- alert threshold;
- detected events.

### C. Current vs expected
- current signal;
- baseline;
- absolute deviation;
- relative deviation;
- spike score.

### D. Window summary
- transaction count;
- suspicious/high-risk count;
- mean fraud probability;
- total amount;
- high-risk amount.

### E. Alert detail
- severity;
- onset;
- detection time;
- spike score;
- reason;
- defensive recommendation.

### F. Observable anomaly contributors
- high-risk-volume shift;
- amount shift;
- product/card/device/address group shifts when technically supported.

**Constraint**
- contributors are associations, not causal claims.

**Verification Method**
- UI reads artifact files only;
- no prediction recomputation;
- no customer accusation language.

---

## Task 6.9 - Final reproducibility bundle

**Objective**
- Make final experiment reproducible.

**Output Artifact**
`reports/final/run_manifest.json` containing:
- dataset hashes;
- split boundaries;
- feature-manifest hash;
- preprocessing version;
- XGBoost parameters;
- calibration decision;
- `tau_tx`;
- window size;
- baseline length;
- z/EWMA/CUSUM parameters;
- Isolation Forest parameters;
- selected spike detector;
- `tau_spike`;
- min proxy volume;
- 95/97.5/99 ground-truth thresholds;
- severity thresholds;
- event tolerance;
- cost config;
- random seeds;
- library/runtime versions;
- model versions.

**Verification Method**
- clean rerun reproduces metrics within deterministic tolerance.

---

# 12. Cross-Cutting Test Strategy

## 12.1 Leakage tests

Mandatory tests:

### Future deletion invariant
Deleting all data after timestamp `T` must not alter features/predictions at or before `T`.

### Preprocessing fit-scope test
Extreme validation/test values must not alter training median/category artifacts.

### Same-timestamp permutation test
Reordering rows within the same `TransactionDT` must not change their historical features.

### Label isolation test
Replay and spike live modules must operate without `isFraud`.

### Test-seal dependency test
No fit/select/optimize artifact may list final test labels as an input.

---

## 12.2 Boundary tests

Test:
- `TransactionDT = 3599`;
- `3600`;
- `3601`;
- split timestamp ties;
- empty window;
- first non-empty window after gap;
- zero rolling std;
- new categorical value;
- missing device info;
- event beginning/ending exactly on an alert window.

---

## 12.3 Determinism tests

Same:
- data;
- seed;
- config;
- model version

must generate identical:
- feature artifacts;
- probabilities where deterministic;
- windows;
- spike scores;
- alerts;
- metrics.

---

# 13. Performance and Optimization Strategy

Do not optimize prematurely.

Expected bottlenecks:

1. IEEE-CIS width and memory usage.
2. Expanding-window XGBoost CV.
3. High-cardinality categoricals.
4. Stateful entity-history generation.
5. Multiple validation replay experiments.

Recommended sequence:

### First
- correct small-sample pipeline;
- exhaustive leakage tests.

### Then
- parquet persistence;
- categorical memory optimization;
- vectorized safe transformations where possible;
- stateful streaming structures for history;
- cache fold-independent raw audit artifacts.

### Only after correctness
- parallel XGBoost trials;
- feature pruning;
- batch/state optimization.

Performance optimization must never replace chronological correctness.

---

# 14. Required Experiment Order

Run experiments in this order:

1. Simple transaction baseline.
2. XGBoost with cleaned raw features.
3. XGBoost with approved past-only behavioral features.
4. Calibration assessment.
5. `tau_tx` validation sweep.
6. 3,600s rolling z-score Layer 2.
7. EWMA.
8. CUSUM.
9. Isolation Forest comparator.
10. Cost sensitivity.
11. Freeze all parameters.
12. Final untouched test replay.
13. Final dashboard/report.

Do not jump directly to Isolation Forest or advanced models.

---

# 15. Agent Execution Protocol

The AI coding agent should work phase-by-phase.

## Before each phase

The agent must:
1. read this blueprint;
2. read the canonical Technical Specification v1.0 if available;
3. identify the current phase;
4. list the files it will create/modify;
5. state the phase gate that must pass.

## During a phase

The agent must:
- implement only modules required by the current phase;
- add unit tests at the same time;
- never bypass a failing leakage test;
- log artifacts and manifests;
- avoid using notebooks as production logic.

## After a phase

The agent must provide:
- completed tasks;
- generated artifacts;
- tests run;
- test results;
- unresolved defects;
- explicit PASS/FAIL for the phase gate.

The agent must not continue to the next phase if the current gate fails.

---

# 16. Stop Conditions

Stop implementation and report instead of guessing if any of the following occurs:

- expected IEEE-CIS key columns are missing;
- joined row count changes unexpectedly;
- `TransactionDT` cannot be ordered reliably;
- duplicate `TransactionID` behavior contradicts assumptions;
- native XGBoost categorical support is unavailable/incompatible in the chosen environment;
- chronological feature tests fail;
- labels become required in the replay scoring path;
- the final test partition was accidentally inspected for tuning;
- an architecture decision would violate the two-layer contract.

If native XGBoost categoricals are technically unavailable, the permitted fallback is:
- train-only frequency encoding for high-cardinality categoricals;
- one-hot for low-cardinality categoricals;
- document the deviation before proceeding.

---

# 17. Final v1 Acceptance Criteria

A v1 implementation is complete only when all are true:

- [ ] IEEE-CIS transaction/identity data joins correctly.
- [ ] Data is sorted by `TransactionDT`.
- [ ] 70/15/15 chronological partitions are frozen.
- [ ] Timestamp ties do not leak.
- [ ] `isFraud` is not a predictive feature.
- [ ] `TransactionID` is not a predictive feature.
- [ ] Train-only numerical imputation works.
- [ ] Categorical `Unknown` handling is stable.
- [ ] Past-only entity features pass future-deletion tests.
- [ ] Expanding CV is chronological.
- [ ] Simple supervised baseline exists.
- [ ] XGBoost outputs fraud probabilities.
- [ ] Calibration is evaluated and decision documented.
- [ ] Recall@Precision 50/70/80/90 is reported where attainable.
- [ ] `tau_tx` is validation-selected.
- [ ] 3,600-second tumbling aggregation works.
- [ ] Empty windows are preserved safely.
- [ ] Rolling baseline excludes current window.
- [ ] Full-history warm-up is enforced.
- [ ] Z-score detector works.
- [ ] EWMA and CUSUM comparisons exist.
- [ ] Isolation Forest compact comparator exists.
- [ ] Replay is deterministic.
- [ ] Proxy spike truth is generated offline only.
- [ ] Q10 training-volume eligibility is frozen.
- [ ] 95/97.5/99 truth sensitivity is reported.
- [ ] Strict during-event detection is the headline event metric.
- [ ] One-window-late metric is secondary.
- [ ] Cost framework is parameterized.
- [ ] NORMAL/ELEVATED/HIGH alert states work.
- [ ] Alerts remain defense-only.
- [ ] Final 15% was not used for tuning.
- [ ] Layer-1 final model remains trained on first 70%.
- [ ] Final transaction metrics are reported.
- [ ] Final spike-event metrics and latency are reported.
- [ ] Streamlit dashboard reads frozen artifacts only.
- [ ] Final run manifest is reproducible.

---

# 18. Canonical End-to-End Flow

```text
IEEE-CIS RAW DATA
        |
        v
INGEST + AUDIT
        |
        v
LEFT JOIN IDENTITY
        |
        v
SORT BY TransactionDT
        |
        v
FREEZE 70 / 15 / 15
        |
        v
TRAIN-ONLY PREPROCESSING
        |
        v
PAST-ONLY FEATURE ENGINE
        |
        v
EXPANDING TEMPORAL CV
        |
        v
XGBOOST LAYER 1
        |
        v
OPTIONAL CALIBRATION
        |
        v
fraud_probability
        |
        v
tau_tx
        |
        v
suspicious_flag
        |
        v
3,600s TUMBLING AGGREGATION
        |
        v
PAST-ONLY ROLLING BASELINE
        |
        +-------------------------+
        |            |            |
        v            v            v
     Z-SCORE        EWMA         CUSUM
        |
        +-------------------------+
        |
        +------ Isolation Forest comparator
        |
        v
VALIDATION-SELECTED SPIKE POLICY
        |
        v
NORMAL / ELEVATED / HIGH
        |
        v
STRUCTURED ALERT
        |
        v
OFFLINE LABEL REVEAL
        |
        v
TRANSACTION + SPIKE EVALUATION
        |
        v
PARAMETERIZED COST ANALYSIS
        |
        v
STREAMLIT DASHBOARD
```

---

# 19. First Build Action for the AI Agent

When implementation begins, do **not** start with XGBoost.

Start with **Phase 1 only**:

1. initialize repository structure;
2. create configuration skeletons;
3. implement dataset ingestion;
4. implement data integrity checks;
5. implement left join;
6. implement schema/type audit;
7. sort chronologically;
8. freeze split manifest;
9. write Phase-1 unit/leakage tests;
10. stop and report Phase-1 gate status.

Only after Phase 1 passes should Phase 2 begin.

---

# 20. Version Control of Decisions

**Blueprint version:** 1.0

The following formerly open decisions are now locked:
- tumbling windows;
- fixed relative-time-grid anchor;
- empty-window emission with no detector update;
- native XGBoost categorical handling;
- conditional calibration;
- multi-point Recall@Precision reporting;
- single v1 high-risk threshold equal to `tau_tx`;
- no 85% final refit;
- full-baseline warm-up;
- Q10 minimum-volume eligibility;
- strict during-event headline matching;
- suspicious-rate primary change-detector input;
- compact Isolation Forest features;
- observable dashboard drivers before SHAP;
- Streamlit dashboard.

Any future change to these requires:
1. a written reason;
2. a new blueprint version;
3. updated tests;
4. explicit comparison against v1 behavior.
