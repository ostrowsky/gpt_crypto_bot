# V2 Belief Filter V1

Status: research-only  
Last updated: 2026-05-18

## Purpose

Move from isolated per-bar reconstruction to live-like belief trajectories over symbol
lifecycle states.

## Inputs

- the v2 state graph;
- causal reconstruction features;
- train-only scaled nearest-centroid distances as emission evidence;
- chronological OOS split from the state-reconstruction baseline.

## V1 Filter

At each closed bar:

1. predict prior belief through the allowed transition graph;
2. compute emission likelihoods from distance to train centroids;
3. multiply prior by emission likelihood;
4. normalize;
5. emit the most likely state and full belief vector.

## Transition Prior

- allowed transitions only;
- self-transition bias `0.70`;
- remaining probability spread uniformly across other allowed outgoing edges.

## Emission Prior

- inverse-softmax over centroid distance;
- temperature is reported and versioned;
- no test-window fitting.

## Reset Rule

- beliefs reset at symbol / local-day boundary;
- this keeps the first baseline aligned with teacher-day construction.

## Metrics

- weighted accuracy;
- macro F1;
- per-state recall;
- emerging-move recall;
- comparison vs isolated nearest-centroid baseline.

## Acceptance Criteria

1. Filter uses no future information.
2. Belief vectors remain normalized.
3. OOS report compares against isolated centroid baseline.
4. If filtering degrades early-state recall, report that honestly instead of promoting it.

