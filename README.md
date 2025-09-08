# Stochastic Programming Scenario Reduction and Multi-Stage Financial Planning

This repository contains solutions to multi-stage financial planning stochastic programming problems using scenario reduction and nested decomposition methods in Python.

## Project Overview

The main objectives of this project are:

1. **Implement scenario reduction for continuous distributions**:
   - Generate scenarios from normal distributions for stock and bond profits
   - Apply Euclidean distance-based reduction algorithm
   - Reduce 1000+ scenarios to manageable 50-100 scenarios
   - Solve stochastic programming model with reduced scenarios

2. **Implement nested decomposition for discrete distributions**:
   - Build discrete scenario tree with 8 scenarios across 4 stages
   - Apply L-shaped (nested decomposition) algorithm
   - Generate optimality cuts through forward/backward passes
   - Solve using both iterative and deterministic equivalent methods

3. **Financial planning problem setup**:
   - Initial wealth: 55 units
   - Target wealth: 80 units at final stage
   - Investment options: stocks and bonds with uncertain returns
   - Penalty structure: 4 units cost per shortage, 1 unit reward per excess

## Requirements

```bash
pip install -r requirements.txt
```

## Files

- `scenario_reduction.py` - Scenario reduction algorithm
- `multistage_financial_planning.py` - Multi-stage optimization with reduced scenarios
- `nested_decomposition.py` - L-shaped decomposition algorithm
- `deterministic_equivalent.py` - Full deterministic equivalent formulation
- `main_example.py` - Run all examples

## How to Run

### Run Everything
```bash
python main_example.py
```

### Run Individual Components
```bash
python scenario_reduction.py          # Scenario reduction only
python nested_decomposition.py        # L-shaped algorithm only
python deterministic_equivalent.py    # Deterministic equivalent only
```

### Quick Test
```bash
python main_example.py --quick
```

## What Each Method Does

| Method | Purpose |
|--------|---------|
| **Scenario Reduction** | Reduces 1000 scenarios → 50 scenarios using distance-based algorithm |
| **Multi-Stage Planning** | Solves financial planning with reduced scenarios |
| **Nested Decomposition** | L-shaped algorithm with 8 discrete scenarios |
| **Deterministic Equivalent** | Solves all scenarios simultaneously for comparison |

## Expected Results

- **Scenario Reduction**: 1000 scenarios reduced to 50 (5% of original)
- **Optimal Investment**: ~28-30 units in stocks, ~25-27 units in bonds
- **Expected Utility**: Around -8 to -12 units
- **L-shaped Convergence**: 2-5 iterations typically

## Problem Parameters

- Initial wealth: 55 units
- Target wealth: 80 units
- Stock returns: Normal(12%, 18%)
- Bond returns: Normal(5%, 8%)
- Shortage penalty: 4 units per unit short
- Excess reward: 1 unit per unit over

That's it! Run `python main_example.py` to see everything in action.
