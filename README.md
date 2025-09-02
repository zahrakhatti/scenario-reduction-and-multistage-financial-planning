# Stochastic Programming Scenario Reduction and Stochastic Multi Stage financial planning

This repository contains solutions to the multi-stage financial planning stochastic programming problems using scenario reduction and nested decomposition methods in Python.

## Project Overview

The main objectives of this project are:

1. **Implement scenario reduction for continuous distributions**:
   * Generate scenarios from normal distributions for stock and bond profits
   * Apply Euclidean distance-based reduction algorithm
   * Reduce 1000+ scenarios to manageable 50-100 scenarios
   * Solve stochastic programming model with reduced scenarios

2. **Implement nested decomposition for discrete distributions**:
   * Build discrete scenario tree with 8 scenarios across 4 stages
   * Apply L-shaped (nested decomposition) algorithm
   * Generate optimality cuts through forward/backward passes
   * Solve using both iterative and deterministic equivalent methods

3. **Financial planning problem setup**:
   * Initial wealth: 55 units
   * Target wealth: 80 units at final stage
   * Investment options: stocks and bonds with uncertain returns
   * Penalty structure: 4 units cost per shortage, 1 unit reward per excess
