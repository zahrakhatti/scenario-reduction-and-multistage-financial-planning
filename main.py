"""
Main example runner for Stochastic Programming Financial Planning
Demonstrates all methods: Scenario Reduction, Multi-stage Planning, 
Nested Decomposition, and Deterministic Equivalent
"""

def run_all_examples():
    """Run all examples in sequence"""
    
    print("="*80)
    print("STOCHASTIC PROGRAMMING FINANCIAL PLANNING - COMPLETE DEMONSTRATION")
    print("="*80)
    
    print("\nPROBLEM SETUP:")
    print("- Initial wealth: 55 units")
    print("- Target wealth: 80 units")
    print("- Penalty for shortage: 4 units per unit short")
    print("- Reward for excess: 1 unit per unit over")
    print("- Investment options: Stocks and Bonds with uncertain returns")
    
    # Example 1: Scenario Reduction
    print("\n" + "="*80)
    print("EXAMPLE 1: SCENARIO REDUCTION")
    print("="*80)
    
    try:
        from scenario_reduction import generate_financial_scenarios, ScenarioReduction
        
        print("Generating 1000 scenarios from continuous distributions...")
        scenarios, probabilities = generate_financial_scenarios(1000, 4)
        print(f"Generated {len(scenarios)} scenarios with {scenarios.shape[1]} variables")
        
        print("\nApplying Euclidean distance-based reduction...")
        reducer = ScenarioReduction(scenarios, probabilities)
        reduced_scenarios, reduced_probs, selected_idx = reducer.reduce_scenarios(50)
        
        print(f"Reduced to {len(reduced_scenarios)} scenarios")
        print(f"Reduction ratio: {len(reduced_scenarios)/len(scenarios)*100:.1f}%")
        
        print("\nFirst 3 reduced scenarios:")
        for i in range(3):
            print(f"  Scenario {i+1}: Returns {reduced_scenarios[i][:4]}, Prob: {reduced_probs[i]:.4f}")
        
    except Exception as e:
        print(f"Error in scenario reduction: {e}")
    
    # Example 2: Multi-stage with Scenario Reduction
    print("\n" + "="*80)
    print("EXAMPLE 2: MULTI-STAGE PLANNING WITH REDUCED SCENARIOS")
    print("="*80)
    
    try:
        from multistage_financial_planning import MultiStageFinancialPlanning, run_complete_example
        run_complete_example()
        
    except Exception as e:
        print(f"Error in multi-stage planning: {e}")
    
    # Example 3: Nested Decomposition
    print("\n" + "="*80)
    print("EXAMPLE 3: NESTED DECOMPOSITION (L-SHAPED ALGORITHM)")
    print("="*80)
    
    try:
        from nested_decomposition import run_nested_decomposition_example
        run_nested_decomposition_example()
        
    except Exception as e:
        print(f"Error in nested decomposition: {e}")
    
    # Example 4: Deterministic Equivalent
    print("\n" + "="*80)
    print("EXAMPLE 4: DETERMINISTIC EQUIVALENT & METHOD COMPARISON")
    print("="*80)
    
    try:
        from deterministic_equivalent import run_deterministic_equivalent_example, compare_methods
        run_deterministic_equivalent_example()
        print("\n" + "="*60)
        compare_methods()
        
    except Exception as e:
        print(f"Error in deterministic equivalent: {e}")
    


def run_quick_demo():
    """Run a quick demonstration of key concepts"""
    print("QUICK DEMO: Stochastic Programming Financial Planning\n")
    
    # Quick scenario reduction demo
    print("1. Scenario Reduction Demo:")
    try:
        from scenario_reduction import generate_financial_scenarios, ScenarioReduction
        
        scenarios, probs = generate_financial_scenarios(100, 2)  # Smaller for demo
        reducer = ScenarioReduction(scenarios, probs)
        reduced_scenarios, reduced_probs, _ = reducer.reduce_scenarios(10)
        
        print(f"   Reduced {len(scenarios)} scenarios to {len(reduced_scenarios)}")
        print(f"   Original mean return: {scenarios.mean():.4f}")
        print(f"   Reduced mean return: {reduced_scenarios.mean():.4f}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Quick deterministic equivalent demo
    print("\n2. Deterministic Equivalent Demo:")
    try:
        from deterministic_equivalent import DeterministicEquivalent
        
        det_model = DeterministicEquivalent()
        solution, status = det_model.solve_deterministic_equivalent()
        
        if solution:
            print(f"   Status: {status}")
            print(f"   Optimal value: {solution['objective_value']:.4f}")
            print(f"   First-stage: Stock={solution['first_stage']['stock']:.2f}, "
                  f"Bond={solution['first_stage']['bond']:.2f}")
        else:
            print(f"   {status}")
            
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        run_all_examples()