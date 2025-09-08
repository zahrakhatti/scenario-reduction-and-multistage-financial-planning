import numpy as np
from scipy.optimize import linprog
import pulp

class MultiStageFinancialPlanning:
    def __init__(self, initial_wealth=55, target_wealth=80, penalty_shortage=4, reward_excess=1):
        self.initial_wealth = initial_wealth
        self.target_wealth = target_wealth
        self.penalty_shortage = penalty_shortage
        self.reward_excess = reward_excess
        
    def solve_with_reduced_scenarios(self, reduced_scenarios, reduced_probabilities):
        """
        Solve multi-stage financial planning with reduced scenarios
        """
        n_scenarios = len(reduced_scenarios)
        n_stages = reduced_scenarios.shape[1] // 2  # 2 assets per stage
        
        # Create optimization problem
        prob = pulp.LpProblem("FinancialPlanning", pulp.LpMaximize)
        
        # Decision variables
        # x[s][t][a] = investment in asset a at stage t for scenario s
        x = {}  # x[scenario][stage][asset] (0=stock, 1=bond)
        w = {}  # w[scenario][stage] = wealth at stage t for scenario s
        shortage = {}  # shortage variables
        excess = {}    # excess variables
        
        # Initialize variables
        for s in range(n_scenarios):
            x[s] = {}
            w[s] = {}
            for t in range(n_stages):
                x[s][t] = {}
                x[s][t][0] = pulp.LpVariable(f"x_stock_{s}_{t}", lowBound=0)
                x[s][t][1] = pulp.LpVariable(f"x_bond_{s}_{t}", lowBound=0)
                w[s][t] = pulp.LpVariable(f"wealth_{s}_{t}", lowBound=0)
            
            shortage[s] = pulp.LpVariable(f"shortage_{s}", lowBound=0)
            excess[s] = pulp.LpVariable(f"excess_{s}", lowBound=0)
        
        # Objective function: maximize expected final utility
        expected_utility = 0
        for s in range(n_scenarios):
            expected_utility += reduced_probabilities[s] * (
                excess[s] * self.reward_excess - shortage[s] * self.penalty_shortage
            )
        
        prob += expected_utility
        
        # Constraints
        for s in range(n_scenarios):
            # Initial wealth constraint
            prob += w[s][0] == self.initial_wealth
            
            # Budget constraints for each stage
            for t in range(n_stages):
                if t == 0:
                    # First stage: invest initial wealth
                    prob += x[s][t][0] + x[s][t][1] == w[s][t]
                else:
                    # Later stages: invest returns from previous stage
                    stock_return = 1 + reduced_scenarios[s][2*(t-1)]
                    bond_return = 1 + reduced_scenarios[s][2*(t-1)+1]
                    
                    # Wealth from previous investments
                    prob += w[s][t] == (x[s][t-1][0] * stock_return + 
                                      x[s][t-1][1] * bond_return)
                    
                    # Investment constraint
                    if t < n_stages - 1:
                        prob += x[s][t][0] + x[s][t][1] == w[s][t]
            
            # Final stage constraint (target wealth)
            final_stage = n_stages - 1
            if final_stage > 0:
                stock_return = 1 + reduced_scenarios[s][2*(final_stage-1)]
                bond_return = 1 + reduced_scenarios[s][2*(final_stage-1)+1]
                
                final_wealth = x[s][final_stage-1][0] * stock_return + x[s][final_stage-1][1] * bond_return
                
                # Target wealth constraint with shortage and excess
                prob += final_wealth + shortage[s] - excess[s] == self.target_wealth
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return self._extract_solution(prob, x, w, shortage, excess, n_scenarios, n_stages)
    
    def _extract_solution(self, prob, x, w, shortage, excess, n_scenarios, n_stages):
        """Extract solution from solved problem"""
        if prob.status != pulp.LpStatusOptimal:
            return None, f"Problem status: {pulp.LpStatus[prob.status]}"
        
        solution = {
            'objective_value': pulp.value(prob.objective),
            'scenarios': []
        }
        
        for s in range(min(5, n_scenarios)):  # Show first 5 scenarios
            scenario_solution = {
                'scenario': s,
                'investments': [],
                'wealth': [],
                'shortage': pulp.value(shortage[s]) if shortage[s] else 0,
                'excess': pulp.value(excess[s]) if excess[s] else 0
            }
            
            for t in range(n_stages):
                if t < n_stages - 1:  # Don't show investments for final stage
                    investments = {
                        'stage': t,
                        'stock': pulp.value(x[s][t][0]) if x[s][t][0] else 0,
                        'bond': pulp.value(x[s][t][1]) if x[s][t][1] else 0
                    }
                    scenario_solution['investments'].append(investments)
                
                wealth_val = pulp.value(w[s][t]) if w[s][t] else 0
                scenario_solution['wealth'].append(wealth_val)
            
            solution['scenarios'].append(scenario_solution)
        
        return solution, "Optimal solution found"

def run_complete_example():
    """Run complete example with scenario generation, reduction, and optimization"""
    from scenario_reduction import generate_financial_scenarios, ScenarioReduction
    
    print("=== Multi-Stage Financial Planning with Scenario Reduction ===\n")
    
    # Step 1: Generate scenarios
    print("1. Generating scenarios...")
    scenarios, probabilities = generate_financial_scenarios(1000, 4)
    print(f"   Generated {len(scenarios)} scenarios")
    
    # Step 2: Reduce scenarios
    print("\n2. Reducing scenarios...")
    reducer = ScenarioReduction(scenarios, probabilities)
    reduced_scenarios, reduced_probs, selected_idx = reducer.reduce_scenarios(50)
    print(f"   Reduced to {len(reduced_scenarios)} scenarios")
    
    # Step 3: Solve optimization problem
    print("\n3. Solving financial planning problem...")
    planner = MultiStageFinancialPlanning(initial_wealth=55, target_wealth=80)
    solution, status = planner.solve_with_reduced_scenarios(reduced_scenarios, reduced_probs)
    
    if solution:
        print(f"   Status: {status}")
        print(f"   Optimal expected utility: {solution['objective_value']:.4f}")
        
        print("\n4. Solution details (first 3 scenarios):")
        for scenario_sol in solution['scenarios'][:3]:
            s = scenario_sol['scenario']
            print(f"\n   Scenario {s+1}:")
            print(f"     Shortage: {scenario_sol['shortage']:.2f}")
            print(f"     Excess: {scenario_sol['excess']:.2f}")
            
            for inv in scenario_sol['investments']:
                t = inv['stage']
                print(f"     Stage {t+1}: Stock={inv['stock']:.2f}, Bond={inv['bond']:.2f}")
    else:
        print(f"   Error: {status}")

if __name__ == "__main__":
    run_complete_example()