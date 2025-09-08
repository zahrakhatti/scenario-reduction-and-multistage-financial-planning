import numpy as np
import pulp
from itertools import product

class DeterministicEquivalent:
    def __init__(self, initial_wealth=55, target_wealth=80, penalty_shortage=4, reward_excess=1):
        self.initial_wealth = initial_wealth
        self.target_wealth = target_wealth
        self.penalty_shortage = penalty_shortage
        self.reward_excess = reward_excess
    
    def generate_discrete_scenarios(self):
        """Generate the same 8-scenario tree as nested decomposition"""
        stock_returns = {'high': 0.15, 'low': 0.08}
        bond_returns = {'high': 0.07, 'low': 0.03}
        
        scenarios = []
        probabilities = []
        
        combinations = list(product(['high', 'low'], repeat=6))[:8]
        
        for combo in combinations:
            scenario = []
            for j in range(0, len(combo), 2):
                stock_ret = stock_returns[combo[j]]
                bond_ret = bond_returns[combo[j+1]] if j+1 < len(combo) else bond_returns[combo[j+1]]
                scenario.extend([stock_ret, bond_ret])
            
            scenarios.append(scenario)
            probabilities.append(1.0 / 8)
        
        return np.array(scenarios), np.array(probabilities)
    
    def solve_deterministic_equivalent(self):
        """Solve the full deterministic equivalent formulation"""
        scenarios, probabilities = self.generate_discrete_scenarios()
        n_scenarios = len(scenarios)
        n_stages = len(scenarios[0]) // 2 + 1  # +1 for first stage
        
        prob = pulp.LpProblem("DeterministicEquivalent", pulp.LpMaximize)
        
        # Decision variables
        # First stage (scenario-independent)
        x1_stock = pulp.LpVariable("x1_stock", lowBound=0)
        x1_bond = pulp.LpVariable("x1_bond", lowBound=0)
        
        # Subsequent stages (scenario-dependent)
        x = {}  # x[scenario][stage][asset]
        w = {}  # w[scenario][stage] - wealth
        shortage = {}
        excess = {}
        
        for s in range(n_scenarios):
            x[s] = {}
            w[s] = {}
            
            for t in range(1, n_stages):
                x[s][t] = {}
                x[s][t]['stock'] = pulp.LpVariable(f"x_s{s}_t{t}_stock", lowBound=0)
                x[s][t]['bond'] = pulp.LpVariable(f"x_s{s}_t{t}_bond", lowBound=0)
                w[s][t] = pulp.LpVariable(f"w_s{s}_t{t}", lowBound=0)
            
            shortage[s] = pulp.LpVariable(f"shortage_s{s}", lowBound=0)
            excess[s] = pulp.LpVariable(f"excess_s{s}", lowBound=0)
        
        # Objective function
        expected_utility = 0
        for s in range(n_scenarios):
            expected_utility += probabilities[s] * (
                excess[s] * self.reward_excess - shortage[s] * self.penalty_shortage
            )
        
        prob += expected_utility
        
        # Constraints
        # First stage budget constraint (scenario-independent)
        prob += x1_stock + x1_bond == self.initial_wealth
        
        # Scenario-dependent constraints
        for s in range(n_scenarios):
            scenario = scenarios[s]
            
            # Stage 2 constraints
            stock_return_1 = 1 + scenario[0]
            bond_return_1 = 1 + scenario[1]
            
            # Wealth from first stage
            w1 = x1_stock * stock_return_1 + x1_bond * bond_return_1
            prob += w[s][1] == w1
            
            # Budget constraint for stage 2
            if n_stages > 2:
                prob += x[s][1]['stock'] + x[s][1]['bond'] == w[s][1]
            
            # Subsequent stages
            for t in range(2, n_stages):
                if 2*(t-1) < len(scenario):
                    stock_return = 1 + scenario[2*(t-1)]
                    bond_return = 1 + scenario[2*(t-1)+1]
                    
                    # Wealth evolution
                    wealth_from_prev = (x[s][t-1]['stock'] * stock_return + 
                                      x[s][t-1]['bond'] * bond_return)
                    prob += w[s][t] == wealth_from_prev
                    
                    # Budget constraint (if not final stage)
                    if t < n_stages - 1:
                        prob += x[s][t]['stock'] + x[s][t]['bond'] == w[s][t]
            
            # Final stage constraint
            final_stage = n_stages - 1
            if final_stage > 1:
                final_wealth = w[s][final_stage]
            else:
                final_wealth = w[s][1]
            
            # Target wealth constraint
            prob += final_wealth + shortage[s] - excess[s] == self.target_wealth
        
        # Solve
        print("Solving deterministic equivalent...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return self._extract_solution(prob, x1_stock, x1_bond, x, w, shortage, excess, 
                                    scenarios, probabilities, n_stages)
    
    def _extract_solution(self, prob, x1_stock, x1_bond, x, w, shortage, excess, 
                         scenarios, probabilities, n_stages):
        """Extract and format solution"""
        if prob.status != pulp.LpStatusOptimal:
            return None, f"Problem status: {pulp.LpStatus[prob.status]}"
        
        solution = {
            'objective_value': pulp.value(prob.objective),
            'first_stage': {
                'stock': pulp.value(x1_stock),
                'bond': pulp.value(x1_bond)
            },
            'scenarios': []
        }
        
        # Extract scenario solutions (show first 3)
        for s in range(min(3, len(scenarios))):
            scenario_sol = {
                'scenario': s,
                'returns': scenarios[s].tolist(),
                'probability': probabilities[s],
                'shortage': pulp.value(shortage[s]),
                'excess': pulp.value(excess[s]),
                'stages': []
            }
            
            for t in range(1, n_stages):
                if t in x[s]:
                    stage_sol = {
                        'stage': t,
                        'wealth': pulp.value(w[s][t]) if w[s][t] else 0,
                        'stock_investment': pulp.value(x[s][t]['stock']) if x[s][t]['stock'] else 0,
                        'bond_investment': pulp.value(x[s][t]['bond']) if x[s][t]['bond'] else 0
                    }
                else:
                    stage_sol = {
                        'stage': t,
                        'wealth': pulp.value(w[s][t]) if w[s][t] else 0,
                        'stock_investment': 0,
                        'bond_investment': 0
                    }
                scenario_sol['stages'].append(stage_sol)
            
            solution['scenarios'].append(scenario_sol)
        
        return solution, "Optimal solution found"

def compare_methods():
    """Compare deterministic equivalent with nested decomposition"""
    print("=== Comparison: Deterministic Equivalent vs Nested Decomposition ===\n")
    
    # Solve with deterministic equivalent
    det_equiv = DeterministicEquivalent(initial_wealth=55, target_wealth=80)
    det_solution, det_status = det_equiv.solve_deterministic_equivalent()
    
    print("1. Deterministic Equivalent Method:")
    if det_solution:
        print(f"   Status: {det_status}")
        print(f"   Objective value: {det_solution['objective_value']:.4f}")
        print(f"   First stage - Stock: {det_solution['first_stage']['stock']:.2f}, "
              f"Bond: {det_solution['first_stage']['bond']:.2f}")
        
        print("\n   Sample scenario solutions:")
        for scenario_sol in det_solution['scenarios']:
            s = scenario_sol['scenario']
            print(f"     Scenario {s+1}: Shortage={scenario_sol['shortage']:.2f}, "
                  f"Excess={scenario_sol['excess']:.2f}")
    else:
        print(f"   Error: {det_status}")
    
    # Solve with nested decomposition
    print("\n2. Nested Decomposition Method:")
    from nested_decomposition import NestedDecomposition
    
    decomp = NestedDecomposition(initial_wealth=55, target_wealth=80)
    nested_solution, cuts, iterations = decomp.solve_l_shaped()
    
    if nested_solution:
        print(f"   Converged after {iterations} iterations")
        print(f"   First stage - Stock: {nested_solution['x1_stock']:.2f}, "
              f"Bond: {nested_solution['x1_bond']:.2f}")
        print(f"   Expected future value: {nested_solution['theta']:.2f}")
        print(f"   Number of cuts generated: {len(cuts)}")
    else:
        print("   No solution found")
    
    # Compare results
    if det_solution and nested_solution:
        print("\n3. Comparison:")
        stock_diff = abs(det_solution['first_stage']['stock'] - nested_solution['x1_stock'])
        bond_diff = abs(det_solution['first_stage']['bond'] - nested_solution['x1_bond'])
        
        print(f"   First-stage stock difference: {stock_diff:.4f}")
        print(f"   First-stage bond difference: {bond_diff:.4f}")
        print(f"   Methods agree: {stock_diff < 0.01 and bond_diff < 0.01}")

def run_deterministic_equivalent_example():
    """Run standalone deterministic equivalent example"""
    print("=== Deterministic Equivalent Formulation ===\n")
    
    det_equiv = DeterministicEquivalent(initial_wealth=55, target_wealth=80)
    
    # Generate scenarios
    scenarios, probabilities = det_equiv.generate_discrete_scenarios()
    print(f"Generated {len(scenarios)} discrete scenarios")
    
    print("\nScenario tree:")
    for i, (scenario, prob) in enumerate(zip(scenarios, probabilities)):
        print(f"  Scenario {i+1}: Returns {scenario}, Probability: {prob:.3f}")
    
    # Solve
    solution, status = det_equiv.solve_deterministic_equivalent()
    
    if solution:
        print(f"\n{status}")
        print(f"Optimal expected utility: {solution['objective_value']:.4f}")
        print(f"\nFirst-stage decisions:")
        print(f"  Stock investment: {solution['first_stage']['stock']:.2f}")
        print(f"  Bond investment: {solution['first_stage']['bond']:.2f}")
        
        print(f"\nDetailed scenario results:")
        for scenario_sol in solution['scenarios']:
            s = scenario_sol['scenario']
            print(f"\n  Scenario {s+1} (prob: {scenario_sol['probability']:.3f}):")
            print(f"    Returns: {scenario_sol['returns']}")
            print(f"    Final shortage: {scenario_sol['shortage']:.2f}")
            print(f"    Final excess: {scenario_sol['excess']:.2f}")
            
            for stage_sol in scenario_sol['stages']:
                t = stage_sol['stage']
                print(f"    Stage {t+1}: Wealth={stage_sol['wealth']:.2f}, "
                      f"Stock={stage_sol['stock_investment']:.2f}, "
                      f"Bond={stage_sol['bond_investment']:.2f}")
    else:
        print(f"Error: {status}")

if __name__ == "__main__":
    run_deterministic_equivalent_example()
    print("\n" + "="*60 + "\n")
    compare_methods()