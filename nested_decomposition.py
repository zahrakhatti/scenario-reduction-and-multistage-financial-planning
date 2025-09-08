import numpy as np
import pulp
from itertools import product

class NestedDecomposition:
    def __init__(self, initial_wealth=55, target_wealth=80, penalty_shortage=4, reward_excess=1):
        self.initial_wealth = initial_wealth
        self.target_wealth = target_wealth
        self.penalty_shortage = penalty_shortage
        self.reward_excess = reward_excess
        self.tolerance = 1e-6
        self.max_iterations = 50
        
    def generate_discrete_scenario_tree(self):
        """Generate 8-scenario discrete tree for 4 stages"""
        # Define discrete return values (binary: high/low)
        stock_returns = {'high': 0.15, 'low': 0.08}
        bond_returns = {'high': 0.07, 'low': 0.03}
        
        # Generate all combinations for 3 decision stages (stage 0 is deterministic)
        scenarios = []
        probabilities = []
        
        # All possible combinations for 3 stages
        combinations = list(product(['high', 'low'], repeat=6))  # 3 stages × 2 assets
        
        # Take first 8 combinations to get exactly 8 scenarios
        for i, combo in enumerate(combinations[:8]):
            scenario = []
            for j in range(0, len(combo), 2):
                stock_ret = stock_returns[combo[j]]
                bond_ret = stock_returns[combo[j+1]] if j+1 < len(combo) else bond_returns[combo[j+1]]
                scenario.extend([stock_ret, bond_ret])
            
            scenarios.append(scenario)
            probabilities.append(1.0 / 8)  # Equal probability
        
        return np.array(scenarios), np.array(probabilities)
    
    def solve_master_problem(self, cuts):
        """Solve master problem with current cuts"""
        prob = pulp.LpProblem("MasterProblem", pulp.LpMaximize)
        
        # First-stage variables
        x1_stock = pulp.LpVariable("x1_stock", lowBound=0)
        x1_bond = pulp.LpVariable("x1_bond", lowBound=0)
        theta = pulp.LpVariable("theta", lowBound=-1000)  # Future value function
        
        # Objective: maximize future value
        prob += theta
        
        # First stage budget constraint
        prob += x1_stock + x1_bond == self.initial_wealth
        
        # Add optimality cuts
        for cut in cuts:
            prob += theta <= cut['intercept'] + cut['slope_stock'] * x1_stock + cut['slope_bond'] * x1_bond
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return {
                'x1_stock': pulp.value(x1_stock),
                'x1_bond': pulp.value(x1_bond),
                'theta': pulp.value(theta),
                'objective': pulp.value(prob.objective)
            }
        return None
    
    def solve_subproblem(self, x1_stock, x1_bond, scenario):
        """Solve subproblem for given first-stage solution and scenario"""
        prob = pulp.LpProblem("Subproblem", pulp.LpMaximize)
        
        n_stages = len(scenario) // 2 + 1  # +1 for first stage
        
        # Variables for stages 2 onwards
        x = {}  # x[stage][asset]
        w = {}  # wealth at each stage
        
        # Initialize variables
        for t in range(1, n_stages):
            x[t] = {}
            x[t]['stock'] = pulp.LpVariable(f"x{t+1}_stock", lowBound=0)
            x[t]['bond'] = pulp.LpVariable(f"x{t+1}_bond", lowBound=0)
            w[t] = pulp.LpVariable(f"w{t+1}", lowBound=0)
        
        # Final stage variables
        shortage = pulp.LpVariable("shortage", lowBound=0)
        excess = pulp.LpVariable("excess", lowBound=0)
        
        # Objective: maximize final utility
        prob += excess * self.reward_excess - shortage * self.penalty_shortage
        
        # Stage 2 constraints
        stock_return_1 = 1 + scenario[0]
        bond_return_1 = 1 + scenario[1]
        w1 = x1_stock * stock_return_1 + x1_bond * bond_return_1
        
        if n_stages > 2:
            prob += w[1] == w1
            prob += x[1]['stock'] + x[1]['bond'] == w[1]
        
        # Subsequent stage constraints
        for t in range(2, n_stages - 1):
            if t-1 < len(scenario) // 2:
                stock_return = 1 + scenario[2*(t-1)]
                bond_return = 1 + scenario[2*(t-1)+1]
                
                wealth_from_prev = x[t-1]['stock'] * stock_return + x[t-1]['bond'] * bond_return
                prob += w[t] == wealth_from_prev
                prob += x[t]['stock'] + x[t]['bond'] == w[t]
        
        # Final stage constraint
        final_stage_idx = n_stages - 2
        if final_stage_idx > 0 and 2*final_stage_idx-2 >= 0:
            stock_return_final = 1 + scenario[2*final_stage_idx-2]
            bond_return_final = 1 + scenario[2*final_stage_idx-1]
            
            if final_stage_idx in x:
                final_wealth = x[final_stage_idx]['stock'] * stock_return_final + x[final_stage_idx]['bond'] * bond_return_final
            else:
                final_wealth = w1
        else:
            final_wealth = w1
        
        # Target wealth constraint
        prob += final_wealth + shortage - excess == self.target_wealth
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            # Calculate dual values (simplified approach)
            obj_value = pulp.value(prob.objective)
            return {
                'objective': obj_value,
                'cut_intercept': obj_value,
                'cut_slope_stock': 0,  # Simplified - would need dual values
                'cut_slope_bond': 0
            }
        
        return None
    
    def solve_l_shaped(self):
        """Main L-shaped decomposition algorithm"""
        scenarios, probabilities = self.generate_discrete_scenario_tree()
        
        cuts = []
        iteration = 0
        
        print("Starting L-shaped decomposition...")
        
        while iteration < self.max_iterations:
            print(f"Iteration {iteration + 1}")
            
            # Solve master problem
            master_solution = self.solve_master_problem(cuts)
            if not master_solution:
                print("Master problem infeasible")
                break
            
            print(f"  Master: Stock={master_solution['x1_stock']:.2f}, "
                  f"Bond={master_solution['x1_bond']:.2f}, θ={master_solution['theta']:.2f}")
            
            # Solve subproblems and generate cuts
            new_cuts = []
            total_expected_value = 0
            
            for s, (scenario, prob) in enumerate(zip(scenarios, probabilities)):
                subproblem_solution = self.solve_subproblem(
                    master_solution['x1_stock'], 
                    master_solution['x1_bond'], 
                    scenario
                )
                
                if subproblem_solution:
                    total_expected_value += prob * subproblem_solution['objective']
                    
                    # Generate optimality cut (simplified)
                    new_cut = {
                        'intercept': subproblem_solution['cut_intercept'],
                        'slope_stock': subproblem_solution['cut_slope_stock'],
                        'slope_bond': subproblem_solution['cut_slope_bond']
                    }
                    new_cuts.append(new_cut)
            
            # Check convergence
            gap = abs(master_solution['theta'] - total_expected_value)
            print(f"  Gap: {gap:.6f}")
            
            if gap < self.tolerance:
                print("Converged!")
                break
            
            # Add new cuts (use average cut for simplicity)
            if new_cuts:
                avg_cut = {
                    'intercept': np.mean([cut['intercept'] for cut in new_cuts]),
                    'slope_stock': np.mean([cut['slope_stock'] for cut in new_cuts]),
                    'slope_bond': np.mean([cut['slope_bond'] for cut in new_cuts])
                }
                cuts.append(avg_cut)
            
            iteration += 1
        
        return master_solution, cuts, iteration

def run_nested_decomposition_example():
    """Run nested decomposition example"""
    print("=== Nested Decomposition (L-Shaped) Algorithm ===\n")
    
    # Initialize and solve
    decomp = NestedDecomposition(initial_wealth=55, target_wealth=80)
    
    # Generate scenario tree
    scenarios, probabilities = decomp.generate_discrete_scenario_tree()
    print(f"Generated discrete scenario tree with {len(scenarios)} scenarios")
    print("\nFirst 3 scenarios:")
    for i in range(3):
        print(f"  Scenario {i+1}: {scenarios[i]}, Prob: {probabilities[i]:.3f}")
    
    print("\nSolving with L-shaped decomposition...")
    solution, cuts, iterations = decomp.solve_l_shaped()
    
    if solution:
        print(f"\nSolution found after {iterations} iterations:")
        print(f"  First-stage stock investment: {solution['x1_stock']:.2f}")
        print(f"  First-stage bond investment: {solution['x1_bond']:.2f}")
        print(f"  Expected future value: {solution['theta']:.2f}")
        print(f"  Total cuts generated: {len(cuts)}")
    else:
        print("No solution found")

if __name__ == "__main__":
    run_nested_decomposition_example()