import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog

class ScenarioReduction:
    def __init__(self, scenarios, probabilities):
        """
        scenarios: numpy array of shape (n_scenarios, n_variables)
        probabilities: numpy array of probabilities for each scenario
        """
        self.scenarios = scenarios
        self.probabilities = probabilities
        self.n_scenarios = len(scenarios)
        
    def euclidean_distance_matrix(self):
        """Calculate pairwise Euclidean distances between scenarios"""
        distances = pdist(self.scenarios, metric='euclidean')
        return squareform(distances)
    
    def reduce_scenarios(self, target_scenarios=50):
        """
        Reduce scenarios using fast forward selection method
        """
        distance_matrix = self.euclidean_distance_matrix()
        
        # Initialize with scenario having highest probability
        selected_indices = [np.argmax(self.probabilities)]
        remaining_indices = list(range(self.n_scenarios))
        remaining_indices.remove(selected_indices[0])
        
        # Greedily select scenarios
        while len(selected_indices) < target_scenarios and remaining_indices:
            best_candidate = None
            min_max_distance = float('inf')
            
            for candidate in remaining_indices:
                # Find minimum distance from candidate to any selected scenario
                min_distance = min([distance_matrix[candidate][selected] 
                                  for selected in selected_indices])
                
                # Select candidate that maximizes the minimum distance
                if min_distance < min_max_distance:
                    min_max_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        # Redistribute probabilities
        reduced_scenarios = self.scenarios[selected_indices]
        reduced_probabilities = self._redistribute_probabilities(selected_indices, distance_matrix)
        
        return reduced_scenarios, reduced_probabilities, selected_indices
    
    def _redistribute_probabilities(self, selected_indices, distance_matrix):
        """Redistribute probabilities to selected scenarios"""
        reduced_probs = np.zeros(len(selected_indices))
        
        for i in range(self.n_scenarios):
            if i in selected_indices:
                # Keep original probability
                idx = selected_indices.index(i)
                reduced_probs[idx] += self.probabilities[i]
            else:
                # Find closest selected scenario
                distances_to_selected = [distance_matrix[i][j] for j in selected_indices]
                closest_selected_idx = np.argmin(distances_to_selected)
                reduced_probs[closest_selected_idx] += self.probabilities[i]
        
        return reduced_probs

# Example usage and scenario generation
def generate_financial_scenarios(n_scenarios=1000, n_stages=4):
    """Generate scenarios for stock and bond returns"""
    np.random.seed(42)
    
    # Parameters for normal distributions
    stock_mean, stock_std = 0.12, 0.18  # 12% mean return, 18% volatility
    bond_mean, bond_std = 0.05, 0.08    # 5% mean return, 8% volatility
    
    scenarios = []
    
    for stage in range(n_stages):
        stock_returns = np.random.normal(stock_mean, stock_std, n_scenarios)
        bond_returns = np.random.normal(bond_mean, bond_std, n_scenarios)
        
        if stage == 0:
            scenarios = np.column_stack([stock_returns, bond_returns])
        else:
            stage_scenarios = np.column_stack([stock_returns, bond_returns])
            scenarios = np.column_stack([scenarios, stage_scenarios])
    
    probabilities = np.ones(n_scenarios) / n_scenarios
    
    return scenarios, probabilities

if __name__ == "__main__":
    # Generate scenarios
    scenarios, probabilities = generate_financial_scenarios(1000, 4)
    print(f"Generated {len(scenarios)} scenarios with {scenarios.shape[1]} variables")
    
    # Reduce scenarios
    reducer = ScenarioReduction(scenarios, probabilities)
    reduced_scenarios, reduced_probs, selected_idx = reducer.reduce_scenarios(50)
    
    print(f"Reduced to {len(reduced_scenarios)} scenarios")
    print(f"Original probability sum: {probabilities.sum():.4f}")
    print(f"Reduced probability sum: {reduced_probs.sum():.4f}")
    
    # Display first few reduced scenarios
    print("\nFirst 5 reduced scenarios:")
    for i in range(min(5, len(reduced_scenarios))):
        print(f"Scenario {i+1}: {reduced_scenarios[i]}, Prob: {reduced_probs[i]:.4f}")