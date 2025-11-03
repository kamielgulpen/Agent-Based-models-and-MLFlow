"""
SOBOL SENSITIVITY ANALYSIS WITH MLFLOW
======================================
Analyzes how model parameters affect adoption outcomes
"""

import numpy as np
import mlflow
import mlflow.sklearn
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from simple_contagion_model import SimpleContagion


def run_model_sample(params):
    """Run model with given parameters and return metrics."""
    model = SimpleContagion(
        n_people=int(params[0]),
        threshold=params[1],
        initial_adopters=int(params[2])
    )
    model.run(max_steps=50)
    results = model.get_results()
    return results['final_adoption_rate']


def sobol_analysis():
    """Perform Sobol sensitivity analysis with MLFlow tracking."""
    
    # Define parameter space
    problem = {
        'num_vars': 3,
        'names': ['n_people', 'threshold', 'initial_adopters'],
        'bounds': [[50, 200],      # network size
                   [0.1, 0.5],      # threshold
                   [3, 15]]          # initial adopters
    }
    
    # Generate samples (Sobol sequence)
    param_values = saltelli.sample(problem, 512)
    
    print(f"Running {len(param_values)} simulations...")
    
    # Run model for each sample
    with mlflow.start_run(run_name="sobol_sensitivity"):
        mlflow.log_param("analysis_type", "sobol")
        mlflow.log_param("n_samples", len(param_values))
        
        outputs = np.array([run_model_sample(params) for params in param_values])
        
        # Perform Sobol analysis
        Si = sobol.analyze(problem, outputs)
        
        # Log sensitivity indices
        for i, name in enumerate(problem['names']):
            mlflow.log_metric(f"{name}_S1", Si['S1'][i])
            mlflow.log_metric(f"{name}_ST", Si['ST'][i])
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # First-order indices
        ax1.bar(problem['names'], Si['S1'])
        ax1.set_ylabel('First-order Index (S1)')
        ax1.set_title('Direct Effect on Adoption')
        ax1.set_ylim([0, 1])
        
        # Total-order indices
        ax2.bar(problem['names'], Si['ST'])
        ax2.set_ylabel('Total-order Index (ST)')
        ax2.set_title('Total Effect (Including Interactions)')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/sensitivity_results.png', dpi=150)
        mlflow.log_artifact('/mnt/user-data/outputs/sensitivity_results.png')
        
        print("\n=== SENSITIVITY ANALYSIS RESULTS ===")
        print("\nFirst-order indices (S1) - Direct effects:")
        for i, name in enumerate(problem['names']):
            print(f"  {name:20s}: {Si['S1'][i]:.3f}")
        
        print("\nTotal-order indices (ST) - Total effects:")
        for i, name in enumerate(problem['names']):
            print(f"  {name:20s}: {Si['ST'][i]:.3f}")
        
        print(f"\nResults logged to MLFlow")
        print(f"View at: mlruns/")


if __name__ == '__main__':
    # Set MLFlow tracking
    mlflow.set_experiment("contagion_sensitivity")
    sobol_analysis()
