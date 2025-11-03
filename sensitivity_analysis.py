"""
SOBOL SENSITIVITY ANALYSIS WITH MLFLOW
======================================
Analyzes how model parameters affect adoption outcomes
Features checkpointing for resuming after interruptions
"""

import numpy as np
import mlflow
import mlflow.sklearn
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from simple_contagion_model import SimpleContagion
import pickle
import os
from pathlib import Path


CHECKPOINT_FILE = '/mnt/user-data/outputs/sensitivity_checkpoint.pkl'


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


def save_checkpoint(param_values, outputs, completed_idx):
    """Save progress to disk."""
    checkpoint = {
        'param_values': param_values,
        'outputs': outputs,
        'completed_idx': completed_idx
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint():
    """Load previous progress if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def sobol_analysis(save_interval=50):
    """Perform Sobol sensitivity analysis with MLFlow tracking and checkpointing."""
    
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
    
    # Try to load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        print(f"Found checkpoint! Resuming from sample {checkpoint['completed_idx']}/{len(param_values)}")
        outputs = checkpoint['outputs']
        start_idx = checkpoint['completed_idx']
    else:
        print(f"Starting fresh: {len(param_values)} simulations...")
        outputs = np.zeros(len(param_values))
        start_idx = 0
    
    # Run model for remaining samples
    with mlflow.start_run(run_name="sobol_sensitivity"):
        mlflow.log_param("analysis_type", "sobol")
        mlflow.log_param("n_samples", len(param_values))
        mlflow.log_param("checkpoint_interval", save_interval)
        
        try:
            for i in range(start_idx, len(param_values)):
                outputs[i] = run_model_sample(param_values[i])
                
                # Save checkpoint periodically
                if (i + 1) % save_interval == 0:
                    save_checkpoint(param_values, outputs, i + 1)
                    print(f"Progress: {i + 1}/{len(param_values)} ({100*(i+1)/len(param_values):.1f}%) - Checkpoint saved")
            
            # Final checkpoint
            save_checkpoint(param_values, outputs, len(param_values))
            print(f"Completed: {len(param_values)}/{len(param_values)} (100%)")
            
        except KeyboardInterrupt:
            print(f"\nInterrupted! Progress saved at sample {i}")
            save_checkpoint(param_values, outputs, i)
            return
        except Exception as e:
            print(f"\nError at sample {i}: {e}")
            save_checkpoint(param_values, outputs, i)
            raise
        
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
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint file cleaned up")


if __name__ == '__main__':
    # Set MLFlow tracking
    mlflow.set_experiment("contagion_sensitivity")
    sobol_analysis(save_interval=50)  # Save every 50 simulations
