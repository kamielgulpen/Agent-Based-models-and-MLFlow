"""
SIMPLE COMPLEX CONTAGION MODEL
==============================
A beginner-friendly Agent-Based Model

What it does:
- Creates a network of people (agents)
- Some start with a new behavior (adopters)
- Others adopt if enough of their friends have adopted (threshold)
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class SimpleContagion:
    """
    A simple model of complex contagion.
    
    Parameters you can change:
    - n_people: How many people in the network
    - network_type: How people are connected ('random' or 'small_world')
    - threshold: What fraction of friends must adopt before you adopt
    - initial_adopters: How many people start with the behavior
    """
    
    def __init__(self, n_people=100, network_type='small_world', 
                 threshold=0.25, initial_adopters=5):
        self.n_people = n_people
        self.network_type = network_type
        self.threshold = threshold
        self.initial_adopters = initial_adopters
        
        # Create the network
        self._create_network()
        
        # Initialize: 0 = haven't adopted, 1 = have adopted
        self.states = np.zeros(n_people, dtype=int)
        
        # Pick random people to start as adopters
        starters = np.random.choice(n_people, initial_adopters, replace=False)
        self.states[starters] = 1
        
        # Track history
        self.history = [self.states.sum()]
    
    def _create_network(self):
        """Create the social network."""
        if self.network_type == 'random':
            # Random connections
            self.network = nx.erdos_renyi_graph(self.n_people, 0.06)
        elif self.network_type == 'small_world':
            # Small world: local clusters + shortcuts
            self.network = nx.watts_strogatz_graph(self.n_people, 6, 0.1)
        else:
            # Default to small world
            self.network = nx.watts_strogatz_graph(self.n_people, 6, 0.1)
    
    def step(self):
        """Run one time step of the model."""
        new_states = self.states.copy()
        
        # Check each person
        for person in range(self.n_people):
            # Skip if already adopted
            if self.states[person] == 1:
                continue
            
            # Get their friends
            friends = list(self.network.neighbors(person))
            if len(friends) == 0:
                continue
            
            # Count how many friends have adopted
            friends_who_adopted = sum(self.states[friend] for friend in friends)
            fraction = friends_who_adopted / len(friends)
            
            # Adopt if enough friends have adopted
            if fraction >= self.threshold:
                new_states[person] = 1
        
        self.states = new_states
        self.history.append(self.states.sum())
    
    def run(self, max_steps=50):
        """Run the model until it stops changing or max_steps."""
        for _ in range(max_steps):
            before = self.states.sum()
            self.step()
            after = self.states.sum()
            
            # Stop if nothing changed
            if before == after:
                break
        
        return self.history
    
    def get_results(self):
        """Get key results from the simulation."""
        return {
            'final_adopters': self.states.sum(),
            'final_adoption_rate': self.states.sum() / self.n_people,
            'time_to_stop': len(self.history) - 1,
            'total_people': self.n_people
        }
    
    def plot(self):
        """Simple plot of adoption over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history, linewidth=2, marker='o')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Number of Adopters', fontsize=12)
        plt.title(f'Adoption Over Time (Final: {self.states.sum()}/{self.n_people})', 
                  fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


# Simple example usage
if __name__ == '__main__':
    print("Running Simple Contagion Model...\n")
    
    # Create and run the model
    model = SimpleContagion(
        n_people=100,
        network_type='small_world',
        threshold=0.25,
        initial_adopters=5
    )
    
    model.run(max_steps=50)
    results = model.get_results()
    
    # Print results
    print("Results:")
    print(f"  Final adopters: {results['final_adopters']}")
    print(f"  Adoption rate: {results['final_adoption_rate']:.1%}")
    print(f"  Time to stop: {results['time_to_stop']} steps")
    
    # Save plot
    model.plot()
    plt.show()
    print("\nPlot saved!")