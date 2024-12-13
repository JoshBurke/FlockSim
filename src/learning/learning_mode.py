import numpy as np
from typing import List, Type, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime

from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence
from ..core.headless_simulation import HeadlessSimulation

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    best_weights: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary for saving."""
        return {
            'generation': self.generation,
            'avg_fitness': self.avg_fitness,
            'max_fitness': self.max_fitness,
            'min_fitness': self.min_fitness,
            'best_weights': self.best_weights
        }

class LearningMode:
    """Manages the evolution of bot intelligences."""
    
    def __init__(self,
                 scenario_class: Type[Scenario],
                 intelligence_class: Type[Intelligence],
                 population_size: int = 50,
                 generation_frames: int = 500,
                 mutation_rate: float = 0.1,
                 mutation_range: float = 0.2,
                 elite_percentage: float = 0.1,
                 tournament_size: int = 5):
        """Initialize learning mode.
        
        Args:
            scenario_class: Class to use for creating scenarios
            intelligence_class: Class to use for bot intelligence
            population_size: Number of individuals in each generation
            generation_frames: Number of frames to simulate per generation
            mutation_rate: Probability of mutating each weight
            mutation_range: Range of mutation effect
            elite_percentage: Percentage of top performers to keep unchanged
            tournament_size: Number of individuals in each tournament selection
        """
        self.scenario_class = scenario_class
        self.intelligence_class = intelligence_class
        self.population_size = population_size
        self.generation_frames = generation_frames
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range
        self.elite_size = max(1, int(population_size * elite_percentage))
        self.tournament_size = tournament_size
        
        self.generation = 0
        self.stats_history: List[GenerationStats] = []
        
        # Create initial population
        self.population = self._create_initial_population()
    
    def _create_initial_population(self) -> List[Intelligence]:
        """Create initial population with random weights."""
        population = []
        # Create one individual to get weight keys
        template = self.intelligence_class()
        weight_keys = template.weights.keys()
        
        # Create population with random weights
        for _ in range(self.population_size):
            individual = self.intelligence_class()
            for key in weight_keys:
                # Initialize with random values around the default
                base_value = template.weights[key]
                individual.weights[key] = base_value * (1 + np.random.uniform(-0.5, 0.5))
            population.append(individual)
        
        return population
    
    def _tournament_select(self, fitnesses: List[float]) -> Intelligence:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(self.population), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return self.population[winner_idx]
    
    def evolve_generation(self) -> GenerationStats:
        """Evolve one generation and return statistics."""
        # Create scenario and simulation
        scenario = self.scenario_class()
        
        # Evaluate current population
        all_fitnesses = []
        for individual in self.population:
            sim = HeadlessSimulation(scenario, lambda: individual)
            fitnesses = sim.run(self.generation_frames)
            all_fitnesses.append(np.mean(fitnesses))  # Use mean fitness of all bots
        
        # Get statistics
        stats = GenerationStats(
            generation=self.generation,
            avg_fitness=np.mean(all_fitnesses),
            max_fitness=np.max(all_fitnesses),
            min_fitness=np.min(all_fitnesses),
            best_weights=self.population[np.argmax(all_fitnesses)].get_weights()
        )
        self.stats_history.append(stats)
        
        # Sort population by fitness
        sorted_indices = np.argsort(all_fitnesses)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        
        # Create next generation
        next_population = []
        
        # Keep elite individuals
        next_population.extend(self.population[:self.elite_size])
        
        # Fill rest with offspring
        while len(next_population) < self.population_size:
            # Select parents
            parent1 = self._tournament_select(all_fitnesses)
            parent2 = self._tournament_select(all_fitnesses)
            
            # Create offspring
            child = self.intelligence_class()
            
            # Crossover
            child_weights = parent1.crossover(parent2)
            child.set_weights(child_weights)
            
            # Mutation
            child.mutate(self.mutation_rate, self.mutation_range)
            
            next_population.append(child)
        
        self.population = next_population
        self.generation += 1
        
        return stats
    
    def run_evolution(self, num_generations: int, save_dir: Optional[str] = None) -> List[GenerationStats]:
        """Run evolution for specified number of generations.
        
        Args:
            num_generations: Number of generations to evolve
            save_dir: Directory to save progress (None for no saving)
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for gen in range(num_generations):
            stats = self.evolve_generation()
            print(f"Generation {stats.generation}: "
                  f"Avg={stats.avg_fitness:.3f}, "
                  f"Max={stats.max_fitness:.3f}, "
                  f"Min={stats.min_fitness:.3f}")
            
            if save_dir:
                # Save generation stats
                stats_file = os.path.join(save_dir, f"gen_{gen:04d}_stats.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats.to_dict(), f, indent=2)
                
                # Save best weights
                weights_file = os.path.join(save_dir, f"gen_{gen:04d}_best_weights.json")
                with open(weights_file, 'w') as f:
                    json.dump(stats.best_weights, f, indent=2)
        
        return self.stats_history
    
    def load_weights(self, weights_file: str) -> None:
        """Load weights from a file into the population."""
        with open(weights_file, 'r') as f:
            weights = json.load(f)
        
        # Apply weights to all individuals
        for individual in self.population:
            individual.set_weights(weights) 