import numpy as np
from typing import List, Type, Dict, Tuple, Optional, Callable
import json
import os
import time
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt

from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence
from ..core.headless_simulation import HeadlessSimulation
from ..core.simulation import Simulation
from .visualization import MultiPopulationVisualizer
from .multi_population_stats import MultiPopulationStats

def _evaluate_combination(args) -> Dict[str, float]:
    """Helper function to evaluate a combination of individuals from different populations."""
    try:
        scenario_class, intelligence_instances, generation_frames = args
        
        # Calculate total number of bots and validate
        total_bots = sum(len(instances) for instances in intelligence_instances.values())
        if total_bots == 0:
            raise ValueError("No bots to evaluate")
        
        # Create scenario with appropriate predator ratio based on population sizes
        predator_count = len(intelligence_instances['predator'])
        predator_ratio = predator_count/total_bots
        
        # Create non-verbose scenario for evolution
        scenario = scenario_class(predator_ratio=predator_ratio, verbose=False)
        
        # Create intelligence factories and population indices
        intelligence_factories = {}
        population_indices = {}
        
        # Track which individual is being evaluated
        evaluated_indices = {}
        
        # Create factories for each population
        for pop_name, instances in intelligence_instances.items():
            def create_factory(pop_name=pop_name, instances=instances):
                def factory():
                    # Create a new instance with copied weights
                    template = list(instances.values())[0]
                    new_instance = template.__class__()
                    try:
                        new_instance.set_weights(template.get_weights().copy())
                    except Exception as e:
                        print(f"Error copying weights: {str(e)}")
                        # Use default weights as fallback
                        new_instance = template.__class__()
                    return new_instance
                return factory
            
            intelligence_factories[pop_name] = create_factory(pop_name, instances)
            population_indices[pop_name] = sorted(list(instances.keys()))  # Ensure indices are sorted
            # Store the first index as the one being evaluated
            evaluated_indices[pop_name] = population_indices[pop_name][0]
        
        sim = HeadlessSimulation(scenario, intelligence_factories, population_indices)
        results = sim.run(generation_frames)
        
        # Only return scores for the individuals being evaluated
        cleaned_results = {}
        for pop_name, scores in results.items():
            eval_idx = evaluated_indices[pop_name]
            # Find the position of the evaluated index in the population indices
            idx_position = population_indices[pop_name].index(eval_idx)
            score = scores[idx_position]
            # Replace NaN or infinite values with a penalty score
            if np.isnan(score) or np.isinf(score):
                score = -1000.0
            cleaned_results[pop_name] = [float(score)]
        
        return cleaned_results
        
    except Exception as e:
        import traceback
        print(f"Error in evaluation:")
        print(traceback.format_exc())
        return {
            pop_name: [-1000.0]
            for pop_name in intelligence_instances.keys()
        }

class MultiPopulationLearningMode:
    """Manages the co-evolution of multiple populations."""
    
    def __init__(self,
                 scenario_class: Type[Scenario],
                 intelligence_classes: Dict[str, Type[Intelligence]],
                 population_sizes: Dict[str, int],
                 generation_frames: int = 500,
                 mutation_rates: Dict[str, float] = None,
                 mutation_ranges: Dict[str, float] = None,
                 elite_percentages: Dict[str, float] = None,
                 tournament_sizes: Dict[str, int] = None,
                 visualize: bool = False,
                 num_workers: Optional[int] = None,
                 save_dir: Optional[str] = None):
        """Initialize multi-population learning mode.
        
        Args:
            scenario_class: Class to use for creating scenarios
            intelligence_classes: Dict mapping population names to their intelligence classes
            population_sizes: Dict mapping population names to their sizes
            generation_frames: Number of frames to simulate per generation
            mutation_rates: Dict of mutation rates per population (default: 0.1)
            mutation_ranges: Dict of mutation ranges per population (default: 0.2)
            elite_percentages: Dict of elite percentages per population (default: 0.1)
            tournament_sizes: Dict of tournament sizes per population (default: 5)
            visualize: Whether to show real-time visualization
            num_workers: Number of parallel workers (None = use CPU count)
            save_dir: Directory to save progress plots (None = no saving)
        """
        self.scenario_class = scenario_class
        self.intelligence_classes = intelligence_classes
        self.population_sizes = population_sizes
        self.generation_frames = generation_frames
        
        # Initialize evolution parameters for each population
        self.mutation_rates = mutation_rates or {name: 0.1 for name in intelligence_classes}
        self.mutation_ranges = mutation_ranges or {name: 0.2 for name in intelligence_classes}
        self.elite_sizes = {
            name: max(1, int(size * (elite_percentages.get(name, 0.1) if elite_percentages else 0.1)))
            for name, size in population_sizes.items()
        }
        self.tournament_sizes = tournament_sizes or {name: 5 for name in intelligence_classes}
        
        # Create initial populations
        self.populations = {
            name: self._create_initial_population(intel_class, population_sizes[name])
            for name, intel_class in intelligence_classes.items()
        }
        
        # Initialize statistics tracking
        self.generation = 0
        self.stats_history: List[MultiPopulationStats] = []
        
        # Setup visualization if requested
        self.visualizer = MultiPopulationVisualizer(save_dir) if visualize else None
        
        # Setup parallel processing
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.pool = mp.Pool(processes=self.num_workers)
    
    def __del__(self):
        """Cleanup pool on deletion."""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
    
    def _create_initial_population(self, intelligence_class: Type[Intelligence], size: int) -> List[Intelligence]:
        """Create initial population with random weights."""
        population = []
        template = intelligence_class()
        weight_keys = template.weights.keys()
        
        for _ in range(size):
            individual = intelligence_class()
            for key in weight_keys:
                base_value = template.weights[key]
                individual.weights[key] = base_value * (1 + np.random.uniform(-0.5, 0.5))
            population.append(individual)
        
        return population
    
    def _tournament_select(self, population: List[Intelligence], 
                         fitnesses: List[float], tournament_size: int) -> Intelligence:
        """Select an individual using tournament selection."""
        if len(population) != len(fitnesses):
            raise ValueError(f"Population size ({len(population)}) does not match fitness size ({len(fitnesses)})")
        
        # Ensure tournament size doesn't exceed population size
        tournament_size = min(tournament_size, len(population))
        
        # Select random indices for tournament
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        
        # Find the best individual in the tournament
        best_tournament_idx = tournament_indices[0]
        best_fitness = fitnesses[best_tournament_idx]
        
        for idx in tournament_indices[1:]:
            if fitnesses[idx] > best_fitness:
                best_tournament_idx = idx
                best_fitness = fitnesses[idx]
        
        return population[best_tournament_idx]
    
    def _create_evaluation_combinations(self) -> List[Tuple[Dict[str, Dict[int, Intelligence]], int]]:
        """Create combinations of individuals from different populations for evaluation."""
        combinations = []
        
        # Debug logging
        print("\nCreating evaluation combinations:")
        print(f"Population sizes: {[(name, len(pop)) for name, pop in self.populations.items()]}")
        
        # For each individual in each population, create a full simulation setup
        for pop_name, population in self.populations.items():
            for i, individual in enumerate(population):
                # Create intelligence instances for this combination
                intelligence_instances = {}
                
                # Calculate base indices for each population
                predator_base = 0  # Predators start at 0
                prey_base = len(self.populations['predator'])  # Prey start after predators
                
                # Add all predators first
                predator_pop = self.populations['predator']
                intelligence_instances['predator'] = {}
                for pred_idx, pred in enumerate(predator_pop):
                    # If this is the individual being evaluated and it's a predator
                    if pop_name == 'predator' and i == pred_idx:
                        intelligence_instances['predator'][predator_base + pred_idx] = individual
                    else:
                        intelligence_instances['predator'][predator_base + pred_idx] = pred
                
                # Add all prey next
                prey_pop = self.populations['prey']
                intelligence_instances['prey'] = {}
                for prey_idx, prey in enumerate(prey_pop):
                    # If this is the individual being evaluated and it's a prey
                    if pop_name == 'prey' and i == prey_idx:
                        intelligence_instances['prey'][prey_base + prey_idx] = individual
                    else:
                        intelligence_instances['prey'][prey_base + prey_idx] = prey
                
                if self.generation % 10 == 0:  # Only print every 10 generations
                    print(f"\nEvaluating {pop_name} individual {i}")
                    print(f"Created instance indices: {[(name, list(inst.keys())) for name, inst in intelligence_instances.items()]}")
                combinations.append((intelligence_instances, self.generation_frames))
        
        return combinations
    
    def evolve_generation(self) -> MultiPopulationStats:
        """Evolve one generation and return statistics."""
        start_time = time.time()
        
        try:
            # Create evaluation combinations
            combinations = self._create_evaluation_combinations()
            
            # Prepare evaluation arguments
            eval_args = [
                (self.scenario_class, combo[0], combo[1])
                for combo in combinations
            ]
            
            # Evaluate combinations in parallel
            results = self.pool.map(_evaluate_combination, eval_args)
            
            # Aggregate fitness scores for each population
            all_fitnesses = {name: [] for name in self.populations}
            
            # Track which individual we're evaluating for each population
            current_individual = {name: 0 for name in self.populations}
            
            # Process results in order
            for combo, result in zip(combinations, results):
                # Determine which population and individual this result is for
                for pop_name in self.populations:
                    if current_individual[pop_name] < len(self.populations[pop_name]):
                        # This is a result for the current individual of this population
                        score = result[pop_name][0]  # We know there's only one score per result now
                        all_fitnesses[pop_name].append(score)
                        current_individual[pop_name] += 1
                        break
            
            # Calculate statistics and evolve each population
            population_stats = {}
            for pop_name, population in self.populations.items():
                fitnesses = all_fitnesses[pop_name]
                
                if len(fitnesses) != len(population):
                    print(f"Warning: Fitness count mismatch for {pop_name}. Population: {len(population)}, Fitnesses: {len(fitnesses)}")
                    # Pad or truncate fitnesses if necessary
                    while len(fitnesses) < len(population):
                        fitnesses.append(-1000.0)
                    fitnesses = fitnesses[:len(population)]
                
                # Calculate statistics
                avg_fitness = float(np.mean(fitnesses))
                max_fitness = float(np.max(fitnesses))
                min_fitness = float(np.min(fitnesses))
                
                # Find best individual
                best_idx = np.argmax(fitnesses)
                best_weights = population[best_idx].get_weights()
                
                population_stats[pop_name] = {
                    'avg_fitness': avg_fitness,
                    'max_fitness': max_fitness,
                    'min_fitness': min_fitness,
                    'best_weights': best_weights
                }
                
                # Create next generation
                next_population = []
                
                # Sort population by fitness
                sorted_indices = np.argsort(fitnesses)[::-1]
                sorted_population = [population[i] for i in sorted_indices]
                sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
                
                # Keep elite individuals
                elite_count = self.elite_sizes[pop_name]
                next_population.extend(sorted_population[:elite_count])
                
                # Fill rest with offspring
                while len(next_population) < self.population_sizes[pop_name]:
                    # Select parents using tournament selection
                    parent1 = self._tournament_select(sorted_population, sorted_fitnesses, self.tournament_sizes[pop_name])
                    parent2 = self._tournament_select(sorted_population, sorted_fitnesses, self.tournament_sizes[pop_name])
                    
                    # Create offspring
                    child = self.intelligence_classes[pop_name]()
                    
                    try:
                        # Crossover
                        child_weights = parent1.crossover(parent2)
                        child.set_weights(child_weights)
                        
                        # Mutation
                        child.mutate(self.mutation_rates[pop_name], self.mutation_ranges[pop_name])
                    except Exception as e:
                        print(f"Error in reproduction for {pop_name}: {str(e)}")
                        # Use copied weights from better parent as fallback
                        child.set_weights(parent1.get_weights().copy())
                    
                    next_population.append(child)
                
                # Update population
                self.populations[pop_name] = next_population
            
            # Create generation statistics
            stats = MultiPopulationStats(self.generation, population_stats)
            stats.generation_time = time.time() - start_time
            self.stats_history.append(stats)
            
            # Update visualization if enabled
            if self.visualizer:
                self.visualizer.update(self.stats_history)
            
            self.generation += 1
            return stats
            
        except Exception as e:
            print(f"Error in generation evolution: {str(e)}")
            # Create emergency statistics to allow evolution to continue
            population_stats = {
                name: {
                    'avg_fitness': -1000.0,
                    'max_fitness': -1000.0,
                    'min_fitness': -1000.0,
                    'best_weights': population[0].get_weights()
                }
                for name, population in self.populations.items()
            }
            stats = MultiPopulationStats(self.generation, population_stats)
            stats.generation_time = time.time() - start_time
            self.stats_history.append(stats)
            self.generation += 1
            return stats
    
    def run_evolution(self, num_generations: int) -> List[MultiPopulationStats]:
        """Run evolution for specified number of generations."""
        try:
            print(f"\nRunning co-evolution with {self.num_workers} parallel workers")
            
            for gen in range(num_generations):
                stats = self.evolve_generation()
                
                # Print progress
                print(f"\nGeneration {stats.generation}:")
                for pop_name, pop_stats in stats.stats.items():
                    print(f"{pop_name}:")
                    print(f"  Avg={pop_stats['avg_fitness']:.3f}")
                    print(f"  Max={pop_stats['max_fitness']:.3f}")
                    print(f"  Min={pop_stats['min_fitness']:.3f}")
                print(f"Time: {stats.generation_time:.1f}s")
                
                # Save statistics if directory is provided
                if self.visualizer and self.visualizer.save_dir:
                    save_dir = self.visualizer.save_dir
                    # Save generation stats
                    stats_file = os.path.join(save_dir, f"gen_{gen:04d}_stats.json")
                    with open(stats_file, 'w') as f:
                        json.dump(stats.to_dict(), f, indent=2)
                    
                    # Save best weights for each population
                    for pop_name, pop_stats in stats.stats.items():
                        weights_file = os.path.join(save_dir, f"gen_{gen:04d}_{pop_name}_best_weights.json")
                        with open(weights_file, 'w') as f:
                            json.dump(pop_stats['best_weights'], f, indent=2)
        
        finally:
            # Clean up
            if self.visualizer:
                self.visualizer.close()
            self.pool.close()
            self.pool.join()
        
        return self.stats_history
    
    def demo_best_weights(self, weights: Dict[str, Dict[str, float]], num_bots: int = 30) -> None:
        """Run a visual demo of the scenario with the provided weights for each population.
        
        Args:
            weights: Dict mapping population names to their best weights
            num_bots: Number of bots to use in demo
        """
        # Turn off interactive mode for demo
        plt.ioff()
        
        # Create intelligence factories for each population type
        intelligence_factories = {}
        for pop_name, pop_weights in weights.items():
            def create_intelligence(pop_name=pop_name, weights=pop_weights):
                intelligence = self.intelligence_classes[pop_name]()
                intelligence.set_weights(weights)
                return intelligence
            intelligence_factories[pop_name] = create_intelligence
        
        # Calculate total bots and predator ratio
        predator_count = len(self.populations['predator'])
        total_bots = sum(len(pop) for pop in self.populations.values())
        predator_ratio = predator_count / total_bots
        
        # Create scenario with the same ratio
        scenario = self.scenario_class(predator_ratio=predator_ratio, verbose=True)  # Enable verbose for demo
        
        # Create a factory function that returns the appropriate intelligence based on bot index
        def create_bot_intelligence(bot_index: int) -> Intelligence:
            # Predators are first, then prey
            if bot_index < int(num_bots * predator_ratio):
                return intelligence_factories['predator']()
            else:
                return intelligence_factories['prey']()
        
        # Print bot counts
        print(f"\nInitializing {num_bots} bots:")
        predator_demo_count = int(num_bots * predator_ratio)
        print(f"- {predator_demo_count} predators")
        print(f"- {num_bots - predator_demo_count} prey")
        
        # Create and run visual simulation
        sim = Simulation(scenario, create_bot_intelligence, num_bots)
        
        # Run simulation and block until window is closed
        sim.run()
        plt.show(block=True)
        
        # Restore interactive mode if visualization was enabled
        if self.visualizer:
            plt.ion()