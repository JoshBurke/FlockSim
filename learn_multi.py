#!/usr/bin/env python3
"""Root-level entrypoint for multi-population learning mode."""

import argparse
import os
from datetime import datetime
from typing import Dict

from src.registry import SCENARIOS, INTELLIGENCES, get_choices
from src.utils.load_class import load_class
from src.learning.multi_population_learning_mode import MultiPopulationLearningMode
from src.intelligences.predator_prey import PredatorIntelligence, PreyIntelligence

def main():
    parser = argparse.ArgumentParser(description='Run multi-population co-evolution')
    
    # Core parameters
    parser.add_argument('--scenario', type=str, default='predator_prey',
                      choices=['predator_prey'],  # For now, only support predator-prey
                      help='Scenario to use (currently only predator-prey)')
    
    # Evolution parameters
    parser.add_argument('--generations', type=int, default=100,
                      help='Number of generations to evolve (default: 100)')
    parser.add_argument('--predator-population', type=int, default=50,
                      help='Size of predator population (default: 50)')
    parser.add_argument('--prey-population', type=int, default=50,
                      help='Size of prey population (default: 50)')
    parser.add_argument('--tournament-size', type=int, default=5,
                      help='Number of individuals in each tournament selection (default: 5)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Probability of mutating each weight (default: 0.1)')
    parser.add_argument('--mutation-range', type=float, default=0.2,
                      help='Range of mutation effect (default: 0.2)')
    parser.add_argument('--elite-percentage', type=float, default=0.1,
                      help='Percentage of top performers to keep unchanged (default: 0.1)')
    
    # Simulation parameters
    parser.add_argument('--generation-frames', type=int, default=500,
                      help='Number of frames to simulate per generation (default: 500)')
    parser.add_argument('--num-bots', type=int, default=30,
                      help='Number of bots to use in demo (default: 30)')
    
    # Performance parameters
    parser.add_argument('--num-workers', type=int, default=None,
                      help='Number of parallel workers (default: CPU count)')
    
    # Visualization control
    parser.add_argument('--no-visualization', action='store_true',
                      help='Disable real-time visualization of evolution progress')
    parser.add_argument('--no-demo', action='store_true',
                      help='Skip demo of best weights after evolution')
    
    # Save/load options
    parser.add_argument('--save-dir', type=str, default=None,
                      help='Directory to save evolution progress (default: auto-generated)')
    parser.add_argument('--load-weights', type=str, default=None,
                      help='Load and demo weights from a file (skips evolution)')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("evolution_results", f"multi_predator_prey_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nStarting co-evolution with:")
    print(f"Scenario: {args.scenario}")
    print("\nEvolution parameters:")
    print(f"- Generations: {args.generations}")
    print(f"- Predator population: {args.predator_population}")
    print(f"- Prey population: {args.prey_population}")
    print(f"- Tournament size: {args.tournament_size}")
    print(f"- Elite percentage: {args.elite_percentage*100}%")
    print(f"- Mutation rate: {args.mutation_rate}")
    print(f"- Mutation range: ±{args.mutation_range*100}%")
    print(f"\nSaving results to: {save_dir}")
    
    # Load scenario class
    scenario_class = load_class(SCENARIOS[args.scenario])
    
    # Create learning mode
    learning_mode = MultiPopulationLearningMode(
        scenario_class=scenario_class,
        intelligence_classes={
            'predator': PredatorIntelligence,
            'prey': PreyIntelligence
        },
        population_sizes={
            'predator': args.predator_population,
            'prey': args.prey_population
        },
        tournament_sizes={
            'predator': args.tournament_size,
            'prey': args.tournament_size
        },
        mutation_rates={
            'predator': args.mutation_rate,
            'prey': args.mutation_rate
        },
        mutation_ranges={
            'predator': args.mutation_range,
            'prey': args.mutation_range
        },
        elite_percentages={
            'predator': args.elite_percentage,
            'prey': args.elite_percentage
        },
        visualize=not args.no_visualization,  # Default to showing visualization
        num_workers=args.num_workers,
        save_dir=save_dir
    )
    
    # Run evolution
    stats_history = learning_mode.run_evolution(args.generations)
    
    # Find best generation for each population
    best_weights = {}
    
    # Find best generation for each population using average fitness
    for pop_name in learning_mode.populations.keys():
        best_gen = max(stats_history, key=lambda x: x.stats[pop_name]['avg_fitness'])
        best_weights[pop_name] = best_gen.stats[pop_name]['best_weights']
        print(f"\nBest {pop_name} performance in generation {best_gen.generation}:")
        print(f"  Pop Avg: {best_gen.stats[pop_name]['avg_fitness']:.3f}")
        print(f"  Best Individual: {best_gen.stats[pop_name]['max_fitness']:.3f}")
    
    print(f"\nResults saved in: {save_dir}")
    
    # Run demo unless disabled
    if not args.no_demo:
        print("\nRunning demo with best weights...")
        learning_mode.demo_best_weights(best_weights)

if __name__ == "__main__":
    main() 