import argparse
import os
from datetime import datetime
from ..scenarios.free_roam import FreeRoamScenario
from ..intelligences.flocking import FlockingIntelligence
from .learning_mode import LearningMode

def main():
    parser = argparse.ArgumentParser(description='Run learning mode for bot evolution')
    
    # Learning parameters
    parser.add_argument('--generations', type=int, default=100,
                      help='Number of generations to evolve (default: 100)')
    parser.add_argument('--population-size', type=int, default=50,
                      help='Number of individuals in population (default: 50)')
    parser.add_argument('--generation-frames', type=int, default=500,
                      help='Number of frames to simulate per generation (default: 500)')
    
    # Evolution parameters
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Probability of mutating each weight (default: 0.1)')
    parser.add_argument('--mutation-range', type=float, default=0.2,
                      help='Range of mutation effect (default: 0.2)')
    parser.add_argument('--elite-percentage', type=float, default=0.1,
                      help='Percentage of top performers to keep unchanged (default: 0.1)')
    parser.add_argument('--tournament-size', type=int, default=5,
                      help='Number of individuals in tournament selection (default: 5)')
    
    # Save/load parameters
    parser.add_argument('--save-dir', type=str, default=None,
                      help='Directory to save evolution progress (default: None)')
    parser.add_argument('--load-weights', type=str, default=None,
                      help='Load initial weights from file (default: None)')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp if not specified
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = os.path.join('evolution_results', f'run_{timestamp}')
    
    # Initialize learning mode
    learning = LearningMode(
        scenario_class=FreeRoamScenario,
        intelligence_class=FlockingIntelligence,
        population_size=args.population_size,
        generation_frames=args.generation_frames,
        mutation_rate=args.mutation_rate,
        mutation_range=args.mutation_range,
        elite_percentage=args.elite_percentage,
        tournament_size=args.tournament_size
    )
    
    # Load initial weights if specified
    if args.load_weights:
        learning.load_weights(args.load_weights)
    
    # Run evolution
    stats_history = learning.run_evolution(args.generations, args.save_dir)
    
    # Print final results
    best_gen = max(stats_history, key=lambda x: x.max_fitness)
    print("\nEvolution complete!")
    print(f"Best fitness: {best_gen.max_fitness:.3f} (Generation {best_gen.generation})")
    print(f"Best weights: {best_gen.best_weights}")
    if args.save_dir:
        print(f"Results saved in: {args.save_dir}")

if __name__ == "__main__":
    main() 