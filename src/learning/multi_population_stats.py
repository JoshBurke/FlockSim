from typing import Dict

class MultiPopulationStats:
    """Statistics for a generation of multiple populations."""
    
    def __init__(self, generation: int, population_stats: Dict[str, Dict[str, float]]):
        """Initialize multi-population statistics.
        
        Args:
            generation: Generation number
            population_stats: Dict mapping population names to their statistics
                Each population's stats is a dict with keys:
                - avg_fitness: Average fitness of the population
                - max_fitness: Maximum fitness in the population
                - min_fitness: Minimum fitness in the population
                - best_weights: Weights of the best individual
        """
        self.generation = generation
        self.stats = population_stats
        self.generation_time = 0.0
    
    def to_dict(self) -> Dict:
        """Convert stats to a dictionary for saving."""
        return {
            'generation': self.generation,
            'generation_time': self.generation_time,
            'population_stats': self.stats
        } 