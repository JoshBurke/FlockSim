from dataclasses import dataclass
from typing import Dict

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    best_weights: Dict[str, float]
    generation_time: float  # Time in seconds to evaluate this generation
    
    def to_dict(self) -> Dict:
        """Convert stats to dictionary for saving."""
        return {
            'generation': self.generation,
            'avg_fitness': self.avg_fitness,
            'max_fitness': self.max_fitness,
            'min_fitness': self.min_fitness,
            'best_weights': self.best_weights,
            'generation_time': self.generation_time
        } 