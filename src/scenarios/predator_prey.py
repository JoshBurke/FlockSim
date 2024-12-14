from typing import List, Dict, Any
from src.core.scenario import Scenario
from src.core.bot import Bot
from src.core.spatial_grid import SpatialGrid

class PredatorPreyScenario(Scenario):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.predators: List[Bot] = []
        self.prey: List[Bot] = []
        self.spatial_grid = SpatialGrid(
            config["world_size"],
            config["cell_size"]
        )
        
    def setup(self):
        # Create initial populations
        self._spawn_predators(self.config["num_predators"])
        self._spawn_prey(self.config["num_prey"])
        
    def step(self):
        # Update all agents
        for predator in self.predators:
            predator.update()
        for prey in self.prey:
            prey.update()
            
        # Handle interactions (catches, escapes)
        self._handle_interactions()
        
        # Optional: Handle respawning if using fixed population sizes
        
    def _handle_interactions(self):
        # Use spatial grid to efficiently check for nearby agents
        # Handle catches, removes prey if caught, update scores
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "predator_catches": self.total_catches,
            "surviving_prey": len(self.prey),
            "avg_predator_fitness": self._calculate_predator_fitness(),
            "avg_prey_fitness": self._calculate_prey_fitness()
        } 