# OmniFlock

A Python-based swarm intelligence simulator that supports multiple scenarios and pluggable bot intelligences. The system includes both direct simulation and evolutionary learning capabilities, with high-performance parallel evolution.

## Features

- Real-time visualization of swarm behavior
- Multiple bot intelligence types
- Configurable scenarios
- High-performance parallel evolution (8x faster with 8 cores!)
- Progress visualization and tracking
- Extensible architecture

## Requirements

- Python 3.8+
- pipenv (for dependency management)

Core Dependencies (managed by pipenv):
- NumPy
- Matplotlib

## Installation

1. Install pipenv if you haven't already:
```bash
pip install pipenv
```

2. Install project dependencies:
```bash
pipenv install
```

3. Activate the virtual environment:
```bash
pipenv shell
```

## Usage

The project provides two main entry points:

### Direct Simulation (`main.py`)

Run direct simulations with configurable parameters:

```bash
python main.py [options]
```

Options:
- `--scenario`: Scenario to run (default: free_roam)
- `--intelligence`: Intelligence type to use (default: flocking)
- `--num-bots`: Number of bots to simulate (default: 30)
- `--width`: Width of simulation world (default: 800)
- `--height`: Height of simulation world (default: 600)
- `--wrap`: Enable wrapping at world boundaries (default: False)

Bot Parameters:
- `--max-speed`: Maximum speed of bots (default: 2.0)
- `--max-force`: Maximum force that can be applied (default: 0.1)
- `--perception-radius`: How far bots can see (default: 60.0)
- `--separation-radius`: Distance for separation force (default: 25.0)
- `--wall-detection-distance`: Wall avoidance distance (default: 50.0)

Flocking Weights:
- `--cohesion-weight`: Weight of cohesion force (default: 1.5)
- `--alignment-weight`: Weight of alignment force (default: 3.0)
- `--separation-weight`: Weight of separation force (default: 0.8)
- `--wall-avoidance-weight`: Weight of wall avoidance (default: 2.5)
- `--leader-bias`: How much to favor bots in front (default: 4.0)

### Evolution Mode (`learn.py`)

Run evolutionary learning to optimize bot behavior. Evolution mode automatically utilizes all available CPU cores for parallel processing, providing near-linear speedup (e.g., 8x faster with 8 cores).

```bash
python learn.py [options]
```

Core Parameters:
- `--scenario`: Scenario to use (default: free_roam)
- `--intelligence`: Intelligence type to use (default: flocking)
- `--list`: List available scenarios and intelligences

Evolution Parameters:
- `--generations`: Number of generations (default: 100)
- `--population-size`: Population size (default: 50)
- `--tournament-size`: Tournament selection size (default: 5)
- `--mutation-rate`: Mutation probability (default: 0.1)
- `--mutation-range`: Mutation effect range (default: 0.2)
- `--elite-percentage`: Elite preservation rate (default: 0.1)

Simulation Parameters:
- `--generation-frames`: Frames per generation (default: 500)
- `--num-bots`: Number of bots in demo (default: 30)

Performance:
- `--num-workers`: Number of parallel workers (default: uses all CPU cores)
  - For best performance, let it use all cores
  - Each worker runs on a separate CPU core
  - Near-linear speedup (e.g., 8 cores = ~8x faster)
  - Example: A 100-generation evolution that takes 2 hours on 1 core can complete in 15 minutes on 8 cores

Visualization:
- `--no-visualization`: Disable real-time visualization
- `--no-demo`: Skip final demo
- `--save-dir`: Save directory (default: auto-generated)
- `--load-weights`: Load and demo existing weights

## Adding Custom Components

### Adding a New Intelligence

1. Create a new class in `src/intelligences/` that inherits from `Intelligence`:
```python
from .base import Intelligence

class MyIntelligence(Intelligence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = {
            'param1': 1.0,
            'param2': 2.0
        }
    
    def calculate_move(self, position, velocity, neighbors, world_size, **kwargs):
        # Implement movement logic
        return force_vector
    
    def update_fitness_metrics(self, position, velocity, neighbors, world_size):
        # Track metrics for fitness calculation
        pass
    
    def calculate_fitness(self) -> float:
        # Return fitness score (0 to 1)
        return fitness_score
```

2. Register in `src/registry.py`:
```python
INTELLIGENCES = {
    'my_intelligence': 'src.intelligences.my_intelligence.MyIntelligence'
}
```

### Adding a New Scenario

1. Create a new class in `src/scenarios/` that inherits from `Scenario`:
```python
from .base import Scenario

class MyScenario(Scenario):
    def initialize_bots(self, num_bots):
        # Return list of (position, velocity) tuples
        return initial_states
    
    def get_specific_params(self):
        # Return scenario-specific parameters
        return {'param1': value1}
    
    def check_completion(self, positions, velocities):
        # Return True if scenario is complete
        return is_complete
```

2. Register in `src/registry.py`:
```python
SCENARIOS = {
    'my_scenario': 'src.scenarios.my_scenario.MyScenario'
}
```

## Evolution Results

Evolution results are saved in `evolution_results/` with timestamp-based directories:
- `gen_XXXX_stats.json`: Statistics for each generation
- `gen_XXXX_best_weights.json`: Best weights from each generation
- `evolution_progress.png`: Real-time progress plot
- `evolution_final.png`: High-quality final plot
