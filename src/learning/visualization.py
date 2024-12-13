import matplotlib.pyplot as plt
from typing import List
from .stats import GenerationStats

class EvolutionVisualizer:
    """Real-time visualization of evolution progress."""
    
    def __init__(self):
        """Initialize the visualizer."""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title('Evolution Progress')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        
        # Initialize empty lines
        self.max_line, = self.ax.plot([], [], 'g-', label='Max Fitness')
        self.avg_line, = self.ax.plot([], [], 'b-', label='Avg Fitness')
        self.min_line, = self.ax.plot([], [], 'r-', label='Min Fitness')
        
        self.ax.legend()
        self.ax.grid(True)
        
    def update(self, stats_history: List[GenerationStats]):
        """Update the visualization with new data."""
        generations = [s.generation for s in stats_history]
        max_fitness = [s.max_fitness for s in stats_history]
        avg_fitness = [s.avg_fitness for s in stats_history]
        min_fitness = [s.min_fitness for s in stats_history]
        
        # Update data
        self.max_line.set_data(generations, max_fitness)
        self.avg_line.set_data(generations, avg_fitness)
        self.min_line.set_data(generations, min_fitness)
        
        # Adjust limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the visualization."""
        plt.close(self.fig)
        plt.ioff() 