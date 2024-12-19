import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os
from .multi_population_stats import MultiPopulationStats

class MultiPopulationVisualizer:
    """Visualizes co-evolution progress for multiple populations."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots (None for no saving)
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure with subplots for different metrics
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 15))
        self.fig.suptitle('Co-Evolution Progress', fontsize=14)
        
        # Configure subplots
        titles = ['Average Fitness', 'Maximum Fitness', 'Minimum Fitness']
        metrics = ['avg_fitness', 'max_fitness', 'min_fitness']
        for ax, title, metric in zip(self.axes, titles, metrics):
            ax.set_title(title)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
        
        # Initialize plot data
        self.plot_data = {}  # Will store line objects for each population/metric
        
        # Set up color cycle for populations
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors
        self.color_index = 0
        
        # Show the window without blocking
        plt.show(block=False)
    
    def _get_next_color(self) -> tuple:
        """Get next color from cycle."""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
    
    def update(self, stats_history: List[MultiPopulationStats]):
        """Update visualization with new statistics.
        
        Args:
            stats_history: List of MultiPopulationStats objects
        """
        generations = list(range(len(stats_history)))
        
        # Get all population names from the most recent generation
        population_names = stats_history[-1].stats.keys()
        
        # Update each subplot
        for ax_idx, (ax, metric) in enumerate(zip(self.axes, ['avg_fitness', 'max_fitness', 'min_fitness'])):
            # Clear existing lines if this is the first update
            if not self.plot_data:
                ax.clear()
                ax.grid(True)
            
            # Plot each population's data
            for pop_name in population_names:
                # Create unique line identifier
                line_id = f"{pop_name}_{metric}"
                
                # Get data for this population and metric
                values = [gen.stats[pop_name][metric] for gen in stats_history]
                
                if line_id not in self.plot_data:
                    # Create new line with next color
                    color = self._get_next_color()
                    line, = ax.plot(generations, values, label=pop_name, color=color)
                    self.plot_data[line_id] = line
                else:
                    # Update existing line
                    self.plot_data[line_id].set_data(generations, values)
            
            # Update axis limits
            ax.relim()
            ax.autoscale_view()
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Adjust layout and draw
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save plot if directory is provided
        if self.save_dir:
            plot_file = os.path.join(self.save_dir, f"gen_{len(stats_history)-1:04d}_plot.png")
            self.fig.savefig(plot_file, bbox_inches='tight')
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)


class EvolutionVisualizer:
    """Original visualizer for single population evolution (kept for compatibility)."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots (None for no saving)
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create figure and configure plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title('Evolution Progress')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.ax.grid(True)
        
        # Initialize plot lines
        self.lines = {
            'avg': None,
            'max': None,
            'min': None
        }
    
    def update(self, stats_history: List):
        """Update visualization with new statistics."""
        generations = list(range(len(stats_history)))
        
        # Extract fitness values
        avg_fitness = [s.avg_fitness for s in stats_history]
        max_fitness = [s.max_fitness for s in stats_history]
        min_fitness = [s.min_fitness for s in stats_history]
        
        # Create or update plot lines
        if self.lines['avg'] is None:
            self.lines['avg'], = self.ax.plot(generations, avg_fitness, 'b-', label='Average')
            self.lines['max'], = self.ax.plot(generations, max_fitness, 'g-', label='Maximum')
            self.lines['min'], = self.ax.plot(generations, min_fitness, 'r-', label='Minimum')
            self.ax.legend()
        else:
            self.lines['avg'].set_data(generations, avg_fitness)
            self.lines['max'].set_data(generations, max_fitness)
            self.lines['min'].set_data(generations, min_fitness)
        
        # Update axis limits
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Draw plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save plot if directory is provided
        if self.save_dir:
            plot_file = os.path.join(self.save_dir, f"gen_{len(stats_history)-1:04d}_plot.png")
            self.fig.savefig(plot_file, bbox_inches='tight')
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)