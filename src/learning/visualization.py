import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
from .stats import GenerationStats

class EvolutionVisualizer:
    """Real-time visualization of evolution progress."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save progress plots (None = no saving)
        """
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax_fitness, self.ax_time) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        self.save_dir = save_dir
        
        # Setup fitness plot
        self.ax_fitness.set_title('Fitness Progress')
        self.ax_fitness.set_xlabel('Generation')
        self.ax_fitness.set_ylabel('Fitness')
        
        # Initialize fitness lines
        self.max_line, = self.ax_fitness.plot([], [], 'g-', label='Max Fitness')
        self.avg_line, = self.ax_fitness.plot([], [], 'b-', label='Avg Fitness')
        self.min_line, = self.ax_fitness.plot([], [], 'r-', label='Min Fitness')
        
        self.ax_fitness.legend()
        self.ax_fitness.grid(True)
        
        # Setup timing plot
        self.ax_time.set_title('Generation Time')
        self.ax_time.set_xlabel('Generation')
        self.ax_time.set_ylabel('Time (seconds)')
        
        # Initialize timing line
        self.time_line, = self.ax_time.plot([], [], 'k-', label='Generation Time')
        self.time_avg_line, = self.ax_time.plot([], [], 'k--', label='Running Average')
        
        self.ax_time.legend()
        self.ax_time.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
    def update(self, stats_history: List[GenerationStats]):
        """Update the visualization with new data."""
        generations = [s.generation for s in stats_history]
        
        # Update fitness plot
        max_fitness = [s.max_fitness for s in stats_history]
        avg_fitness = [s.avg_fitness for s in stats_history]
        min_fitness = [s.min_fitness for s in stats_history]
        
        self.max_line.set_data(generations, max_fitness)
        self.avg_line.set_data(generations, avg_fitness)
        self.min_line.set_data(generations, min_fitness)
        
        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()
        
        # Update timing plot
        times = [s.generation_time for s in stats_history]
        self.time_line.set_data(generations, times)
        
        # Calculate and plot running average (last 10 generations)
        window_size = min(10, len(times))
        running_avg = np.convolve(times, np.ones(window_size)/window_size, mode='valid')
        # Pad the start of running average to match generations
        padding = len(times) - len(running_avg)
        if padding > 0:
            running_avg = np.pad(running_avg, (padding, 0), mode='edge')
        self.time_avg_line.set_data(generations, running_avg)
        
        self.ax_time.relim()
        self.ax_time.autoscale_view()
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save progress plot if directory is specified
        if self.save_dir:
            # Save current progress plot
            progress_file = os.path.join(self.save_dir, 'evolution_progress.png')
            self.fig.savefig(progress_file, dpi=100, bbox_inches='tight')
    
    def save_final_plot(self):
        """Save a high-quality final version of the plot."""
        if self.save_dir:
            # Save high-quality final plot
            final_file = os.path.join(self.save_dir, 'evolution_final.png')
            self.fig.savefig(final_file, dpi=300, bbox_inches='tight')
    
    def close(self):
        """Save final plot and close visualization."""
        if self.save_dir:
            self.save_final_plot()
        plt.close(self.fig)
        plt.ioff()