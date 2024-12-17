import numpy as np
from typing import List, Type, Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from .bot import Bot
from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence

class Simulation:
    def __init__(self, scenario: Scenario, intelligence_factory: Callable[[int], Intelligence], num_bots: int = 30):
        """Initialize simulation with a scenario and intelligence factory.
        
        Args:
            scenario: Scenario instance defining the simulation environment
            intelligence_factory: Function that creates intelligence instances based on bot index
            num_bots: Number of bots to simulate
        """
        self.scenario = scenario
        self.world_size = scenario.get_world_bounds()
        
        # Initialize bots with scenario-provided positions and velocities
        bot_states = scenario.initialize_bots(num_bots)
        self.bots = [
            Bot(position, velocity, intelligence_factory(i))
            for i, (position, velocity) in enumerate(bot_states)
        ]
        
        # Setup visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.world_size[0])
        self.ax.set_ylim(0, self.world_size[1])
        # Remove axis labels and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
    def update(self, frame):
        """Update simulation state and visualization."""
        self.ax.clear()
        self.ax.set_xlim(0, self.world_size[0])
        self.ax.set_ylim(0, self.world_size[1])
        # Remove axis labels and ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Get current state of all bots
        all_states = [bot.get_state() for bot in self.bots]
        
        # Update each bot
        for i, bot in enumerate(self.bots):
            # Get neighbors within perception radius
            neighbors = []
            neighbor_indices = []  # Track indices of neighbors
            bot_pos = bot.position
            for j, other_bot in enumerate(self.bots):
                if other_bot != bot:
                    other_pos, other_vel = other_bot.get_state()
                    distance = np.linalg.norm(bot_pos - other_pos)
                    if distance < bot.intelligence.perception_radius:
                        neighbors.append((other_pos, other_vel))
                        neighbor_indices.append(j)  # Store the neighbor's index
            
            # Get scenario parameters and add bot-specific parameters
            scenario_params = self.scenario.get_scenario_params()
            scenario_params['bot_index'] = i  # Add bot's own index
            scenario_params['neighbor_indices'] = neighbor_indices  # Add neighbor indices
            
            # Update bot with scenario parameters
            bot.update(neighbors, self.world_size, **scenario_params)
            
            # Visualize bot
            circle = Circle(bot.position, 5, color=bot.intelligence.color, alpha=0.8)
            self.ax.add_patch(circle)
            
            # Draw velocity vector
            self.ax.arrow(bot.position[0], bot.position[1],
                         bot.velocity[0]*5, bot.velocity[1]*5,
                         head_width=3, head_length=5, 
                         fc=bot.intelligence.trail_color, 
                         ec=bot.intelligence.trail_color,
                         alpha=0.6)
        
        # Check scenario completion
        positions = [bot.position for bot in self.bots]
        velocities = [bot.velocity for bot in self.bots]
        if self.scenario.check_completion(positions, velocities):
            print("Scenario completed!")
            plt.close()
    
    def run(self, frames=200):
        """Run the simulation animation."""
        anim = FuncAnimation(self.fig, self.update, frames=frames, interval=50)
        plt.show() 