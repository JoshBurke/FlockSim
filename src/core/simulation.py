import numpy as np
from typing import List, Type
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from .bot import Bot
from ..scenarios.base import Scenario
from ..intelligences.base import Intelligence

class Simulation:
    def __init__(self, scenario: Scenario, intelligence_class: Type[Intelligence], num_bots: int = 30):
        """Initialize simulation with a scenario and intelligence type.
        
        Args:
            scenario: Scenario instance defining the simulation environment
            intelligence_class: Class to use for bot intelligence
            num_bots: Number of bots to simulate
        """
        self.scenario = scenario
        self.world_size = scenario.get_world_bounds()
        
        # Initialize bots with scenario-provided positions and velocities
        bot_states = scenario.initialize_bots(num_bots)
        self.bots = [
            Bot(position, velocity, intelligence_class())
            for position, velocity in bot_states
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
        for bot in self.bots:
            # Get neighbors within perception radius
            neighbors = []
            bot_pos = bot.position
            for other_bot in self.bots:
                if other_bot != bot:
                    other_pos, other_vel = other_bot.get_state()
                    distance = np.linalg.norm(bot_pos - other_pos)
                    if distance < bot.intelligence.perception_radius:
                        neighbors.append((other_pos, other_vel))
            
            # Update bot with scenario parameters
            bot.update(neighbors, self.world_size, **self.scenario.get_scenario_params())
            
            # Visualize bot
            circle = Circle(bot.position, 5, color='blue', alpha=0.6)
            self.ax.add_patch(circle)
            # Draw velocity vector
            self.ax.arrow(bot.position[0], bot.position[1],
                         bot.velocity[0]*5, bot.velocity[1]*5,
                         head_width=3, head_length=5, fc='red', ec='red')
        
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