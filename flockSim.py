import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import random

class Bot:
    def __init__(self, x, y, world_size):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.acceleration = np.array([0.0, 0.0])
        self.max_speed = 2.0
        self.max_force = 0.1
        self.perception_radius = 50
        self.world_size = world_size
        
    def update(self):
        # Update position and velocity
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity
        
        # Wrap around world boundaries
        self.position[0] %= self.world_size[0]
        self.position[1] %= self.world_size[1]
        
        # Reset acceleration
        self.acceleration = np.array([0.0, 0.0])
    
    def apply_force(self, force):
        self.acceleration += force

class SwarmSimulation:
    def __init__(self, num_bots=30, width=800, height=600):
        self.width = width
        self.height = height
        self.bots = []
        
        # Create bots with random initial positions
        for _ in range(num_bots):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            self.bots.append(Bot(x, y, (width, height)))
        
        # Setup visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
    
    def apply_flocking_behavior(self):
        for bot in self.bots:
            # Get neighbors within perception radius
            neighbors = self.get_neighbors(bot)
            
            if neighbors:
                # Calculate flocking forces
                cohesion = self.cohesion(bot, neighbors)
                alignment = self.alignment(bot, neighbors)
                separation = self.separation(bot, neighbors)
                
                # Apply forces with different weights
                bot.apply_force(cohesion * 1.0)
                bot.apply_force(alignment * 1.2)
                bot.apply_force(separation * 1.5)
    
    def get_neighbors(self, bot):
        neighbors = []
        for other in self.bots:
            if other != bot:
                distance = np.linalg.norm(bot.position - other.position)
                if distance < bot.perception_radius:
                    neighbors.append(other)
        return neighbors
    
    def cohesion(self, bot, neighbors):
        if not neighbors:
            return np.zeros(2)
        
        # Calculate weighted center of mass based on frontal position
        weighted_positions = []
        weights = []
        for n in neighbors:
            # Calculate direction to neighbor
            to_neighbor = n.position - bot.position
            # Calculate dot product with current velocity to determine if neighbor is in front
            # Normalize velocity for the dot product
            norm_velocity = bot.velocity / np.linalg.norm(bot.velocity) if np.linalg.norm(bot.velocity) > 0 else np.zeros(2)
            norm_to_neighbor = to_neighbor / np.linalg.norm(to_neighbor) if np.linalg.norm(to_neighbor) > 0 else np.zeros(2)
            front_weight = np.dot(norm_velocity, norm_to_neighbor)
            # Remap from [-1, 1] to [0.5, 2.0] to give higher weight to frontal neighbors
            front_weight = 1.75 + front_weight * 0.75
            
            weighted_positions.append(n.position * front_weight)
            weights.append(front_weight)
        
        # Calculate weighted center
        center = np.average(weighted_positions, axis=0, weights=weights)
        
        # Create steering force towards center
        desired = center - bot.position
        if np.linalg.norm(desired) > 0:
            desired = (desired / np.linalg.norm(desired)) * bot.max_speed
        
        steer = desired - bot.velocity
        return np.clip(steer, -bot.max_force, bot.max_force)
    
    def alignment(self, bot, neighbors):
        if not neighbors:
            return np.zeros(2)
        
        # Calculate weighted average velocity based on frontal position
        weighted_velocities = []
        weights = []
        for n in neighbors:
            # Calculate direction to neighbor
            to_neighbor = n.position - bot.position
            # Calculate dot product with current velocity
            norm_velocity = bot.velocity / np.linalg.norm(bot.velocity) if np.linalg.norm(bot.velocity) > 0 else np.zeros(2)
            norm_to_neighbor = to_neighbor / np.linalg.norm(to_neighbor) if np.linalg.norm(to_neighbor) > 0 else np.zeros(2)
            front_weight = np.dot(norm_velocity, norm_to_neighbor)
            # Remap from [-1, 1] to [0.5, 2.0]
            front_weight = 1.75 + front_weight * 0.75
            
            weighted_velocities.append(n.velocity * front_weight)
            weights.append(front_weight)
        
        # Calculate weighted average velocity
        avg_vel = np.average(weighted_velocities, axis=0, weights=weights)
        
        if np.linalg.norm(avg_vel) > 0:
            avg_vel = (avg_vel / np.linalg.norm(avg_vel)) * bot.max_speed
            
        steer = avg_vel - bot.velocity
        return np.clip(steer, -bot.max_force, bot.max_force)
    
    def separation(self, bot, neighbors):
        if not neighbors:
            return np.zeros(2)
        
        steer = np.zeros(2)
        for other in neighbors:
            diff = bot.position - other.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                # Weight by distance (closer neighbors have stronger effect)
                steer += (diff / dist) / dist
        
        if np.linalg.norm(steer) > 0:
            steer = (steer / np.linalg.norm(steer)) * bot.max_speed
            
        return np.clip(steer - bot.velocity, -bot.max_force, bot.max_force)
    
    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        
        # Apply flocking behavior and update all bots
        self.apply_flocking_behavior()
        for bot in self.bots:
            bot.update()
            circle = Circle(bot.position, 5, color='blue', alpha=0.6)
            self.ax.add_patch(circle)
            # Draw velocity vector
            self.ax.arrow(bot.position[0], bot.position[1], 
                         bot.velocity[0]*5, bot.velocity[1]*5,
                         head_width=3, head_length=5, fc='red', ec='red')
    
    def run(self, frames=200):
        anim = FuncAnimation(self.fig, self.update, frames=frames, interval=50)
        plt.show()

# Example usage
if __name__ == "__main__":
    sim = SwarmSimulation(num_bots=30)
    sim.run()