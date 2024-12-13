import argparse
from src.scenarios.free_roam import FreeRoamScenario
from src.intelligences.flocking import FlockingIntelligence
from src.core.simulation import Simulation

def main():
    parser = argparse.ArgumentParser(description='Run swarm simulation with different scenarios and intelligences')
    parser.add_argument('--scenario', default='free_roam', choices=['free_roam'],
                      help='Scenario to run (default: free_roam)')
    parser.add_argument('--intelligence', default='flocking', choices=['flocking'],
                      help='Intelligence type to use (default: flocking)')
    parser.add_argument('--num-bots', type=int, default=30,
                      help='Number of bots to simulate (default: 30)')
    parser.add_argument('--width', type=float, default=800,
                      help='Width of simulation world (default: 800)')
    parser.add_argument('--height', type=float, default=600,
                      help='Height of simulation world (default: 600)')
    
    args = parser.parse_args()
    
    # Map scenario names to classes
    scenarios = {
        'free_roam': FreeRoamScenario
    }
    
    # Map intelligence names to classes
    intelligences = {
        'flocking': FlockingIntelligence
    }
    
    # Create scenario and simulation
    scenario_class = scenarios[args.scenario]
    intelligence_class = intelligences[args.intelligence]
    
    scenario = scenario_class(width=args.width, height=args.height)
    sim = Simulation(scenario, intelligence_class, args.num_bots)
    
    print(f"Running simulation with:")
    print(f"- Scenario: {args.scenario}")
    print(f"- Intelligence: {args.intelligence}")
    print(f"- Number of bots: {args.num_bots}")
    print(f"- World size: {args.width}x{args.height}")
    
    sim.run()

if __name__ == "__main__":
    main() 