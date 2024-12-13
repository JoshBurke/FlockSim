import argparse
from src.registry import SCENARIOS, INTELLIGENCES, get_choices
from src.scenarios.free_roam import FreeRoamScenario
from src.intelligences.flocking import FlockingIntelligence
from src.core.simulation import Simulation

def main():
    parser = argparse.ArgumentParser(description='Run swarm simulation with different scenarios and intelligences')
    # Scenario parameters
    parser.add_argument('--scenario', default='free_roam', choices=get_choices(SCENARIOS),
                      help='Scenario to run (default: free_roam)')
    parser.add_argument('--intelligence', default='flocking', choices=get_choices(INTELLIGENCES),
                      help='Intelligence type to use (default: flocking)')
    parser.add_argument('--num-bots', type=int, default=30,
                      help='Number of bots to simulate (default: 30)')
    parser.add_argument('--width', type=float, default=800,
                      help='Width of simulation world (default: 800)')
    parser.add_argument('--height', type=float, default=600,
                      help='Height of simulation world (default: 600)')
    parser.add_argument('--wrap', action='store_true',
                      help='Enable wrapping at world boundaries (default: False, bots bounce off walls)')
    
    # Bot parameters
    parser.add_argument('--max-speed', type=float, default=2.0,
                      help='Maximum speed of bots (default: 2.0)')
    parser.add_argument('--max-force', type=float, default=0.1,
                      help='Maximum force that can be applied to bots (default: 0.1)')
    parser.add_argument('--perception-radius', type=float, default=60.0,
                      help='How far bots can see other bots (default: 60.0)')
    parser.add_argument('--separation-radius', type=float, default=25.0,
                      help='Distance at which separation force starts (default: 25.0)')
    parser.add_argument('--wall-detection-distance', type=float, default=50.0,
                      help='Distance at which bots start avoiding walls (default: 50.0)')
    
    # Flocking weights
    parser.add_argument('--cohesion-weight', type=float, default=1.5,
                      help='Weight of cohesion force (default: 1.5)')
    parser.add_argument('--alignment-weight', type=float, default=3.0,
                      help='Weight of alignment force - high for strong flock synchronization (default: 3.0)')
    parser.add_argument('--separation-weight', type=float, default=0.8,
                      help='Weight of separation force (default: 0.8)')
    parser.add_argument('--wall-avoidance-weight', type=float, default=2.5,
                      help='Weight of wall avoidance force (default: 2.5)')
    parser.add_argument('--leader-bias', type=float, default=4.0,
                      help='How much to favor bots in front - higher values create stronger leader following (default: 4.0)')
    
    args = parser.parse_args()
    
    # Create scenario and simulation
    scenario_class = SCENARIOS[args.scenario]
    intelligence_class = INTELLIGENCES[args.intelligence]
    
    # Configure intelligence with parameters
    def create_intelligence():
        return FlockingIntelligence(
            max_speed=args.max_speed,
            max_force=args.max_force,
            perception_radius=args.perception_radius,
            separation_radius=args.separation_radius,
            wall_detection_distance=args.wall_detection_distance,
            cohesion_weight=args.cohesion_weight,
            alignment_weight=args.alignment_weight,
            separation_weight=args.separation_weight,
            wall_avoidance_weight=args.wall_avoidance_weight,
            leader_bias=args.leader_bias
        )
    
    scenario = scenario_class(
        width=args.width, 
        height=args.height,
        enable_wrapping=args.wrap
    )
    sim = Simulation(scenario, create_intelligence, args.num_bots)
    
    print(f"Running simulation with:")
    print(f"- Scenario: {args.scenario}")
    print(f"- Intelligence: {args.intelligence}")
    print(f"- Number of bots: {args.num_bots}")
    print(f"- World size: {args.width}x{args.height}")
    print(f"- World wrapping: {'enabled' if args.wrap else 'disabled'}")
    print("\nBot parameters:")
    print(f"- Max speed: {args.max_speed}")
    print(f"- Max force: {args.max_force}")
    print(f"- Perception radius: {args.perception_radius}")
    print(f"- Separation radius: {args.separation_radius}")
    print(f"- Wall detection distance: {args.wall_detection_distance}")
    print("\nFlocking weights:")
    print(f"- Cohesion: {args.cohesion_weight}")
    print(f"- Alignment: {args.alignment_weight}")
    print(f"- Separation: {args.separation_weight}")
    print(f"- Wall avoidance: {args.wall_avoidance_weight}")
    print(f"- Leader bias: {args.leader_bias}")
    
    sim.run()

if __name__ == "__main__":
    main() 