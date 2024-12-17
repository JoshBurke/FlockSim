import argparse
from src.registry import SCENARIOS, INTELLIGENCES, get_choices
from src.utils.load_class import load_class
from src.intelligences.base import Intelligence
from src.intelligences.predator_prey import PredatorIntelligence, PreyIntelligence
from src.core.simulation import Simulation
from src.config import FLOCKING_DEFAULTS

def main():
    parser = argparse.ArgumentParser(description='Run swarm simulation with different scenarios and intelligences')
    # Scenario parameters
    parser.add_argument('--scenario', default='free_roam', choices=get_choices(SCENARIOS),
                      help='Scenario to run (default: free_roam)')
    parser.add_argument('--intelligence', default='flocking', choices=get_choices(INTELLIGENCES),
                      help='Intelligence type to use (default: flocking)')
    parser.add_argument('--num-bots', type=int, default=30,
                      help='Total number of bots to simulate (default: 30)')
    parser.add_argument('--width', type=float, default=800,
                      help='Width of simulation world (default: 800)')
    parser.add_argument('--height', type=float, default=600,
                      help='Height of simulation world (default: 600)')
    parser.add_argument('--wrap', action='store_true',
                      help='Enable wrapping at world boundaries (default: False, bots bounce off walls)')
    
    # Predator-prey specific parameters
    parser.add_argument('--predator-ratio', type=float, default=0.167,
                      help='Ratio of predators to total bots (default: 0.167 - roughly 1:5 predator:prey)')
    parser.add_argument('--max-time', type=int, default=1000,
                      help='Maximum simulation frames (default: 1000)')
    parser.add_argument('--win-threshold', type=float, default=0.90,
                      help='Ratio of prey that must be caught for predators to win (default: 0.90)')
    
    # Bot parameters
    parser.add_argument('--max-speed', type=float, default=2.0,
                      help='Maximum speed of bots (default: 2.0)')
    parser.add_argument('--max-force', type=float, default=0.1,
                      help='Maximum force that can be applied to bots (default: 0.1)')
    parser.add_argument('--perception-radius', type=float, default=FLOCKING_DEFAULTS['perception_radius'],
                      help=f'How far bots can see other bots (default: {FLOCKING_DEFAULTS["perception_radius"]})')
    parser.add_argument('--separation-radius', type=float, default=FLOCKING_DEFAULTS['separation_radius'],
                      help=f'Distance at which separation force starts (default: {FLOCKING_DEFAULTS["separation_radius"]})')
    parser.add_argument('--wall-detection-distance', type=float, default=50.0,
                      help='Distance at which bots start avoiding walls (default: 50.0)')
    
    # Flocking weights
    parser.add_argument('--cohesion-weight', type=float, default=FLOCKING_DEFAULTS['cohesion'],
                      help=f'Weight of cohesion force (default: {FLOCKING_DEFAULTS["cohesion"]})')
    parser.add_argument('--alignment-weight', type=float, default=FLOCKING_DEFAULTS['alignment'],
                      help=f'Weight of alignment force (default: {FLOCKING_DEFAULTS["alignment"]})')
    parser.add_argument('--separation-weight', type=float, default=FLOCKING_DEFAULTS['separation'],
                      help=f'Weight of separation force (default: {FLOCKING_DEFAULTS["separation"]})')
    parser.add_argument('--wall-avoidance-weight', type=float, default=FLOCKING_DEFAULTS['wall_avoidance'],
                      help=f'Weight of wall avoidance force (default: {FLOCKING_DEFAULTS["wall_avoidance"]})')
    parser.add_argument('--leader-bias', type=float, default=FLOCKING_DEFAULTS['leader_bias'],
                      help=f'How much to favor bots in front (default: {FLOCKING_DEFAULTS["leader_bias"]})')
    
    args = parser.parse_args()
    
    # Create scenario
    scenario_class = load_class(SCENARIOS[args.scenario])
    
    # Create scenario with appropriate parameters
    if args.scenario == 'predator_prey':
        # For predator-prey, default to 30 total bots if not specified
        if args.num_bots == 30:  # If user didn't specify a different number
            args.num_bots = 30  # 25 prey + 5 predators
        
        scenario = scenario_class(
            width=args.width,
            height=args.height,
            enable_wrapping=args.wrap,
            predator_ratio=args.predator_ratio,
            max_time=args.max_time,
            win_threshold=args.win_threshold
        )
        
        # Create intelligence factory for predator-prey scenario
        def create_intelligence(bot_index: int) -> Intelligence:
            if bot_index in scenario.predator_indices:
                return PredatorIntelligence(
                    max_speed=args.max_speed * 1.25,  # Predators are faster
                    max_force=args.max_force * 1.5,
                    perception_radius=args.perception_radius * 2.0  # Predators see twice as far
                )
            else:
                return PreyIntelligence(
                    max_speed=args.max_speed,
                    max_force=args.max_force,
                    perception_radius=args.perception_radius  # Prey have base perception
                )
    else:
        scenario = scenario_class(
            width=args.width,
            height=args.height,
            enable_wrapping=args.wrap
        )
        
        # Create intelligence factory for other scenarios
        def create_intelligence(bot_index: int) -> Intelligence:
            intelligence_class = load_class(INTELLIGENCES[args.intelligence])
            return intelligence_class(
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
    
    # Create and run simulation
    sim = Simulation(scenario, create_intelligence, args.num_bots)
    
    # Print simulation parameters
    print(f"\nRunning simulation with:")
    print(f"- Scenario: {args.scenario}")
    if args.scenario == 'predator_prey':
        print(f"- Predators: {len(scenario.predator_indices)}")
        print(f"- Prey: {args.num_bots - len(scenario.predator_indices)}")
    else:
        print(f"- Intelligence: {args.intelligence}")
    print(f"- Number of bots: {args.num_bots}")
    print(f"- World size: {args.width}x{args.height}")
    print(f"- World wrapping: {'enabled' if args.wrap else 'disabled'}")
    
    sim.run(frames=args.max_time if args.scenario == 'predator_prey' else 200)

if __name__ == "__main__":
    main() 