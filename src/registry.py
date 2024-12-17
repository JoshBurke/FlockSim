"""Central registry for available scenarios and intelligences."""

# Registry of available scenarios and intelligences
SCENARIOS = {
    'free_roam': 'src.scenarios.free_roam.FreeRoamScenario',
    'predator_prey': 'src.scenarios.predator_prey.PredatorPreyScenario'
}

INTELLIGENCES = {
    'flocking': 'src.intelligences.flocking.FlockingIntelligence',
    'predator': 'src.intelligences.predator_prey.PredatorIntelligence',
    'prey': 'src.intelligences.predator_prey.PreyIntelligence'
}

def get_choices(registry: dict) -> list:
    """Get list of available choices from a registry."""
    return list(registry.keys()) 