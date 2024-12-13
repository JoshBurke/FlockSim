"""Central registry for available scenarios and intelligences."""

# Registry of available scenarios and intelligences
SCENARIOS = {
    'free_roam': 'src.scenarios.free_roam.FreeRoamScenario'
}

INTELLIGENCES = {
    'flocking': 'src.intelligences.flocking.FlockingIntelligence'
}

def get_choices(registry: dict) -> list:
    """Get list of available choices from a registry."""
    return list(registry.keys()) 