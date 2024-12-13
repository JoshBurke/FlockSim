import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict

class SpatialGrid:
    """A spatial partitioning system that divides space into a grid for efficient neighbor queries."""
    
    def __init__(self, width: float, height: float, cell_size: float):
        """Initialize the spatial grid.
        
        Args:
            width: Width of the world
            height: Height of the world
            cell_size: Size of each grid cell (should be >= max perception radius)
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))
        
        # Initialize empty grid
        self.clear()
    
    def clear(self):
        """Clear all objects from the grid."""
        # Use defaultdict to avoid key checks
        self.cells = defaultdict(list)
        # Track object cell locations for quick updates
        self.object_cells = {}
    
    def _get_cell_coords(self, position: np.ndarray) -> Tuple[int, int]:
        """Get grid cell coordinates for a position."""
        x = int(position[0] / self.cell_size)
        y = int(position[1] / self.cell_size)
        # Clamp to grid bounds
        x = max(0, min(x, self.cols - 1))
        y = max(0, min(y, self.rows - 1))
        return (x, y)
    
    def _get_neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get coordinates of neighboring cells (including diagonal and self)."""
        x, y = cell
        neighbors = []
        
        # Check all adjacent cells (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                # Only include cells within grid bounds
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def update_object(self, obj_id: int, new_pos: np.ndarray):
        """Update an object's position in the grid.
        
        Args:
            obj_id: Unique identifier for the object
            new_pos: New position of the object
        """
        new_cell = self._get_cell_coords(new_pos)
        old_cell = self.object_cells.get(obj_id)
        
        # If object has moved to a new cell
        if old_cell != new_cell:
            # Remove from old cell if it existed
            if old_cell is not None:
                self.cells[old_cell].remove(obj_id)
            
            # Add to new cell
            self.cells[new_cell].append(obj_id)
            self.object_cells[obj_id] = new_cell
    
    def get_potential_neighbors(self, position: np.ndarray) -> Set[int]:
        """Get all object IDs in cells adjacent to the given position.
        
        Args:
            position: Position to find neighbors for
            
        Returns:
            Set of object IDs that could be neighbors
        """
        cell = self._get_cell_coords(position)
        neighbor_cells = self._get_neighbor_cells(cell)
        
        # Collect all objects in neighboring cells
        neighbors = set()
        for nc in neighbor_cells:
            neighbors.update(self.cells[nc])
        
        return neighbors 