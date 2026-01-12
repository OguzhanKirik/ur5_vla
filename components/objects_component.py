"""
Objects Component - Scene objects and table

Handles object spawning, manipulation, and scene setup.
"""

import numpy as np
import pybullet as p
import pybullet_data


class ObjectsComponent:
    """Scene objects manager"""
    
    def __init__(self):
        """Initialize objects component"""
        self.objects = []  # List of object IDs
        self.table_id = None
        self.plane_id = None
        self.container_box_id = None  # Storage box for placing objects
    
    def setup_basic_scene(self):
        """Setup basic scene with plane"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        return self.plane_id
    
    def create_table(self, position, size, color=[0.6, 0.4, 0.2, 1], mass=0):
        """
        Create a box-shaped table.
        
        Args:
            position: [x, y, z] center position
            size: [width, depth, height] half-extents
            color: [r, g, b, a] RGBA color
            mass: table mass (0 for static)
            
        Returns:
            table_id: PyBullet body ID
        """
        # Create collision and visual shapes
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        
        # Create table body
        self.table_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        
        print(f"Table created at {position}, size: {size}")
        return self.table_id
    
    def create_container_box(self, position, size=[0.15, 0.15, 0.05], wall_thickness=0.01, color=[0.3, 0.3, 0.3, 1]):
        """
        Create a container box (open-top box) for placing objects.
        Position should be on the table surface next to the robot.
        
        Args:
            position: [x, y, z] center position of the box base
            size: [width, depth, height] inner dimensions
            wall_thickness: thickness of box walls
            color: [r, g, b, a] RGBA color
            
        Returns:
            container_id: PyBullet body ID of the container base
        """
        width, depth, height = size
        thick = wall_thickness
        
        # Create container as a compound shape with bottom and 4 walls
        # Bottom
        bottom_half = [width/2 + thick, depth/2 + thick, thick/2]
        bottom_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=bottom_half)
        bottom_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=bottom_half, rgbaColor=color)
        
        bottom_pos = [position[0], position[1], position[2]]
        
        # Create base with bottom
        self.container_box_id = p.createMultiBody(
            baseMass=0,  # Static container
            baseCollisionShapeIndex=bottom_col,
            baseVisualShapeIndex=bottom_vis,
            basePosition=bottom_pos
        )
        
        # Add 4 walls as separate bodies (simpler approach)
        wall_height = height / 2
        
        # Front wall (along X, at negative Y)
        front_wall_size = [width/2 + thick, thick/2, wall_height]
        front_wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=front_wall_size)
        front_wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=front_wall_size, rgbaColor=color)
        front_wall_pos = [position[0], position[1] - depth/2, position[2] + thick + wall_height]
        p.createMultiBody(0, front_wall_col, front_wall_vis, front_wall_pos)
        
        # Back wall (along X, at positive Y)
        back_wall_pos = [position[0], position[1] + depth/2, position[2] + thick + wall_height]
        p.createMultiBody(0, front_wall_col, front_wall_vis, back_wall_pos)
        
        # Left wall (along Y, at negative X)
        side_wall_size = [thick/2, depth/2 + thick, wall_height]
        side_wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_wall_size)
        side_wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=side_wall_size, rgbaColor=color)
        left_wall_pos = [position[0] - width/2, position[1], position[2] + thick + wall_height]
        p.createMultiBody(0, side_wall_col, side_wall_vis, left_wall_pos)
        
        # Right wall (along Y, at positive X)
        right_wall_pos = [position[0] + width/2, position[1], position[2] + thick + wall_height]
        p.createMultiBody(0, side_wall_col, side_wall_vis, right_wall_pos)
        
        print(f"Container box created at {position}, inner size: {size}")
        return self.container_box_id
    
    def get_table_surface_height(self):
        """Get the Z-coordinate of the table's top surface"""
        if self.table_id is None:
            return 0.0
        
        aabb = p.getAABB(self.table_id)
        return aabb[1][2]  # Maximum Z coordinate
    
    def add_box(self, position, size, color, mass=0.08):
        """
        Add a box object.
        
        Args:
            position: [x, y, z]
            size: box half-extents (single value or [x, y, z])
            color: [r, g, b, a]
            mass: object mass
            
        Returns:
            object_id: PyBullet body ID
        """
        if isinstance(size, (int, float)):
            half_extents = [size/2, size/2, size/2]
        else:
            half_extents = [s/2 for s in size]
        
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        
        # Set high friction for better gripping
        p.changeDynamics(obj_id, -1, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.01)
        
        self.objects.append(obj_id)
        return obj_id
    
    def add_sphere(self, position, radius, color, mass=0.05):
        """
        Add a sphere object.
        
        Args:
            position: [x, y, z]
            radius: sphere radius
            color: [r, g, b, a]
            mass: object mass
            
        Returns:
            object_id: PyBullet body ID
        """
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        
        # Set high friction for better gripping
        p.changeDynamics(obj_id, -1, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.01)
        
        self.objects.append(obj_id)
        return obj_id
    
    def add_cylinder(self, position, radius, height, color, mass=0.05):
        """
        Add a cylinder object.
        
        Args:
            position: [x, y, z]
            radius: cylinder radius
            height: cylinder height
            color: [r, g, b, a]
            mass: object mass
            
        Returns:
            object_id: PyBullet body ID
        """
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position
        )
        
        # Set high friction for better gripping
        p.changeDynamics(obj_id, -1, lateralFriction=2.0, spinningFriction=0.1, rollingFriction=0.01)
        
        self.objects.append(obj_id)
        return obj_id
    
    def spawn_graspable_objects(self, table_height, workspace_bounds=None, randomize=False):
        """
        Spawn a set of graspable objects on the table.
        
        Args:
            table_height: Z-coordinate of table surface
            workspace_bounds: [[x_min, x_max], [y_min, y_max]] or None for defaults
            randomize: Whether to randomize object positions within workspace
            
        Returns:
            object_ids: list of spawned object IDs
        """
        # Default workspace if not specified
        if workspace_bounds is None:
            workspace_bounds = [[0.3, 0.7], [0.3, 0.7]]
        
        x_min, x_max = workspace_bounds[0]
        y_min, y_max = workspace_bounds[1]
        
        # Clear existing objects
        self.clear_objects()
        
        # Define object configurations
        if randomize:
            # Randomize positions within workspace with minimum separation
            positions = []
            min_distance = 0.15  # Minimum 15cm between objects
            max_attempts = 50
            
            for obj_idx in range(4):  # 4 objects
                for attempt in range(max_attempts):
                    x = np.random.uniform(x_min, x_max)
                    y = np.random.uniform(y_min, y_max)
                    
                    # Check distance to all previous objects
                    valid = True
                    for prev_pos in positions:
                        dist = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                        if dist < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        positions.append([x, y])
                        break
                else:
                    # If couldn't find valid position after max_attempts, use fallback
                    print(f"Warning: Could not find valid position for object {obj_idx}, using fallback")
                    positions.append([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
            
            configs = [
                ('box', 0.03, [positions[0][0], positions[0][1], table_height + 0.015], [1, 0, 0, 1]),  # Red cube
                ('sphere', 0.025, [positions[1][0], positions[1][1], table_height + 0.025], [0, 1, 0, 1]),  # Green sphere
                ('cylinder', (0.02, 0.05), [positions[2][0], positions[2][1], table_height + 0.025], [0, 0, 1, 1]),  # Blue cylinder
                ('box', 0.03, [positions[3][0], positions[3][1], table_height + 0.015], [0, 0, 1, 1]),  # Blue cube
            ]
        else:
            configs = [
                # (type, params, position, color)
                ('box', 0.03, [0.5, 0.4, table_height + 0.015], [1, 0, 0, 1]),  # Red cube
                ('sphere', 0.025, [0.3, 0.3, table_height + 0.025], [0, 1, 0, 1]),  # Green sphere
                ('cylinder', (0.02, 0.05), [0.7, 0.5, table_height + 0.025], [0, 0, 1, 1]),  # Blue cylinder
                ('box', 0.03, [0.6, 0.3, table_height + 0.015], [0, 0, 1, 1]),  # Blue cube
            ]
        
        spawned_ids = []
        for config in configs:
            obj_type = config[0]
            params = config[1]
            position = config[2]
            color = config[3]
            
            if obj_type == 'box':
                obj_id = self.add_box(position, params, color)
            elif obj_type == 'sphere':
                obj_id = self.add_sphere(position, params, color)
            elif obj_type == 'cylinder':
                radius, height = params
                obj_id = self.add_cylinder(position, radius, height, color)
            
            spawned_ids.append(obj_id)
        
        print(f"Spawned {len(spawned_ids)} graspable objects")
        return spawned_ids
    
    def get_object_poses(self):
        """
        Get positions and orientations of all objects.
        
        Returns:
            poses: list of (position, orientation) tuples
        """
        poses = []
        for obj_id in self.objects:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            poses.append((np.array(pos), np.array(orn)))
        return poses
    
    def get_nearest_object(self, reference_point):
        """
        Find the nearest object to a reference point.
        
        Args:
            reference_point: [x, y, z] reference position
            
        Returns:
            nearest_id: object ID of nearest object
            distance: distance to nearest object
        """
        min_dist = float('inf')
        nearest_id = None
        
        ref_point = np.array(reference_point)
        for obj_id in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = np.linalg.norm(ref_point - np.array(obj_pos))
            if dist < min_dist:
                min_dist = dist
                nearest_id = obj_id
        
        return nearest_id, min_dist
    
    def clear_objects(self):
        """Remove all spawned objects (not table or plane)"""
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects.clear()
    
    def clear_all(self):
        """Remove all objects including table and plane"""
        self.clear_objects()
        
        if self.table_id is not None:
            p.removeBody(self.table_id)
            self.table_id = None
        
        if self.plane_id is not None:
            p.removeBody(self.plane_id)
            self.plane_id = None
