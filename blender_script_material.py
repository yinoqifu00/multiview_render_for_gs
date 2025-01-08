"""
Blender script to render images of 3D models.

Modified from https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py

Example usage:
    blender -b -P blender_script_material.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --render_space VIEW \

"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple, Literal
from mathutils import Vector, Matrix, Euler
import numpy as np
import bpy
import logging
import threading
import json


logger = logging.getLogger("bpy")

logger.setLevel(logging.WARNING)


def generate_random_numbers(total_sum, num_elements, min_value):
    # subtract the minimum value from the total sum
    adjusted_sum = total_sum - num_elements * min_value
    
    # generate random numbers and normalize them to sum to adjusted_sum
    random_numbers = np.random.rand(num_elements)
    random_numbers = (random_numbers / np.sum(random_numbers)) * adjusted_sum
    
    # add the minimum value to each number
    random_numbers += min_value
    
    return random_numbers



class BlenderRendering():
    def __init__(self, args) -> None:
        
        self.args = args

        context = bpy.context
        self.scene = context.scene
        self.render = self.scene.render
        self.scene.use_nodes = True
        self.tree = self.scene.node_tree
        self.links = self.tree.links
        self.cam_locations = None

        self.render.engine = args.engine
        self.render.image_settings.file_format = "PNG"
        self.render.image_settings.color_mode = "RGBA"
        self.render.resolution_x = args.resolution
        self.render.resolution_y = args.resolution
        self.render.resolution_percentage = 100

        self.scene.cycles.device = "GPU"
        self.scene.cycles.samples = 128
        self.scene.cycles.diffuse_bounces = 1
        self.scene.cycles.glossy_bounces = 1
        self.scene.cycles.transparent_max_bounces = 3
        self.scene.cycles.transmission_bounces = 3
        self.scene.cycles.filter_width = 0.01
        self.scene.cycles.use_denoising = True
        self.scene.render.film_transparent = True
        self.scene.render.dither_intensity = 0.0

        self.scene.render.use_persistent_data = True # use persistent data for speed up
        context.view_layer.use_pass_normal = True # for normal rendering
        context.view_layer.use_pass_z = True # for depth rendering
        context.view_layer.use_pass_position = True
        context.view_layer.pass_alpha_threshold = 0.5

        # set the device_type
        cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
        cycles_preferences.compute_device_type = "CUDA"  # or "OPENCL"
        cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
        for device in cuda_devices:
            device.use = True

    def compose_RT(self, R, T):
        return np.hstack((R, T.reshape(-1, 1)))

    def sample_point_on_sphere(self, radius: float) -> Tuple[float, float, float]:
        theta = random.random() * 2 * math.pi
        phi = math.acos(2 * random.random() - 1)
        return (
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi),
        )

    def sample_spherical(self, radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
        correct = False
        while not correct:
            vec = np.random.uniform(-1, 1, 3)
    #         vec[2] = np.abs(vec[2])
            radius = np.random.uniform(radius_min, radius_max, 1)
            vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
            if maxz > vec[2] > minz:
                correct = True
        return vec

    def set_camera_location(self, camera, i, option: str):
        assert option in ['fixed', 'random', 'front']

        if option == 'fixed':
            locations = [
                [ 0, -1,  0],
                [-1, -1,  0],
                [-1,  0,  0],
                [-1,  1,  0],
                [ 0,  1,  0],
                [ 1,  1,  0],
                [ 1,  0,  0],
                [ 1, -1,  0]
            ]
            vec = locations[i]
            radius = 2
            vec = vec / np.linalg.norm(vec, axis=0) * radius
            x, y, z = vec
        elif option == 'random':
            # from https://blender.stackexchange.com/questions/18530/
            x, y, z = self.sample_spherical(radius_min=1.9, radius_max=2.6, maxz=1.60, minz=-0.75)
        elif option == 'front':
            x, y, z = 0, -np.random.uniform(1.9, 2.6, 1)[0], 0

        camera.location = x, y, z

        # adjust orientation
        direction = - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        return camera
    

    def _create_light(
        self,
        name: str,
        light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float,
        use_shadow: bool = False,
        use_contact_shadow: bool = False,
        specular_factor: float = 1.0,
        radius: float = 0.25,
    ):
        """Creates a light object.

        Args:
            name (str): Name of the light object.
            light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
            location (Tuple[float, float, float]): Location of the light.
            rotation (Tuple[float, float, float]): Rotation of the light.
            energy (float): Energy of the light.
            use_shadow (bool, optional): Whether to use shadows. Defaults to False.
            specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

        Returns:
            bpy.types.Object: The light object.
        """

        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_object = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = location
        light_object.rotation_euler = rotation
        light_data.use_shadow = use_shadow
        light_data.use_contact_shadow = use_contact_shadow
        light_data.specular_factor = specular_factor
        light_data.energy = energy
        if light_type=="SUN":
            light_data.angle=0.5
        if light_type=="POINT":
            light_data.shadow_soft_size = radius
        if light_type=="AREA":
            light_data.size = radius

        return light_object



    def randomize_lighting(self):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """
        # Add random angle offset in 0-90
        angle_offset = random.uniform(0, math.pi / 2)
        

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # Create key light
        key_light = self._create_light(
            name="Key_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, -0.785398 + angle_offset), # 45 0 -45
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=False,
        )

        # Create rim light
        rim_light = self._create_light(
            name="Rim_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(-0.785398, 0, -3.92699 + angle_offset), # -45 0 -225
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=False,
        )

        # Create fill light
        fill_light = self._create_light(
            name="Fill_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, 2.35619 + angle_offset), # 45 0 135
            energy=random.choice([2, 3, 3.5]),
        )

        # Create small light
        small_light1 = self._create_light(
            name="S1_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 0.785398 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        small_light2 = self._create_light(
            name="S2_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 3.92699 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        # Create bottom light
        bottom_light = self._create_light(
            name="Bottom_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(3.14159, 0, 0), # 180 0 0
            energy=random.choice([1, 2, 3]),
        )

        return dict(
            key_light=key_light,
            fill_light=fill_light,
            rim_light=rim_light,
            bottom_light=bottom_light,
            small_light1=small_light1,
            small_light2=small_light2,
        )
    
    
    def randomize_lighting(self):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """
        # Add random angle offset in 0-90
        angle_offset = random.uniform(0, math.pi / 2)
        

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # Create key light
        key_light = self._create_light(
            name="Key_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, -0.785398 + angle_offset), # 45 0 -45
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=False,
        )

        # Create rim light
        rim_light = self._create_light(
            name="Rim_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(-0.785398, 0, -3.92699 + angle_offset), # -45 0 -225
            energy=random.choice([2.5, 3.25, 4]),
            use_shadow=False,
        )

        # Create fill light
        fill_light = self._create_light(
            name="Fill_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(0.785398, 0, 2.35619 + angle_offset), # 45 0 135
            energy=random.choice([2, 3, 3.5]),
        )

        # Create small light
        small_light1 = self._create_light(
            name="S1_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 0.785398 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        small_light2 = self._create_light(
            name="S2_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(1.57079, 0, 3.92699 + angle_offset), # 90 0 45
            energy=random.choice([0.25, 0.5, 1]),
        )

        # Create bottom light
        bottom_light = self._create_light(
            name="Bottom_Light",
            light_type="SUN",
            location=(0, 0, 0),
            rotation=(3.14159, 0, 0), # 180 0 0
            energy=random.choice([1, 2, 3]),
        )

        return dict(
            key_light=key_light,
            fill_light=fill_light,
            rim_light=rim_light,
            bottom_light=bottom_light,
            small_light1=small_light1,
            small_light2=small_light2,
        )
    

    def randomize_point_lighting(self, cam_location, num_lights=None):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        nodes = bpy.context.scene.world.node_tree.nodes
        background_node = nodes.get("Background")
        background_node.inputs['Strength'].default_value = 0.1

        radius = random.uniform(4, 5)
        origin = Vector((0, 0, 0))
        up_vector = Vector((0, 0, 1))
        cam_location = Vector(cam_location)
        cam_direction = (cam_location - origin).normalized()
        rotation_matrix = up_vector.rotation_difference(cam_direction).to_matrix().to_4x4()

        num_lights = random.choice([1, 2, 3]) if num_lights is None else num_lights
        light_energys = generate_random_numbers(2400, num_lights, 300)

        for i in range(num_lights):
            # Generate random spherical coordinates for the hemisphere
            theta = random.uniform(0, 2 * math.pi)  # azimuthal angle (0 to 2*pi)
            phi = random.uniform(0, math.pi / 3)    # polar angle (0 to pi/2 for hemisphere)

            # Convert spherical coordinates to Cartesian coordinates in the local frame
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            # Transform the local position to the global frame
            local_position = Vector((x, y, z))
            light_position = rotation_matrix @ local_position + origin

            # Set light rotation to look at the origin (0, 0, 0)
            light_direction = (light_position - origin).normalized()
            light_direction = light_direction.to_track_quat('Z', 'Y').to_euler()

            self._create_light(
                name=f"Point_Light_{i}",
                light_type="POINT",
                location=light_position,
                rotation=light_direction,
                energy=light_energys[i],
                use_shadow=False,
                radius=random.uniform(0.25, 1.0),
            )

        return
    
    def randomize_area_lighting(self, cam_location, num_lights=1):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        nodes = bpy.context.scene.world.node_tree.nodes
        background_node = nodes.get("Background")
        background_node.inputs['Strength'].default_value = 0.1

        radius = random.uniform(4, 5)
        origin = Vector((0, 0, 0))
        up_vector = Vector((0, 0, 1))
        cam_location = Vector(cam_location)
        cam_direction = (cam_location - origin).normalized()
        rotation_matrix = up_vector.rotation_difference(cam_direction).to_matrix().to_4x4()


        for i in range(num_lights):
            # Generate random spherical coordinates for the hemisphere
            theta = random.uniform(0, 2 * math.pi)  # azimuthal angle (0 to 2*pi)
            phi = random.uniform(0, math.pi / 3)    # polar angle (0 to pi/2 for hemisphere)

            # Convert spherical coordinates to Cartesian coordinates in the local frame
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            # Transform the local position to the global frame
            local_position = Vector((x, y, z))
            light_position = rotation_matrix @ local_position + origin

            # Set light rotation to look at the origin (0, 0, 0)
            light_direction = (light_position - origin).normalized()
            light_direction = light_direction.to_track_quat('Z', 'Y').to_euler()

            self._create_light(
                name=f"Point_Light_{i}",
                light_type="AREA",
                location=light_position,
                rotation=light_direction,
                energy=random.uniform(1000, 2000),
                use_shadow=False,
                radius=random.uniform(3, 10)
            )

        return
    
    def randomize_env_lighting(self):
        """Randomizes the lighting in the scene.

        Returns:
            Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
                "key_light", "fill_light", "rim_light", and "bottom_light".
        """

        # Clear existing lights
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()
        
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(0, 0, 1), scale=(1, 1, 1), rotation=(0.0, 0.0, 0.0))
        #bpy.context.object.data.energy = 10
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(0, 0, -1), scale=(1, 1, 1), rotation=(np.pi, 0.0, 0.0))
        #bpy.context.object.data.energy = 10
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(0, 1, 0), scale=(1, 1, 1), rotation=(-np.pi / 2, 0.0, 0.0))
        #bpy.context.object.data.energy = 10
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(0, -1, 0), scale=(1, 1, 1), rotation=(np.pi / 2, 0.0, 0.0))
        #bpy.context.object.data.energy = 10
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(1, 0, 0), scale=(1, 1, 1), rotation=(0.0, np.pi / 2, 0.0))
        #bpy.context.object.data.energy = 10
        #bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(-1, 0, 0), scale=(1, 1, 1), rotation=(0.0, -np.pi / 2, 0.0))
        #bpy.context.object.data.energy = 10


        # Reset the background strength
        nodes = bpy.context.scene.world.node_tree.nodes
        background_node = nodes.get("Background")
        background_node.inputs['Strength'].default_value = 2#random.uniform(0.5, 3.0)
        #world_node_tree = bpy.context.scene.world.node_tree 
        #world_node_tree.nodes.clear()

        #path_to_image = '/etc/blender/4.2/datafiles/studiolights/world/city.exr'
        #image_obj = bpy.data.images.load(path_to_image)
        
        #environment_texture_node = world_node_tree.nodes.new(type="ShaderNodeTexEnvironment") 
        #environment_texture_node.image = image_obj
        

        #background_node = world_node_tree.nodes.new(type="ShaderNodeBackground" ) 
        #background_node.inputs["Strength"].default_value =2.0

        #world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld" ) 

        #from_node=	environment_texture_node	
        #to_node = background_node
        #world_node_tree.links.new(from_node.outputs["Color"],to_node.inputs["Color"])
        #from_node = background_node
        #to_node = world_output_node
        #world_node_tree.links.new(from_node.outputs["Background"],to_node.inputs["Surface"])
        bpy.context.scene.render.film_transparent = True
        #bpy.context.scene.world.cycles_visibility.scatter = False
        #bpy.context.scene.world.cycles_visibility.transmission = False
        #bpy.context.scene.world.cycles_visibility.glossy = False
        #bpy.context.scene.world.cycles_visibility.diffuse = False
        #bpy.context.scene.world.cycles_visibility.camera = False

        return
    
    
    def add_lighting(self, option: str) -> None:
        assert option in ['fixed', 'random']
        
        # delete the default light
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
        
        # add a new light
        if option == 'fixed':
            #Make light just directional, disable shadows.
            bpy.ops.object.light_add(type='SUN')
            light = bpy.context.object
            light.name = 'Light'
            light.data.use_shadow = False
            # Possibly disable specular shading:
            light.data.specular_factor = 1.0
            light.data.energy = 5.0

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light2 = bpy.context.object
            light2.name = 'Light2'
            light2.data.use_shadow = False
            light2.data.specular_factor = 1.0
            light2.data.energy = 3 #0.015
            bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light2'].rotation_euler[0] += 180

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light3 = bpy.context.object
            light3.name = 'Light3'
            light3.data.use_shadow = False
            light3.data.specular_factor = 1.0
            light3.data.energy = 3 #0.015
            bpy.data.objects['Light3'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light3'].rotation_euler[0] += 90

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light4'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light4'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light4'].rotation_euler[0] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light5'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light5'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light5'].rotation_euler[1] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light6'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light6'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light6'].rotation_euler[1] += 90

        elif option == 'random':
            bpy.ops.object.light_add(type="AREA")
            light = bpy.data.lights["Area"]
            light.energy = random.uniform(500000, 600000)
            bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

            # set light scale
            bpy.data.objects["Area"].scale[0] = 200
            bpy.data.objects["Area"].scale[1] = 200
            bpy.data.objects["Area"].scale[2] = 200


    def add_lighting(self, option: str) -> None:
        assert option in ['fixed', 'random']
        
        # delete the default light
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
        
        # add a new light
        if option == 'fixed':
            #Make light just directional, disable shadows.
            bpy.ops.object.light_add(type='SUN')
            light = bpy.context.object
            light.name = 'Light'
            light.data.use_shadow = False
            # Possibly disable specular shading:
            light.data.specular_factor = 1.0
            light.data.energy = 5.0

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light2 = bpy.context.object
            light2.name = 'Light2'
            light2.data.use_shadow = False
            light2.data.specular_factor = 1.0
            light2.data.energy = 3 #0.015
            bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light2'].rotation_euler[0] += 180

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light3 = bpy.context.object
            light3.name = 'Light3'
            light3.data.use_shadow = False
            light3.data.specular_factor = 1.0
            light3.data.energy = 3 #0.015
            bpy.data.objects['Light3'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light3'].rotation_euler[0] += 90

            #Add another light source so stuff facing away from light is not completely dark
            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light4'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light4'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light4'].rotation_euler[0] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light5'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light5'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light5'].rotation_euler[1] += -90

            bpy.ops.object.light_add(type='SUN')
            light4 = bpy.context.object
            light4.name = 'Light6'
            light4.data.use_shadow = False
            light4.data.specular_factor = 1.0
            light4.data.energy = 3 #0.015
            bpy.data.objects['Light6'].rotation_euler = bpy.data.objects['Light'].rotation_euler
            bpy.data.objects['Light6'].rotation_euler[1] += 90

        elif option == 'random':
            bpy.ops.object.light_add(type="AREA")
            light = bpy.data.lights["Area"]
            light.energy = random.uniform(500000, 600000)
            bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
            bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

            # set light scale
            bpy.data.objects["Area"].scale[0] = 200
            bpy.data.objects["Area"].scale[1] = 200
            bpy.data.objects["Area"].scale[2] = 200


    def reset_scene(self) -> None:
        """Resets the scene to a clean state."""
        # delete everything that isn't part of a camera or a light
        for obj in bpy.data.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        # delete all the materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        # delete all the textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        # delete all the images
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)



    # load the glb model
    def load_object(self, object_path: str) -> None:
        """Loads a glb model into the scene."""
        if object_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=object_path)
        elif object_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=object_path)
        elif object_path.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=object_path)
        else:
            raise ValueError(f"Unsupported file type: {object_path}")




    def scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in self.scene_meshes() if single_obj is None else [single_obj]:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        
        # bbox_min=(-1.05,-1.05,-1.05)
        # bbox_max=(1.05,1.05,1.05)
        return Vector(bbox_min), Vector(bbox_max)
    


    def scene_root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj


    def scene_meshes(self):
        for obj in bpy.context.scene.objects.values():
            if isinstance(obj.data, (bpy.types.Mesh)):
                yield obj


    def normalize_scene(self, box_scale: float):
        bbox_min, bbox_max = self.scene_bbox()
    
        scale = box_scale / max(bbox_max - bbox_min)
        for obj in self.scene_root_objects():
            obj.scale = obj.scale * scale
        # Apply scale to matrix_world.
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        
        #offset = -(bbox_min + bbox_max) / 2
        #for obj in self.scene_root_objects():
            #obj.matrix_world.translation += offset

        
        

        bpy.ops.object.select_all(action="DESELECT")

        # with open('412edafac9b7473d975bdc2d9303a877.json','w') as f:
        #     json.dump([{
        #         'scale': f'{scale}',
        #         'offset': f'{offset}'
        #     }],f)
        #bpy.ops.export_scene.obj(filepath='bowl2.obj', use_selection=False)
        

    def normalize_scene_sphere(self, radius: float):
        self.normalize_scene(1)

        bbox_min, bbox_max, max_dist = self.scene_bbox()
        center = (bbox_min + bbox_max) / 2

        scale = radius / max_dist
        for obj in self.scene_root_objects():
            obj.scale = obj.scale * scale

        bpy.context.view_layer.update()

        bbox_min, bbox_max, _ = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2

        

        #for obj in self.scene_root_objects():
            #obj.matrix_world.translation += offset

        # bpy.ops.export_scene.obj(
        #     filepath="teapot.obj",
        #     use_selection=True,
        # )

        
        bpy.ops.object.select_all(action="DESELECT")


    def setup_camera(self):
        cam = self.scene.objects["Camera"]
        cam.location = (0, 4, 0)
        cam.data.lens = 50
        cam.data.sensor_width = 32
        cam.data.sensor_height = 32  # affects instrinsics calculation, should be set explicitly
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        return cam, cam_constraint


    def get_bsdf_node_from_material(self, mat):
        nodes = mat.node_tree.nodes

        node_types = []
        principled_bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled_bsdf_node.name = 'New BSDF'
        principled_bsdf_node.label = 'New BSDF'
        for node in nodes:
            node_types.append(node.type)
            if "BSDF" in node.type and 'New BSDF' not in node.name:
                # Create a new Principled BSDF node and link the color input
                # FIX: the input name may not be Color
                is_BSDF = True
                assert (
                    "Color" in node.inputs or "Base Color" in node.inputs 
                ), f"BSDF node {node.type} does not have a 'Color' input"

                color_input_name = 'Base Color' if node.type == 'BSDF_PRINCIPLED' else 'Color'

                if node.inputs[color_input_name].is_linked:
                    color_link = node.inputs[color_input_name].links[0].from_socket
                    mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Base Color"])
                else:
                    if not principled_bsdf_node.inputs["Base Color"].is_linked:
                        principled_bsdf_node.inputs["Base Color"].default_value = node.inputs[
                            color_input_name
                        ].default_value

                if "Roughness" in node.inputs:
                    if node.inputs["Roughness"].is_linked:
                        color_link = node.inputs["Roughness"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Roughness"])
                    else:
                        principled_bsdf_node.inputs["Roughness"].default_value = node.inputs[
                            "Roughness"
                        ].default_value

                if "Metallic" in node.inputs:
                    if node.inputs["Metallic"].is_linked:
                        color_link = node.inputs["Metallic"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Metallic"])
                    else:
                        principled_bsdf_node.inputs["Metallic"].default_value = node.inputs[
                            "Metallic"
                        ].default_value

                if "Normal" in node.inputs:
                    if node.inputs["Normal"].is_linked:
                        color_link = node.inputs["Normal"].links[0].from_socket
                        mat.node_tree.links.new(color_link, principled_bsdf_node.inputs["Normal"])
                    else:
                        principled_bsdf_node.inputs["Normal"].default_value = node.inputs[
                            "Normal"
                        ].default_value

        bump_linked = nodes['New BSDF'].inputs['Normal'].is_linked
        is_scan = set(['EMISSION', 'LIGHT_PATH', 'BSDF_TRANSPARENT', 'MIX_SHADER', 'TEX_IMAGE']).issubset(set(node_types))
        return bump_linked


    def assign_material_value(
        self, node_tree, combine_node, bsdf_node, channel_name, material_name, rand=False
        ):
            """
            Assigns the specified material property to a channel of the CombineRGB node.
            param combine_node: The CombineRGB node to which the material property will be assigned.
            param bsdf_node: The BSDF node from which the material property is sourced.
            param channel_name: The channel name in the CombineRGB node (e.g., "R", "G", or "B").
            param material_name: The name of the material property in the BSDF node (e.g., "Roughness", "Metallic").
            """
            if material_name not in bsdf_node.inputs:
                rand = True


            if rand:
                # TODO unused
                rand_val = random.choice([0.05, 0.2, 0.5, 0.8, 1.0])
                combine_node.inputs[channel_name].default_value = rand_val
                print(mesh_name, "Random", material_name, "value:", rand_val)

                return

            if bsdf_node.inputs[material_name].is_linked:
                input_link = bsdf_node.inputs[material_name].links[0].from_socket
                combine_node.inputs[channel_name].default_value = 1.0  # ensure full value if linked
                node_tree.links.new(input_link, combine_node.inputs[channel_name])
            else:
                # assign a single value
                combine_node.inputs[channel_name].default_value = bsdf_node.inputs[
                    material_name
                ].default_value


    def get_material_nodes(self):

        self.original_mats = []
        is_scan = []
        for obj in bpy.context.scene.objects:
            # each obj is a sub-mesh
            if not (obj.type == "MESH"):
                continue

            mesh_name = obj.name
            if obj.data.materials:
                mat = obj.data.materials[0]
            else:
                mat = bpy.data.materials.new(name=f"{mesh_name}_new_material")
                mat.use_nodes = True
            is_scan.append(self.get_bsdf_node_from_material(mat=mat))
            
            self.original_mats.append(mat)

    def update_material_nodes(self, mode, add_noise=False, rand=False):
        """
        Update the material nodes to either display the base color (albedo) or a combination of roughness and metallic values.
        param use_albedo: If True, sets up for albedo map; otherwise, sets up for roughness-metallic map.
        param rand: If True, randomizes the roughness and metallic values.
        """
        #for obj in bpy.data.objects:
        #    if obj.type == "MESH":
        #        obj.data.materials.clear()
        mat_id = 0
        for mat in bpy.data.materials:
            # each obj is a sub-mesh
            #if not (obj.type == "MESH"):
            #    continue

            #mesh_name = obj.name

            #mat = self.original_mats[mat_id]
            #obj.data.materials.clear()
            #obj.data.materials.append(mat)
            mat_id = mat_id + 1

            # process the new material
            # for mat in obj.data.materials:
            if not (mat and mat.node_tree):
                continue
            
            nodes = mat.node_tree.nodes
            for node in nodes:
                print(node.name)
            principled_bsdf_node = nodes['Principled BSDF']
            #if not principled_bsdf_node:
            #    continue  # Skip this mesh if no suitable BSDF node is found

            emission_node = nodes.new(type="ShaderNodeEmission")
            if mode == "albedo":
                # Link albedo to emission
                if principled_bsdf_node.inputs["Base Color"].is_linked:
                    input_link = principled_bsdf_node.inputs["Base Color"].links[0].from_socket
                    mat.node_tree.links.new(input_link, emission_node.inputs["Color"])
                else:
                    emission_node.inputs["Color"].default_value = principled_bsdf_node.inputs[
                        "Base Color"
                    ].default_value
            elif mode == "roughness_metallic":
                # mode is like "roughness_metallic"
                # Create a mix of roughness (G) and metallic (B) values
                combine_node = nodes.new(type="ShaderNodeCombineRGB")
                mat_fn = lambda ch, name: self.assign_material_value( mat.node_tree,
                    combine_node, principled_bsdf_node, ch, name, rand=rand
                )

                combine_node.inputs["R"].default_value = 1  # R is fixed
                # mat_fn("R", "Specular")
                mat_fn("G", "Roughness")
                mat_fn("B", "Metallic")
                mat.node_tree.links.new(
                    combine_node.outputs["Image"], emission_node.inputs["Color"]
                )
            elif mode == "bump":
                if principled_bsdf_node.inputs['Normal'].is_linked:
                    normal_map_node = principled_bsdf_node.inputs['Normal'].links[0].from_node
                    if normal_map_node.type == 'NORMAL_MAP' and normal_map_node.space == 'TANGENT':
                        material_node = normal_map_node.inputs['Color'].links[0].from_node
                        if material_node.type == 'TEX_IMAGE':
                            mat.node_tree.links.new(
                                material_node.outputs["Color"], emission_node.inputs["Color"]
                            )
                        else:
                            emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                    else:
                        emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                else:
                    emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
            elif mode == "position":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Position'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])
            elif mode == "normal":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')
                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                # Create vector transform node to convert normals from world to camera space
                vector_transform_node = nodes.new(type='ShaderNodeVectorTransform')
                vector_transform_node.vector_type = 'NORMAL'
                vector_transform_node.convert_from = 'WORLD'
                vector_transform_node.convert_to = 'CAMERA'

                mat.node_tree.links.new(geometry_node.outputs['Normal'], vector_transform_node.inputs[0])

                mat.node_tree.links.new(vector_transform_node.outputs[0], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])

            if mode == 'rendering':
                #if add_noise:
                    if principled_bsdf_node.inputs['Roughness'].is_linked:
                        input_link = principled_bsdf_node.inputs['Roughness'].links[0]
                        mat.node_tree.links.remove(input_link)
                        add_node = nodes.new(type='ShaderNodeMath')
                        add_node.operation = 'ADD'
                        add_node.use_clamp = True
                        principled_bsdf_node.inputs['Roughness'].default_value = 100
                        #mat.node_tree.links.new(input_link, add_node.inputs[0])
                        #mat.node_tree.links.new(add_node.outputs[0], principled_bsdf_node.inputs['Roughness'])
                    else:
                        # assign a single value
                        principled_bsdf_node.inputs[
                            'Roughness'
                        ].default_value = 100

                    if principled_bsdf_node.inputs['Metallic'].is_linked:
                        input_link = principled_bsdf_node.inputs['Metallic'].links[0]
                        mat.node_tree.links.remove(input_link)
                        principled_bsdf_node.inputs['Metallic'].default_value = 0
                    else:
                        # assign a single value
                        principled_bsdf_node.inputs[
                            'Metallic'
                        ].default_value = 0

            else:
                # Connect emission to material output
                mat.node_tree.links.new(
                    emission_node.outputs["Emission"],
                    nodes["Material Output"].inputs["Surface"],
                )
            if 'glass' in mat.name:
                 #bpy.data.materials.remove(mat)
                 principled_bsdf_node.inputs['IOR'].default_value = 100000000000
            #principled_bsdf_node.inputs['Emission'].default_value =
            mat.use_transparent_shadow = False
 

    def update_material_nodes_uv(self, mode, rand=False):
        """
        Update the material nodes to either display the base color (albedo) or a combination of roughness and metallic values.
        param use_albedo: If True, sets up for albedo map; otherwise, sets up for roughness-metallic map.
        param rand: If True, randomizes the roughness and metallic values.
        """

        for mat in bpy.data.materials:
            nodes = mat.node_tree.nodes
            if 'New BSDF' not in nodes.keys():
                continue

            principled_bsdf_node = nodes['New BSDF']
            if not principled_bsdf_node:
                continue  # Skip this mesh if no suitable BSDF node is found

            emission_node = nodes.new(type="ShaderNodeEmission")
            if mode == "albedo":
                # Link albedo to emission
                if principled_bsdf_node.inputs["Base Color"].is_linked:
                    albedo_node = principled_bsdf_node.inputs["Base Color"].links[0].from_node
                    mat.node_tree.links.new(albedo_node.outputs["Color"], emission_node.inputs["Color"])
                else:
                    emission_node.inputs["Color"].default_value = principled_bsdf_node.inputs[
                        "Base Color"
                    ].default_value
            elif mode == "roughness_metallic":
                # mode is like "roughness_metallic"
                # Create a mix of roughness (G) and metallic (B) values
                combine_node = nodes.new(type="ShaderNodeCombineRGB")
                mat_fn = lambda ch, name: self.assign_material_value( mat.node_tree,
                    combine_node, principled_bsdf_node, ch, name, rand=rand
                )

                combine_node.inputs["R"].default_value = 1  # R is fixed
                # mat_fn("R", "Specular")
                mat_fn("G", "Roughness")
                mat_fn("B", "Metallic")
                mat.node_tree.links.new(
                    combine_node.outputs["Image"], emission_node.inputs["Color"]
                )
            elif mode == "bump":
                if principled_bsdf_node.inputs['Normal'].is_linked:
                    normal_map_node = principled_bsdf_node.inputs['Normal'].links[0].from_node
                    if normal_map_node.type == 'NORMAL_MAP' and normal_map_node.space == 'TANGENT':
                        material_node = normal_map_node.inputs['Color'].links[0].from_node
                        if material_node.type == 'TEX_IMAGE':
                            mat.node_tree.links.new(
                                material_node.outputs["Color"], emission_node.inputs["Color"]
                            )
                        else:
                            emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                    else:
                        emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
                else:
                    emission_node.inputs["Color"].default_value = (0.5, 0.5, 1, 1)
            elif mode == "position":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Position'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])
            elif mode == "normal":
                geometry_node = nodes.new(type='ShaderNodeNewGeometry')
                separate_node = nodes.new(type='ShaderNodeSeparateXYZ')
                combine_node = nodes.new(type='ShaderNodeCombineXYZ')

                
                mul_node = nodes.new(type='ShaderNodeMath')
                mul_node.operation = "MULTIPLY"
                mul_node.inputs[1].default_value = (-1.0)
                add_node = nodes.new(type='ShaderNodeVectorMath')
                add_node.operation = "ADD"
                add_node.inputs[1].default_value = (1.0, 1.0, 1.0)
                devide_node = nodes.new(type='ShaderNodeVectorMath')
                devide_node.operation = "DIVIDE"
                devide_node.inputs[1].default_value = (2.0, 2.0, 2.0)

                mat.node_tree.links.new(geometry_node.outputs['Normal'], separate_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[1], mul_node.inputs[0])

                mat.node_tree.links.new(separate_node.outputs[0], combine_node.inputs[0])
                mat.node_tree.links.new(separate_node.outputs[2], combine_node.inputs[1])
                mat.node_tree.links.new(mul_node.outputs[0], combine_node.inputs[2])

                mat.node_tree.links.new(combine_node.outputs[0], add_node.inputs[0])
                mat.node_tree.links.new(add_node.outputs[0], devide_node.inputs[0])
                mat.node_tree.links.new(devide_node.outputs[0], emission_node.inputs["Color"])


            if mode == 'rendering':
                mat.node_tree.links.new(
                    principled_bsdf_node.outputs['BSDF'],
                    nodes["Material Output"].inputs["Surface"],
                )
            else:
                # Connect emission to material output
                mat.node_tree.links.new(
                    emission_node.outputs["Emission"],
                    nodes["Material Output"].inputs["Surface"],
                )


    def save_material_images(self, mode='albedo', save_camera=False) -> None:
        """Saves rendered images of the object in the scene."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # prepare to save
        img_dir = os.path.join(self.args.output_dir, self.object_uid, mode)
        pose_dir = os.path.join(self.args.output_dir, self.object_uid, 'pose')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        if mode == 'albedo':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 2048
        elif mode == 'roughness_metallic' or mode == 'bump' or mode == 'normal':
            self.scene.view_settings.view_transform = 'Raw'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 2048
        elif mode == 'rgba_new':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = True
            self.scene.cycles.samples = 2048
        bpy.context.scene.cycles.max_bounces = 128
        bpy.context.scene.eevee.clamp_surface_indirect = 0
        bpy.context.scene.eevee.light_threshold = 0

        self.update_material_nodes(mode=mode, add_noise=False)

        for i in range(self.args.num_images):
            # set the camera position
            self.camera.location = self.cam_locations[i]
            self.camera.rotation_euler = self.cam_rotations[i]

            # render the image
            render_path = os.path.join(img_dir, f"{i:03d}_0001.png")
            self.render.filepath = render_path

            bpy.ops.render.render(write_still=True)

            # save camera RT matrix (C2W)
            RT_path = os.path.join(pose_dir, f"{i:03d}.npy")
            if save_camera:
                location, rotation = self.camera.matrix_world.decompose()[0:2]
                print(type(rotation))
                print(type(rotation.to_matrix()))
                RT = self.compose_RT(rotation.to_matrix(), np.array(location))
                print(RT.shape)
                np.save(RT_path, RT)
        
        # save the camera intrinsics
        if save_camera:
            intrinsics = self.get_calibration_matrix_K_from_blender(self.camera.data, return_principles=True)
            with open(os.path.join(self.args.output_dir, self.object_uid,'intrinsics.npy'), 'wb') as f_intrinsics:
                np.save(f_intrinsics, intrinsics)

    
    def set_random_lighting(self, cam_pos, lighting_type):
        if lighting_type == 'POINT':
            self.randomize_point_lighting(cam_pos)
        elif lighting_type == 'AREA':
            self.randomize_area_lighting(cam_pos)
        elif lighting_type == 'ENV':
            self.randomize_env_lighting()
        else:
            raise ValueError(f"Lighting type {lighting_type} not supported.")


    def save_material_images_multi_lighting(self, mode='rendering', save_camera=False) -> None:
        """Saves rendered images of the object in the scene."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # prepare to save
        img_dir = os.path.join(self.args.output_dir, self.object_uid, mode)
        pose_dir = os.path.join(self.args.output_dir, self.object_uid, 'pose')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        if mode == 'albedo':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 2048
        elif mode == 'roughness_metallic' or mode == 'bump' or mode == 'normal':
            self.scene.view_settings.view_transform = 'Raw'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 2048
        elif mode == 'rendering':
            self.scene.view_settings.view_transform = 'Standard'
            self.scene.cycles.use_denoising = False
            self.scene.cycles.samples = 2048
            bpy.context.scene.eevee.taa_render_samples = 16384
        #bpy.context.scene.cycles.max_bounces = 100000000
        #bpy.context.scene.cycles.sample_clamp_indirect = 100000000
        #bpy.context.scene.cycles.sample_clamp_direct = 10000000
        #bpy.context.scene.cycles.glossy_bounces = 100000000
        self.update_material_nodes(mode=mode, add_noise=False)
        #bpy.context.scene.cycles.use_denoising = True
        #bpy.context.scene.cycles.min_light_bounces = 100
        #bpy.context.scene.cycles.use_adaptive_sampling = False
        #bpy.context.scene.cycles.min_light_bounces = 100
        #bpy.context.scene.cycles.min_transparent_bounces = 100




        #lighting_types = ['POINT']*3 + ['AREA']*3 + ['ENV']*1
        lighting_types = ['ENV']*1

        for i in range(self.args.num_images):
            # set the camera position
            self.camera.location = self.cam_locations[i]
            self.camera.rotation_euler = self.cam_rotations[i]

            for j, lt in enumerate(lighting_types):

                self.set_random_lighting(self.cam_locations[i], lt)

                # render the image
                render_path = os.path.join(img_dir, f"{i:03d}_{j:04d}.png")
                self.render.filepath = render_path

                bpy.ops.render.render(write_still=True)

            # save camera RT matrix (C2W)
            RT_path = os.path.join(pose_dir, f"{i:03d}.npy")
            if save_camera:
                location, rotation = self.camera.matrix_world.decompose()[0:2]
                RT = self.compose_RT(rotation.to_matrix(), np.array(location))
                np.save(RT_path, RT)
        
        # save the camera intrinsics
        if save_camera:
            intrinsics = self.get_calibration_matrix_K_from_blender(self.camera.data, return_principles=True)
            with open(os.path.join(self.args.output_dir, self.object_uid,'intrinsics.npy'), 'wb') as f_intrinsics:
                np.save(f_intrinsics, intrinsics)


    
    def combine_objects(self):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                obj.select_set(True)

        bpy.context.view_layer.objects.active = bpy.context.selected_objects[-1]
        if len(bpy.context.selected_objects) > 1:
            bpy.ops.object.join()

            bpy.ops.object.mode_set(mode='EDIT')
            obj = bpy.context.active_object
            obj.data.uv_layers.new(name="NewUV")
            obj.data.uv_layers.active_index = len(obj.data.uv_layers) - 1
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
            # bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')

    
    def bake_material_images(self, image_name, bake_type, color_space='sRGB', width=1024, height=1024):
        """Saves baked images of the object in the scene."""

        self.update_material_nodes_uv(mode=image_name)

        # define a new baking image
        bpy.ops.image.new(name=image_name, width=width, height=height)
        bake_image = bpy.data.images[image_name]
        bake_image.colorspace_settings.name = color_space

        # prepare to save
        os.makedirs(self.args.output_dir, exist_ok=True)
        img_dir = os.path.join(self.args.output_dir, self.object_uid, 'material_uv')
        os.makedirs(img_dir, exist_ok=True)

        # add the bake image for each material
        for mat in bpy.data.materials:

            if 'New BSDF' not in mat.node_tree.nodes.keys():
                continue
            # add texture node into tree
            texture_node = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
            texture_node.image = bake_image
            mat.node_tree.nodes.active = texture_node

            # obj.data.materials.clear()
            # obj.data.materials.append(mat)

        # set bake type
        bpy.context.scene.cycles.bake_type = bake_type
        
        # select object and bake
        obj = bpy.context.view_layer.objects.active
        obj.select_set(True)

        if image_name == 'rendering':
            for i, HDR_file in enumerate(self.HDR_files):
                # set texture environment
                bpy.context.scene.world.node_tree.nodes['HDRTex'].image = bpy.data.images.load(HDR_file)
                bpy.ops.object.bake(type=bake_type)
                bake_image.filepath_raw = os.path.join(img_dir, f"{image_name.lower()}_uv_{i:03d}.png")
                bake_image.file_format = 'PNG'
                bake_image.save()
        else:
            bpy.ops.object.bake(type=bake_type)
            bake_image.filepath_raw = os.path.join(img_dir, f"{image_name.lower()}_uv.png")
            bake_image.file_format = 'PNG'
            bake_image.save()


    def download_object(self, object_url: str) -> str:
        """Download the object and return the path."""
        # uid = uuid.uuid4()
        uid = object_url.split("/")[-1].split(".")[0]
        tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
        local_path = os.path.join("tmp-objects", f"{uid}.glb")
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
        urllib.request.urlretrieve(object_url, tmp_local_path)
        os.rename(tmp_local_path, local_path)
        # get the absolute path
        local_path = os.path.abspath(local_path)
        return local_path


    def get_calibration_matrix_K_from_blender(self, camera, return_principles=False):
        """
            Get the camera intrinsic matrix from Blender camera.
            Return also numpy array of principle parameters if specified.
            
            Intrinsic matrix K has the following structure in pixels:
                [fx  0 cx]
                [0  fy cy]
                [0   0  1]
            
            Specified principle parameters are:
                [fx, fy] - focal lengths in pixels
                [cx, cy] - optical centers in pixels
                [width, height] - image resolution in pixels
        """
        # Render resolution
        render = bpy.context.scene.render
        width = render.resolution_x * render.pixel_aspect_x
        height = render.resolution_y * render.pixel_aspect_y

        # Camera parameters
        focal_length = camera.lens  # Focal length in millimeters
        sensor_width = camera.sensor_width  # Sensor width in millimeters
        sensor_height = camera.sensor_height  # Sensor height in millimeters

        # Calculate the focal length in pixel units
        focal_length_x = width * (focal_length / sensor_width)
        focal_length_y = height * (focal_length / sensor_height)

        # Assuming the optical center is at the center of the sensor
        optical_center_x = width / 2
        optical_center_y = height / 2

        # Constructing the intrinsic matrix
        K = np.array([[focal_length_x, 0, optical_center_x],
                    [0, focal_length_y, optical_center_y],
                    [0, 0, 1]])
        
        if return_principles:
            return np.array([
                [focal_length_x, focal_length_y],
                [optical_center_x, optical_center_y],
                [width, height],
            ])
        else:
            return K
        
    # function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
    def get_3x4_RT_matrix_from_blender(self, cam):
        # bcam stands for blender camera
        # R_bcam2cv = Matrix(
        #     ((1, 0,  0),
        #     (0, 1, 0),
        #     (0, 0, 1)))

        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam @ location
        #
        # Use matrix_world instead to account for all constraints
        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        # T_world2bcam = -1*R_world2bcam @ cam.location
        # Use location from matrix_world to account for constraints:     
        T_world2bcam = -1*R_world2bcam @ location

        # # Build the coordinate transform matrix from world to computer vision camera
        # R_world2cv = R_bcam2cv@R_world2bcam
        # T_world2cv = R_bcam2cv@T_world2bcam

        # put into 3x4 matrix
        RT = Matrix((
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],)
            ))
        return RT
        
    
    def generate_pose(self, camera_option='fixed') -> None:
        '''generate camera poses for rendering'''
        self.cam_locations = []
        self.cam_rotations = []
        radius = np.random.uniform(4, 4, 1)[0]
        for i in range(self.args.num_images):
            # camera_option = 'random' if i > 0 else 'front'

            if camera_option == 'fixed':
                locations = [
                    [ 0, -1,  0],
                    [-1,  0,  0],
                    [ 0,  1,  0],
                    [ 1,  0,  0],
                    [-1, -1,  1],
                    [-1,  1,  1],
                    [ 1,  1,  1],
                    [ 1, -1,  1],
                    [ 0,  0,  1],
                    [ 0,  0,  -1]
                ]
                vec = locations[i]
                vec = vec / np.linalg.norm(vec, axis=0) * radius
                x, y, z = vec
            elif camera_option == 'random':
                # from https://blender.stackexchange.com/questions/18530/
                radius_min=3.99 # 1.6
                radius_max=4.01 # 2.2
                maxz=4.01 #2.0
                minz=-4.01 #-0.75
                x, y, z = self.sample_spherical(radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz)
            elif camera_option == 'front':
                x, y, z = 0, -np.random.uniform(1.6, 2.2, 1)[0], 0

            self.cam_locations.append((x, y, z))

            # adjust orientation
            direction = - Vector((x, y, z))
            rot_quat = direction.to_track_quat('-Z', 'Y')
            self.cam_rotations.append(rot_quat.to_euler())

        
    def init_scene(self, object_file: str) -> None:
        """Load object into the scene."""
        self.reset_scene()

        # load the object
        self.load_object(object_file)
        self.object_uid = os.path.basename(object_file).split(".")[0]
        #self.normalize_scene(box_scale=2)
        self.camera, self.cam_constraint = self.setup_camera()

        # create an empty object to track
        empty = bpy.data.objects.new("Empty", None)
        self.scene.collection.objects.link(empty)
        self.cam_constraint.target = empty

        # generate camera pose
        if self.cam_locations is None:
             self.generate_pose(camera_option='random')

        # set the world
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")

        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes.clear()
        background_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeBackground")
        background_node.name = "Background"
        background_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background_node.inputs["Strength"].default_value = 0.1

        output_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
        output_node.name = "WorldOutput"

        bpy.context.scene.world.node_tree.links.new(background_node.outputs["Background"], output_node.inputs["Surface"])

        # add environment texture 
        # we didn't use HDR environment light, you can use it according to your needs 
        # env_texture_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        # env_texture_node.name = "HDRTex"
        # bpy.context.scene.world.node_tree.links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
        # HDR_path = './env_maps'
        # HDR_files = [f for f in os.listdir(HDR_path) if f.endswith('.exr')]
        # self.HDR_files = [os.path.join(HDR_path, f) for f in HDR_files]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"]
    )
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--render_space", type=str, default="VIEW", choices=["VIEW", "UV"])
        
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    print('===================', args.engine, '===================')
    np.random.seed(99999)
    random.seed(99999)

    start_i = time.time()
    if args.object_path.startswith("s3+"):
        import megfile
        local_tmp_obj_dir = "./working_dir/tmp_objs_beta"
        cache_local_item = os.path.join(local_tmp_obj_dir, args.object_path.split("/")[-1])
        megfile.smart_copy(args.object_path, cache_local_item)
        local_path = cache_local_item
    else:
        local_path = args.object_path

    render = BlenderRendering(args)
    render.init_scene(local_path)
    render.get_material_nodes()

    if args.render_space == "VIEW":
        render.save_material_images(mode='normal', save_camera=True)
        render.init_scene(local_path)
        render.save_material_images(mode='position') # CCM (optional)
        render.init_scene(local_path)
        render.save_material_images(mode='albedo')
        render.init_scene(local_path)
        render.save_material_images(mode='roughness_metallic') # G channel is roughness, B channel is metallic
        render.init_scene(local_path)
        render.save_material_images(mode='bump')
        render.init_scene(local_path)
        render.save_material_images_multi_lighting(mode='rendering')
    elif args.render_space == "UV":
        render.combine_objects() # combine all meshes into a single mesh
        render.bake_material_images(image_name='albedo', bake_type='EMIT', color_space='sRGB')
        render.bake_material_images(image_name='bump', bake_type='EMIT', color_space='Non-Color')
        render.bake_material_images(image_name='position', bake_type='EMIT', color_space='Non-Color') # CCM
        render.bake_material_images(image_name='roughness_metallic', bake_type='EMIT', color_space='Non-Color')
        # render.bake_material_images(image_name='rgba_new', bake_type='COMBINED', color_space='sRGB', width=512, height=512)

    end_i = time.time()
    print("Finished " + local_path + " in " + f"{end_i - start_i}" + " seconds")

    if args.object_path.startswith("http"):
        os.remove(local_path)
