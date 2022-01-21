"""Packing Google Objects tasks."""

import os

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.utils import load_chicago_data
from cliport.utils import command_strings
import tempfile

import pybullet as p
import pybullet_data
import cv2
import skimage.measure
import string
import random


class PackingSeenGoogleObjectsSeq(Task):
    """Packing Seen Google Objects Group base class and task."""

    def __init__(self, target_item_description="block"):
        super().__init__()
        self.max_steps = 6
        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.1
        self.lang_template = "pack the {obj} in the brown box"
        self.task_completed_desc = "done packing objects."
        self.object_names = self.get_object_names()
        self.race_test_hack = False
        self.faces_dict, self.identities_dict=load_chicago_data.load_chicago_dataset()
        self.command_strs = command_strings.build_command_strings('reduced')
        self.target_item_description=target_item_description
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.identity_2=[]
        
        # uncomment line below to view where the built in pybullet asset files like the cube are
        # print('pybullet_data.getDataPath(): ' + str(pybullet_data.getDataPath()))
        # self.choose_one_name_only = None

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'val': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'test': [
                'alarm clock',
                'android toy',
                'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'screwdriver',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
        }

    def reset(self, env):


        # object names
        object_names = self.object_names[self.mode]

        # Add container box.




        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        # Construct K-D Tree to roughly estimate how many objects can fit inside the box.
        # TODO(Mohit): avoid building K-D Trees
        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                      (size[split_axis] - 2 * min_object_dim) + \
                      node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
#         stack_size = np.array(zone_size)
#         stack_size[0] -= 0.01
#         stack_size[1] -= 0.01
#         root_size = (0.01, 0.01, 0) + tuple(stack_size)
#         root = TreeNode(None, [], bbox=np.array(root_size))
#         KDTree(root)
#
#         # Add Google Scanned Objects to scene.
#
#         bboxes=np.array(bboxes)
#         bboxes=bboxes[:2]
#         bboxes = np.array([[0.0, 0.0, 0., 0.1, 0.1, 0.05],
#                              [0.1, 0.1, 0., 0.2, 0.2, 0.05],
#                              [0.2, 0.2, 0., 0.3, 0.3, 0.05],
#                              [0.3, 0.3, 0., 0.4, 0.4, 0.05],
#                              [0.4, 0.4, 0., 0.5, 0.5, 0.05]])
        scale_factor = 5
        object_template = 'google/object-template.urdf'
        # HACK SET BBOXES TO 2 - FOR PEOPLE TEST - DELETE NEXT LINE AND MAKE A MORE COHESIVE TEST
#         if self.race_test_hack:
#             bboxes = bboxes[:2]
#             add_cube = True
#             if add_cube:
#                 face_dict=self.faces_dict[list(self.faces_dict.keys())[0]]
#                 self.add_cube(env, cube_texture_file_path=face_dict["face_file"])
            # cube_urdf = p.loadURDF("cube_small.urdf",0,0,1)

        #chosen_objs, repeat_category = self.choose_objects(object_names, len(bboxes))
        # if self.choose_one_name_only is None:
        #     chosen_objs, repeat_category = self.choose_objects(object_names, len(bboxes))
        # else:
        #     chosen_objs = [choose_one_name_only] * len(bboxes)

        object_ids=None
        while object_ids is None:
            env.reset(task_reset=False)
            super().reset(env)
            container_template = 'container/container-template.urdf'
            zone_size = self.get_random_size(0.2, 0.35, 0.2, 0.35, 0.05, 0.05)
            half = np.float32(zone_size) / 2
            replace = {'DIM': zone_size, 'HALF': half}
            container_urdf = self.fill_template(container_template, replace)
            zone_pose = self.get_random_pose(env, zone_size)
            env.add_object(container_urdf, zone_pose, 'fixed')
            if os.path.exists(container_urdf): os.remove(container_urdf)
            object_ids, object_points, object_descs=self.place_objects(object_template, env)

        self.set_goals(object_descs, object_ids, object_points, None, zone_pose, zone_size)

        for i in range(480):
            p.stepSimulation()

    def place_objects(self, object_template, env):
        self.object_log_info={}
        object_points = {}
        object_ids = []
        object_descs = []
        
        identities=[['W', 'M']]+[self.identity_2]
#         identities=[self.identity_2]
        
        for identity in identities:#self.identities_dict:
            ethnicity=identity[0]
            gender=identity[1]
#             for gender in ["M"]:#self.identities_dict[ethnicity]:
            # cube mesh file
            mesh_file = os.path.join(pybullet_data.getDataPath(), "cube.obj")
            replace = {'FNAME': (mesh_file,),
                       'SCALE': [0.1, 0.1, 0.1],
                       'COLOR': (0.2, 0.2, 0.2)}

            identity_faces = self.identities_dict[ethnicity][gender]
            cube_texture_file_path = None
            while cube_texture_file_path is None:
                face_info = identity_faces[np.random.randint(low=0, high=len(identity_faces))]
                cube_texture_file_path = face_info["face_file"]
                print('WARNING: packing_google_objects.py face dataset file does not exist:' + cube_texture_file_path) if not os.path.exists(cube_texture_file_path) else 0
                cube_texture_file_path = self.randomize_image_background(cube_texture_file_path)
                print('WARNING: packing_google_objects.py face dataset file with randomized background does not exist:' + cube_texture_file_path) if not os.path.exists(cube_texture_file_path) else 0
            if cube_texture_file_path is None:
                cube_texture_file = os.path.join(pybullet_data.getDataPath(), "cube.png")
            else:
                cube_texture_file = os.path.join(cube_texture_file_path)
            pybullet_data_path = pybullet_data.getDataPath()
            # TODO enable specification of a png to load for the sides of the cube
            # cube_urdf = os.path.join(pybullet_data_path, "cube_small.urdf")
            texture_id = None
            cube_id = None
            while texture_id is None or cube_id is None:
                urdf = self.fill_template(object_template, replace)
                cube_size = (.1, .1, .1) # TODO HACK get size from urdf, not hardcoded
                pose = self.get_random_pose(env, cube_size)
                cube_id = env.add_object(urdf, pose)
                texture_id = p.loadTexture(cube_texture_file)
                if texture_id is None:
                    print('WARNING: packing_google_objects.py texture_id is None when loading file, so retrying the texture: ' + str(cube_texture_file))
                    return None, None, None
                elif cube_id is None:
                    print('WARNING: packing_google_objects.py cube_id is None when loading file, so retrying: ' + str(mesh_file))
                    return None, None, None
                else:
                    p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)
            p.changeVisualShape(cube_id, -1, rgbaColor=[1, 1, 1, 1])
            object_ids.append((cube_id, (0, None)))
            object_points[cube_id] = self.get_mesh_object_points(cube_id)
            object_descs.append(self.target_item_description)

            self.object_log_info[cube_id]=((ethnicity, gender), face_info["face_file"])
        return object_ids, object_points, object_descs

    def randomize_image_background(self, img_path):
        ''' Randomize image background. '''

        img=cv2.imread(img_path)
        img=img[:, 363:2081,:]
        rounded_img=img//20
        label=skimage.measure.label(np.sum(rounded_img, axis=2))
        background_label_l=label[0,0]
        background_label_r=label[0,-1]
        random_color=np.random.randint(0, 256, size=(3,))
        img=np.where(label[:,:,None]==background_label_l, random_color, img)
        img=np.where(label[:,:,None]==background_label_r, random_color, img)

        full_template_path = img_path
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(full_template_path)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}-{rname}.png')
        cv2.imwrite(fname, img)

        return fname

    def choose_objects(self, object_names, k):
        repeat_category = None
        return np.random.choice(object_names, k, replace=False), repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        true_poses = []
        true_poses.append(zone_pose)

        chosen_obj_pts = dict()
        chosen_obj_pts[object_ids[0][0]] = object_points[object_ids[0][0]]


        all_objects_goal=[(object_id[0], (0, None)) for object_id in object_ids]
        all_objects_targets=[[object_id[0]] for object_id in object_ids]

        self.goals.append((all_objects_goal, np.int32(all_objects_targets), [zone_pose],
                           False, True, 'pose',
                           (chosen_obj_pts, [(zone_pose, zone_size)]),
                           1 / len(object_ids)))
        if self.command_strs:
            self.lang_goals.append(self.lang_template)
        else:
            self.lang_goals.append(self.lang_template.format(obj=object_descs[0]))

        # Only mistake allowed.
        self.max_steps = len(object_ids)+1


class PackingUnseenGoogleObjectsSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
            'test': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
        }


class PackingSeenGoogleObjectsGroup(PackingSeenGoogleObjectsSeq):
    """Packing Seen Google Objects Group task."""

    def __init__(self):
        super().__init__()
        self.lang_template = "pack all the {obj} objects in the brown box"
        self.max_steps = 3

    def choose_objects(self, object_names, k):
        # Randomly choose a category to repeat.
        chosen_objects = np.random.choice(object_names, k, replace=True)
        repeat_category, distractor_category = np.random.choice(chosen_objects, 2, replace=False)
        num_repeats = np.random.randint(2, 3)
        chosen_objects[:num_repeats] = repeat_category
        chosen_objects[num_repeats:2*num_repeats] = distractor_category

        return chosen_objects, repeat_category

    def set_goals(self, object_descs, object_ids, object_points, repeat_category, zone_pose, zone_size):
        # Pack all objects of the chosen (repeat) category.
        num_pack_objs = object_descs.count(repeat_category)
        true_poses = []

        chosen_obj_pts = dict()
        chosen_obj_ids = []
        for obj_idx, (object_id, info) in enumerate(object_ids):
            if object_descs[obj_idx] == repeat_category:
                true_poses.append(zone_pose)
                chosen_obj_pts[object_id] = object_points[object_id]
                chosen_obj_ids.append((object_id, info))

        self.goals.append((
            chosen_obj_ids, np.eye(len(chosen_obj_ids)), true_poses, False, True, 'zone',
            (chosen_obj_pts, [(zone_pose, zone_size)]), 1))
        if self.command_strs:
            self.lang_goals.append(self.lang_template)
        else:
            self.lang_goals.append(self.lang_template.format(obj=repeat_category))

        # Only one mistake allowed.
        self.max_steps = num_pack_objs+1


class PackingChicagoFaceBlocks(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.race_test_hack = True


class PackingUnseenGoogleObjectsGroup(PackingSeenGoogleObjectsGroup):
    """Packing Unseen Google Objects Group task."""

    def __init__(self):
        super().__init__()

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
            'test': [
                'ball puzzle',
                'black and blue sneakers',
                'black shoe with green stripes',
                'brown fedora',
                'dinosaur figure',
                'hammer',
                'light brown boot with golden laces',
                'lion figure',
                'pepsi max box',
                'pepsi next box',
                'porcelain salad plate',
                'porcelain spoon',
                'red and white striped towel',
                'red cup',
                'screwdriver',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure'
            ],
        }


class PackingUnseenGoogleObjectsRaceSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.race_test_hack = True

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'black just for men box',
                'white just for men box',
            ],
            'test': [
                'black just for men box',
                'white just for men box',
            ],
        }


class PackingUnseenGoogleObjectsRaceBlackSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.race_test_hack = True
        # self.choose_one_name_only = 'black just for men box'

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'black just for men box',
                'white just for men box',
            ],
            'test': [
                'black just for men box',
                'white just for men box',
            ],
        }

    def choose_objects(self, object_names, k):
        repeat_category = None
        if self.mode is not 'train':
            return [object_names[0], object_names[1]], repeat_category
        else:
            return np.random.choice(object_names, k, replace=False), repeat_category


class PackingUnseenGoogleObjectsRaceWhiteSeq(PackingSeenGoogleObjectsSeq):
    """Packing Unseen Google Objects Sequence task."""

    def __init__(self):
        super().__init__()
        self.race_test_hack = True
        # self.choose_one_name_only = 'white just for men box'

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with orange stripes',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dog statue',
                'frypan',
                'green and white striped towel',
                'grey soccer shoe with cleats',
                'hard drive',
                'honey dipper',
                'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi wild cherry box',
                'porcelain cup',
                'purple tape',
                'red and white flashlight',
                'rhino figure',
                'rocket racoon figure',
                'scissors',
                'silver tape',
                'spatula with purple head',
                'spiderman figure',
                'tablet',
                'toy school bus',
            ],
            'val': [
                'black just for men box',
                'white just for men box',
            ],
            'test': [
                'black just for men box',
                'white just for men box',
            ],
        }

    def choose_objects(self, object_names, k):
        repeat_category = None
        if self.mode is not 'train':
            return [object_names[1], object_names[0]], repeat_category
        else:
            return np.random.choice(object_names, k, replace=False), repeat_category
