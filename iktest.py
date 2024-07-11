import argparse
import math
import numpy as np

from zed_utilities import body_34_parts, body_34_tree, body_34_tpose, ForwardKinematics

from zed_utilities import Quaternion, Position

zero_pose = [Quaternion([0, 0, 0, 1]) for u in range(34)]

class FKSolver:

    def __init__(self, bonelist, bonetree, rootbone, tpose, rootpos = Position([0,0,0])):
        self.bonetree = bonetree
        self.bonelist = bonelist
        self.root = rootbone
        self.tpose = [Position(p) for p in tpose]

        self.parents = [-1 for i in self.bonelist]

        for bstr in self.bonetree:
            pIdx = self.bonelist.index(bstr)
            for cstr in self.bonetree[bstr]:
                cIdx = self.bonelist.index(cstr)
                self.parents[cIdx] = pIdx
            
        print(self.parents)

    def untarget(self, keypoints, rotations, static = False):
        new_positions = [Position.zero()] * len(self.bonelist)

        def _recurse(bone_name, rot, pIdx):
            cIdx = self.bonelist.index(bone_name)
            if (pIdx < 0):
                if (static):
                    new_positions[cIdx] = Position.zero()
                else:
                    new_positions[cIdx] = keypoints[cIdx]
                newrot = rot
            else:
                newrot = rot * rotations[pIdx]
                new_positions[cIdx] = new_positions[pIdx] + newrot.inv().apply(keypoints[cIdx] - keypoints[pIdx])
            if (bone_name in self.bonetree):
                for cname in self.bonetree[bone_name]:
                    _recurse(cname, newrot, cIdx)
                    
        _recurse("PELVIS", Quaternion.zero(), -1)

        if (static):
            self.rootpos = Position([0, 0, 0])
        return new_positions

    def recalc_tpose(self, keypoints, rotations):
        new_pos = self.untarget(keypoints, rotations)
        print([str(p) for p in new_pos])
        print("--")
        print([str(p) for p in self.tpose])
        
    def bone_path(self, b_from, b_to):
        print("Finding %d->%d"%(b_from, b_to))
        
        # Return the list of bones from the 'to' bone to the 'from' bone
        bone_path = [b_to]
        pIdx = self.parents[b_to]
        
        if (b_to == b_from):
            pass # The bone has already been added?

        # We've hit the root bone. Now we have to append the reversed version
        elif (pIdx == -1):
            addendum = [i for i in self.bone_path(b_to, b_from)][:-1]
            bone_path.extend(reversed(addendum))

        else:
            bone_path.extend(self.bone_path(b_from, pIdx))
            
        return bone_path

    def get_rotation(self, effector, effect_pos, pivot_pos):
        # Find the angle from rotating r_joint that minimizes the distance between the effector and
        # the position of e_joint

        # Without constraints, this is equivalent to finding the nearest point
        # to 'effector' on the sphere of radius | pos(r_joint) - pos(e_joint) |

        # Constraints seem to be about finding specific axis and then clamping the rotations.
        # I'm not convinced by this, yet


        # First, find the quaternion that rotates between two given vectors

        if (effector == effect_pos):
            return Quaternion([0, 0, 0, 1]) # We're already there

        target_vec = (effector - pivot_pos).norm()
        cur_vec = (effect_pos - pivot_pos).norm()

        axis_raw = cur_vec.cross(target_vec)
        angle = math.atan2(axis_raw.mag(), target_vec.dot(cur_vec))
        
        axis_norm = axis_raw.norm()

        shalf = math.sin(angle / 2)
        chalf = math.cos(angle / 2)

        ax_sc = axis_norm.scale(shalf)
        q = Quaternion([ax_sc.x, ax_sc.y, ax_sc.z, chalf])
        return q


    def CCD_pass(self, effector, bone_list, pose):
        bone_positions = [pose[i] for i in bone_list]
        rots = []
        
        for pIdx in range(len(bone_positions) - 2, -1, -1):            
            q = self.get_rotation(effector, bone_positions[-1], bone_positions[pIdx])
            rots.append(q)
            # Update the bone positions 
            for uIdx in range(pIdx + 1, len(bone_positions)):
                bone_positions[uIdx] = bone_positions[pIdx] + q.apply(bone_positions[uIdx] - bone_positions[pIdx])

            for i, bpos in enumerate(bone_positions):
                print("Pass: %d, bone_positions: %d: %s"%(pIdx, i, bpos))

        return rots


            
    def CCD_run(self, effector, initial_pose, effector_bone, fixed_bone, max_iters = 100, threshold = None):
        # Go through the bones getting the rotations and applying them to get the updated position of the
        # extreme bone

        # Once you have the list of rotations, apply them all to the skel and get the new pose positions
        # Repeat until the threshold or max iters condition is hit

        new_rots = self.CCD_pass(effector, self.bone_path(effector_bone, fixed_bone), initial_pose)
        
        print([str(r) for r in new_rots])

    def propagate(self, rotations, initial_position):
        keyvector = [Position([0, 0, 0]) for i in range(34)]
        
        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
                new_pos = initial_position
            else:
                n_rot = c_rot * rotations[pIdx]
                new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])
                # print("Old: %d, Nrot:"%cIdx, n_rot)
                # print("Old: %d, NewPos: "%cIdx, new_pos)

            keyvector[cIdx] = new_pos
            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return keyvector

class CCD:
    def __init__(self, skeleton, constraints = None):
        self.skeleton = skeleton
        self.constraints = constraints

    def solve(self, effectors, static_bones, current_pose = None):
        # Solve the ik problem, given the list of target effectors, and the list of bones that are static, and a given tpose
        pass

class FABRIK:
    
    def __init__(self):
        pass

    def solve(self, effectors, static_bones, current_pose = None):
        # Solve the ik problem, given the list of target effectors, and the list of bones that are static, and a given tpose
        pass


import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('f', type = int)
# parser.add_argument('to', type = int)

parser.add_argument('frame', type = int)
parser.add_argument('pivot_bone', type = int)
parser.add_argument('extreme_bone', type = int)
parser.add_argument('effector', type = float, nargs = 3)

args = parser.parse_args()

effector = Position(args.effector)

pose_rots = np.load("S9_posing_1_zed34_test.npz", allow_pickle = True)['quats'][args.frame, :, :]
pose_kps = np.load("S9_posing_1_zed34_test.npz", allow_pickle = True)['keypoints'][args.frame, :, :]

fkn = ForwardKinematics(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)

pquats = [Quaternion(pose_rots[i]) for i in range(pose_rots.shape[0])]
ppos = [Position(pose_kps[i]) for i in range(pose_kps.shape[0])]

fksolver = FKSolver(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)
npose = fksolver.propagate(pquats, Position([0, 0, 0]))

print("Pivot from %s(%d) to extremity %s(%d)"%(body_34_parts[args.pivot_bone], args.pivot_bone, body_34_parts[args.extreme_bone], args.extreme_bone))

#fksolver.CCD_run(effector, npose, args.extreme_bone, args.pivot_bone)

fksolver.recalc_tpose(ppos, pquats)


