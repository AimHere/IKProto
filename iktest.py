import argparse
import math
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd

from zed_utilities import body_34_parts, body_34_tree, body_34_tpose, ForwardKinematics

from zed_utilities import Quaternion, Position

zero_pose = [Quaternion([0, 0, 0, 1]) for u in range(34)]

class Render:
    def __init__(self, skelList, parents, nodots = False, lineplot = True,
                 scale = 1.0,
                 elev = 90.0, azim = 0.0, roll = 0.0,
                 figsize = (1600, 800),
                 effector = None,
                 static_bone = None,
                 extreme_bone = None
                 ):

        self.skeletons = skelList
        self.parents = parents
        
        self.nodots = nodots
        self.lineplot = lineplot

        self.scale = scale
        
        self.elev = elev
        self.azim = azim
        self.roll = roll
        
        self.ax = []
        self.fig = plt.figure()
        self.fig.set_figwidth(figsize[0] / self.fig.get_dpi())
        self.fig.set_figheight(figsize[1] / self.fig.get_dpi())        

        self.effector = effector
        self.static_bone = static_bone        
        self.extreme_bone = extreme_bone

        self.dotpoints = []
        self.lines = []
        
        for idx, skel in enumerate(self.skeletons):
            self.lines.append([])
            self.ax.append(self.fig.add_subplot(1, len(self.skeletons), idx + 1, projection = '3d'))

            self.ax[idx].set_xlim(-self.scale, self.scale)
            self.ax[idx].set_ylim(-self.scale, self.scale)
            self.ax[idx].set_zlim(-self.scale, self.scale)            

            self.ax[idx].view_init(elev = self.elev, azim = self.azim, roll = self.roll, vertical_axis = 'y')

            if (not self.nodots):                
                bluelist = [i for i in range(skel.shape[0]) if (i != self.static_bone and i != self.extreme_bone)]

                self.dotpoints.append(self.ax[idx].scatter(skel[bluelist, 0], skel[bluelist, 1], skel[bluelist, 2], color = 'blue'))
                
                if (self.extreme_bone):
                    eb = self.extreme_bone
                    print("Greening Extreme bone %d"%eb)
                    self.dotpoints.append(self.ax[idx].scatter(skel[eb:eb + 1, 0],
                                                               skel[eb:eb + 1, 1],
                                                               skel[eb:eb + 1, 2], color = "green"))

                if (self.static_bone):

                    sb = self.static_bone
                    print("Reddening Pivot bone %d"%sb)                    
                    self.dotpoints.append(self.ax[idx].scatter(skel[sb:sb + 1, 0],
                                                               skel[sb:sb + 1, 1],
                                                               skel[sb:sb + 1, 2], color = "red"))

 
                if (self.effector):

                    ef = np.array(self.effector).reshape([3, 1])

                    self.dotpoints.append(self.ax[idx].scatter(ef[0],
                                                               ef[1],
                                                               ef[2],
                                                               color = 'magenta',
                                                               s = 50))
                else:
                    print("No effector")
            if (self.lineplot):
                linex, liney, linez = self.build_lines(skel)

                for l in range(0, len(linex)):
                    self.ax[idx].plot(linex[l], liney[l], linez[l])                    
                
        plt.show()

    def build_lines(self, skel):
        linex = []
        liney = []
        linez = []
        for cIdx, pIdx in enumerate(self.parents):


            if (pIdx >= 0):
                linex.append([skel[cIdx, 0], skel[pIdx, 0]])
                liney.append([skel[cIdx, 1], skel[pIdx, 1]])                
                linez.append([skel[cIdx, 2], skel[pIdx, 2]])
        return [linex, liney, linez]
                    
                                          
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
        return new_pos
        
    def bone_path(self, b_from, b_to):
        
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

    def get_rotation(self, effector, effect_pos, pivot_pos, cum_rot = None):
        # Find the angle from rotating r_joint that minimizes the distance between the effector and
        # the position of e_joint

        # Without constraints, this is equivalent to finding the nearest point
        # to 'effector' on the sphere of radius | pos(r_joint) - pos(e_joint) |

        # Constraints seem to be about finding specific axis and then clamping the rotations.
        # I'm not convinced by this, yet


        # First, find the quaternion that rotates between two given vectors
        
        if (cum_rot is None):
            cum_rot = Quaternion.zero()
        
        if (effector == effect_pos):
            return Quaternion([0, 0, 0, 1]) # We're already there

        print("Unsetting old rotation: %s"%cum_rot)
        
        target_vec = cum_rot.inv().apply(effector - pivot_pos).norm()
        cur_vec = cum_rot.inv().apply(effect_pos - pivot_pos).norm()

        axis_raw = cur_vec.cross(target_vec)
        angle = math.atan2(axis_raw.mag(), target_vec.dot(cur_vec))
        
        axis_norm = axis_raw.norm()

        shalf = math.sin(angle / 2)
        chalf = math.cos(angle / 2)

        ax_sc = axis_norm.scale(shalf)
        q = Quaternion([ax_sc.x, ax_sc.y, ax_sc.z, chalf])
        return q


    def cum_rotations(self, rotations, initial_position):
        #keyvector = [Position([0, 0, 0]) for i in range(34)]
        cum_rots = [Quaternion([0,0,0,1]) for i in range(34)]
                    
        def _recurse(bone, c_rot, pIdx):
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
                #new_pos = initial_position
            else:
                n_rot = c_rot * rotations[pIdx]
                #new_pos = keyvector[pIdx] + n_rot.apply(self.tpose[cIdx] - self.tpose[pIdx])
            cum_rots[cIdx] = n_rot
            #keyvector[cIdx] = new_pos
            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return cum_rots

    def CCD_pass(self, effector, bone_list, pose, old_rotations):
        bone_positions = [pose[i] for i in bone_list]

        rots = [Quaternion.zero() for i in bone_positions]

        cum_rots = self.cum_rotations(old_rotations, Position([0,0,0]))
        for i,c in enumerate(cum_rots):
            print("Cumulative: %d: %s"%(i, c))

        print("Initial bone pos: ", [str(p) for p in bone_positions])
            
        for pIdx in range(len(bone_positions) - 2, -1, -1):

            used_bone = bone_list[pIdx]

            q = self.get_rotation(effector, bone_positions[-1], bone_positions[pIdx], cum_rots[used_bone])

            print("Got rotation for bone %d(%s): %s"%(bone_list[pIdx], self.bonelist[bone_list[pIdx]], q))
            #print("Got No-cumul for bone %d(%s): %s"%(bone_list[pIdx], self.bonelist[bone_list[pIdx]], q2))
            if (pIdx == len(bone_positions) - 2):
                rots[pIdx] = q
            else:
                rots[pIdx] = q * rots[pIdx + 1]
                
            # Update the bone positions 
            for uIdx in range(pIdx + 1, len(bone_positions)):

                print("%d: Initial Bone Pos: %s -> %s"%(uIdx, bone_positions[uIdx], bone_positions[pIdx]))
                print("%d: Initial Rotation: %s"%(uIdx, old_rotations[pIdx]))
                print("%d: Additional Rotation: %s"%(uIdx, q))
                print("%d: Full rotation: %s"%(uIdx, q * old_rotations[pIdx]))

                cr = cum_rots[uIdx]
                bone_positions[uIdx] = bone_positions[pIdx] + q.apply(bone_positions[uIdx] - bone_positions[pIdx])
                # print("Bone pos update: %d: %s + Rot [%s] * (%s - %s) [%s] = %s"%
                #       (uIdx,
                #        bone_positions[pIdx],
                #        q,
                #        pose[bone_list[uIdx]],
                #        pose[bone_list[pIdx]],
                #        pose[bone_list[uIdx]] - pose[bone_list[pIdx]],                       
                #        bone_positions[uIdx]))

            print("Post pass bone pos: ", [str(p) for p in bone_positions])
            # for i, bpos in enumerate(bone_positions):
            #     print("Pass: %d, bone_positions: %d: %s"%(pIdx, i, bpos))
        print("Bone list is ", [b for b in bone_list])
        for i, b in enumerate(bone_list):
            print("%d: %d (%s): %s->%s"%(i, b,
                                         self.bonelist[b],
                                         str(pose[b]),
                                         str(bone_positions[i])))
        print("--")
            
        new_rotations = old_rotations.copy()

        # Rotations are cumulative 
        for i, bp in enumerate(rots):
            # The bones here are being applied in the wrong order
            bu = bone_list[i]
            print("Adding rotation %s to bone %d(%s): %s"%(str(bp), bu, str(self.bonelist[bu]), new_rotations[bu]))
            #new_rotations[bu] = bp * new_rotations[bu]
            new_rotations[bu] = bp * new_rotations[bu]
                
        return new_rotations, old_rotations
            
    def CCD_run(self, effector, initial_pose, initial_rots, effector_bone, fixed_bone, max_iters = 100, threshold = None):
        # Go through the bones getting the rotations and applying them to get the updated position of the
        # extreme bone

        # Once you have the list of rotations, apply them all to the skel and get the new pose positions
        # Repeat until the threshold or max iters condition is hit

        bone_path_ = self.bone_path(fixed_bone, effector_bone)
        
        bone_path = [b for b in reversed(bone_path_)]
        
        print("--")
        print("Rev Bone path is ", [b for b in bone_path_])        
        print("Bone path is ", [b for b in bone_path])
        
        new_rots, old_rots = self.CCD_pass(effector, bone_path, initial_pose, initial_rots)
        
        return new_rots, old_rots

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

parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
parser.add_argument("--save", type = str, help = "Save to file")
parser.add_argument("--elev", type = float, help = "Elevation", default = 0)
parser.add_argument("--azim", type = float, help = "Azimuth", default = 0)
parser.add_argument("--roll", type = float, help = "Roll", default = 0)
parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
parser.add_argument("--scale", type = int, help = "Scaling factor", default = 1000.0)
parser.add_argument("--figsize", type = int, nargs = 2, help = "Figure size in pixels", default = (1600, 800))

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

tpose = np.array(body_34_tpose)
apose = np.array([p.np() for p in ppos])
#npose = np.array([p.np() for p in fksolver.recalc_tpose(ppos, pquats)])

runresults, prior_results = fksolver.CCD_run(effector, npose, pquats, args.extreme_bone, args.pivot_bone)
pass1_pose = np.array([p.np() for p in fkn.propagate(runresults, Position.zero())])
old_pose = np.array([p.np() for p in fkn.propagate(prior_results, Position.zero())])


for idx, pr in enumerate(prior_results):
    nr = runresults[idx]
    print("Rot: %s:\t%s\t->\t%s"%(body_34_parts[idx], str(pr), str(nr)))


print("Recalced skel pivot bone pos: %s -> %s"%(npose[args.pivot_bone], str(pass1_pose[args.pivot_bone])))
print("Recalced skel extreme bone pos: %s -> %s"%(npose[args.extreme_bone], str(pass1_pose[args.extreme_bone])))
print("New Result size: %d"%len(runresults))


parentslist = fksolver.parents


for i, b in enumerate(body_34_parts):
    print("Bone %d(%s): %s -> %s"%(i, b, old_pose[i], pass1_pose[i]))


renderer = Render([apose, old_pose, pass1_pose], parentslist, nodots = args.nodots, lineplot = args.lineplot,
                  elev = args.elev,
                  azim = args.azim,
                  roll = args.roll,
                  figsize = args.figsize,
                  scale = args.scale,
                  static_bone = args.pivot_bone,
                  extreme_bone = args.extreme_bone,
                  effector = args.effector)

