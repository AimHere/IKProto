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


class CCDSolver:
    def __init__(self, bonelist, bonetree, rootbone, rotations, keypoints, rootpos = Position([0,0,0])):
        self.bonetree = bonetree
        self.bonelist = bonelist
        self.root = rootbone
        self.keypoints = keypoints
        self.orig_rotations = rotations
        
        self.tpose = self.untarget(keypoints, rotations)

        for i,b in enumerate(self.tpose):
            print("%d(%s) : %s"%(i, self.bonelist[i], str(self.tpose[i])))

        self.parents = [-1 for i in self.bonelist]

        for bstr in self.bonetree:
            pIdx = self.bonelist.index(bstr)
            for cstr in self.bonetree[bstr]:
                cIdx = self.bonelist.index(cstr)
                self.parents[cIdx] = pIdx
            
        grots = self.get_global_rotations(self.orig_rotations)
        print([str(q) for q in grots])
        
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

    def get_global_rotations(self, rotations):
        out_rotations = []        
        def _recurse(bone, c_rot, pIdx):

            
            cIdx = self.bonelist.index(bone)

            if (pIdx < 0):
                n_rot = c_rot
            else:
                n_rot = c_rot * rotations[pIdx]

            out_rotations.append(n_rot)

            for child in self.bonetree[bone]:
                _recurse(child, n_rot, cIdx)
                
        initial_rot = rotations[self.bonelist.index(self.root)]

        _recurse(self.root, initial_rot, -1)
        
        return out_rotations
        

    
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

    
    def recalc_tpose(self, keypoints, rotations):
        new_pos = self.untarget(keypoints, rotations)
        return new_pos
        
    def bone_path(self, b_from, b_to):
        bone_path = [b_to]
        pIdx = self.parents[b_to]

        if (b_to != b_from):
            if (pIdx == -1):
                addendum = [i for i in self.bone_path(b_to, b_from)][:-1]
                bone_path.extend(reversed(addendum))
            else:
                bone_path.extend(self.bone_path(b_from, pIdx))


    def rot_towards_test(self, effector, pivot_bone, end_bone):
        current_rotation = self.orig_rotations[pivot_bone]


        new_rotations = self.orig_rotations.copy()
        #new_rotations[pivot_bone] = rot_diff

        glob_rots = self.get_global_rotations(self.orig_rotations)

        old_globrot = glob_rots[pivot_bone]
        
        rot_diff = self.rotate_towards(effector, pivot_bone, end_bone, prerot = old_globrot)
        new_rotations[pivot_bone] = old_globrot.inv() * rot_diff * new_rotations[pivot_bone]# * old_globrot
        return new_rotations
        
        
    def rotate_towards(self, effector, pivot_bone, end_bone, prerot = Quaternion([0,0,0,1])):
        # Returns the new rotation value that moves the end bone as close to the effector as possible

        pivot = self.keypoints[pivot_bone]
        end_pt = self.keypoints[end_bone]
        effect_pt = Position(effector)
        
        # target_vector = Position(effector) - self.keypoints[pivot_bone]
        # current_vector = self.keypoints[end_bone] - self.keypoints[pivot_bone]
        target_vector = prerot.inv().apply(effect_pt - pivot)
        current_vector = prerot.inv().apply(end_pt - pivot)

        print("Vecs: %s, %s"%(str(target_vector), str(current_vector)))

        axis_raw = current_vector.cross(target_vector)
        axis_norm = axis_raw.norm()
        angle = math.atan2(axis_raw.mag(), target_vector.dot(current_vector))
        
        shalf = math.sin(angle / 2)
        chalf = math.cos(angle / 2)
        ax_sc = axis_norm.scale(shalf)
        q = prerot * Quaternion([ax_sc.x, ax_sc.y, ax_sc.z, chalf])
        print("Difference : %s"%str(q))


        test_pt = pivot + q.apply(end_pt - pivot)
        print("Pivot: %s move target from %s to %s to reach %s"%(pivot, end_pt, test_pt, effect_pt))
        return q


        
    
parser = argparse.ArgumentParser()

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

q_rots = [Quaternion(pose_rots[i]) for i in range(pose_rots.shape[0])]
p_poses = fkn.propagate(q_rots, Position.zero())

iksolver = CCDSolver(body_34_parts, body_34_tree, 'PELVIS', q_rots, p_poses)

# r_tpose = np.array([p.np() for p in fksolver.recalc_tpose(ppos, pquats)])
iksolver.rotate_towards(args.effector, args.pivot_bone, args.extreme_bone)

test_rotations  = iksolver.rot_towards_test (args.effector, args.pivot_bone, args.extreme_bone)

new_poses = fkn.propagate(test_rotations, Position.zero())

rlist = [iksolver.tpose, p_poses, new_poses]

renderlist = [np.array([p.np() for p in pose]) for pose in rlist]





renderer = Render(renderlist,
                  iksolver.parents,
                  lineplot = args.lineplot,
                  elev = args.elev,
                  azim = args.azim,
                  roll = args.roll,
                  figsize = args.figsize,
                  scale = args.scale,
                  static_bone = args.pivot_bone,
                  extreme_bone = args.extreme_bone,
                  effector = args.effector)
                  

