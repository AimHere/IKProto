
import re
import argparse

from newiktest import body_34_parts, body_34_tree, body_34_tpose, CCDSolver, Render
import torch
import numpy as np

from zed_utilities import Quaternion, Position, ForwardKinematics, ForwardKinematics_Torch
BONES_34_COMPACT = [2,3,5,6,12,13,18,19,22,23,26]
teststring = "[NATIVE] [debug] IKTrans: pre-pass rot: -212.291 -20.2782 1698.14 ori 0.017243 -0.99942 -0.0284433 0.007279\n[NATIVE] [debug] IKTrans: post-pass rot: -212.291 -20.2782 1698.14 ori 0.017243 -0.99942 -0.0284433 0.007279\n[NATIVE] [debug] IKTrans: pre-pass rotations [0 0 0 1] [-0.00277718 -0.135105 0.00341807 0.990821] [0.229987 0.0893277 0.652058 0.7169] [0 0.495407 0 0.868661] [0.0804163 0.0332041 -0.456496 0.885462] [0 -0.149236 0 0.988802] [0.189611 -0.00286874 0.0129704 0.981769] [-0.0158086 0 0 0.999875] [0.108066 -0.00952177 0.0863063 0.990345] [0.00106815 0 0 0.999999] [0 0 0 1]\n[NATIVE] [debug] IKTrans: post-pass rotations [0 0 0 1] [-0.00277718 -0.135105 0.00341807 0.990821] [0.229987 0.0893277 0.652058 0.7169] [0 0.495407 0 0.868661] [0.0804163 0.0332041 -0.456496 0.885462] [0 -0.149236 0 0.988802] [0.189611 -0.00286874 0.0129704 0.981769] [-0.0158086 0 0 0.999875] [0.108066 -0.00952177 0.0863063 0.990345] [0.00106815 0 0 0.999999] [0 0 0 1] \n[NATIVE] [debug] IKTrans: pre-pass keypoints [-191.091 -16.8239 1652.55] [-195.92 124.658 1660.65] [-201.639 266.09 1669.04] [-207.359 407.526 1677.43] [-168.686 409.895 1676.87] [-53.4969 414.659 1675.08] [-14.4796 197.746 1699.87] [-75.1796 58.6993 1550] [-87.3195 30.8899 1520.03] [-111.6 -24.7289 1460.09] [-144.662 -2.37941 1513.43] [-246.094 406.693 1678.07] [-361.284 401.929 1679.86] [-482.304 217.235 1700.6] [-593.239 40.373 1657.12] [-615.426 5.00062 1648.43] [-659.8 -65.7442 1631.04] [-594.724 -58.0982 1643.09] [-100.098 -13.6476 1651.33] [-96.073 -368.383 1770.74] [-92.5071 -686.902 1866.86] [-91.7436 -795.371 1795.33] [-282.083 -20.0001 1653.78] [-333.31 -385.956 1713.49] [-378.839 -711.134 1767.27] [-390.998 -808.61 1682.22] [-197.035 536.512 1629.24] [-198.461 578.106 1631.97] [-182.544 605.536 1668.38] [-157.31 581.462 1755.19] [-230.974 604.704 1655.77] [-294.527 579.104 1719.48] [-91.5729 -756.919 1922.74] [-389.962 -787.035 1813.55] \n[NATIVE] [debug] IKTrans: post-pass keypoints [-212.291 -20.2782 1698.14] [-217.121 121.204 1706.24] [-222.84 262.635 1714.63] [-228.56 404.071 1723.02] [-189.887 406.441 1722.46] [-74.6978 411.205 1720.67] [-35.6805 194.292 1745.45] [-96.3805 55.245 1595.59] [-108.52 27.4356 1565.62] [nan -nan(ind) -nan(ind)] [-165.863 -5.83376 1559.02] [-267.295 403.239 1723.66] [-382.484 398.475 1725.45] [-503.505 213.781 1746.19] [-614.44 36.9187 1702.71] [-636.627 1.54632 1694.02] [-681 -69.1985 1676.63] [-615.924 -61.5524 1688.68] [-121.299 -17.1019 1696.92] [-117.274 -371.837 1816.33] [-113.708 -690.357 1912.45] [-112.944 -798.825 1840.91] [-303.284 -23.4545 1699.37] [-354.51 -389.41 1759.07] [-400.039 -714.588 1812.86] [-412.198 -812.064 1727.81] [-218.236 533.057 1674.83] [-219.662 574.651 1677.56] [-203.745 602.082 1713.97] [-178.51 578.007 1800.78] [-252.175 601.25 1701.36] [-315.728 575.649 1765.07] [-112.774 -760.374 1968.33] [-411.163 -790.489 1859.13]\n[NATIVE] [debug] IKTrans: Effector position: -91.0905 183.176 1852.55, bone-chain 5 to 8"



class RootTrans:
    def __init__(self, name, data):
        bef, aft = data.split (" ori ")
        self.name = name
        self.pos = Position([float(x) for x in bef.split(" ")])
        self.ori = Quaternion([float(x) for x in aft.split(" ")])

    def __str__(self):
        return("R: [%s], O: [%s]"%(self.pos, self.ori))

class Vecs:
    def __init__(self, name, data):
        self.name = name
        dprocess1 = [v.replace("[","").replace("]","").strip() for v in data.split("] [")]
        self.vecs = []
        for f in dprocess1:
            try:
                self.vecs.append(Position([float(c) for c in f.split(" ")]))
            except(ValueError):
                self.vecs.append(Position([-999999,-999999, -999999]))
                                 
    def __str__(self):
        return " | ".join([str(v) for v in self.vecs])

class Quats:
    def __init__(self, name, data):
        self.name = name
        try:
            dprocess1 = [v.replace("[","").replace("]","").strip() for v in data.split("] [")]
            self.quats = [Quaternion([float(c) for c in f.split(" ")]) for f in dprocess1]            
        except(ValueError):
            self.quats = []

        
    def __str__(self):
        return " | ".join([str(q) for q in self.quats])

class IKProblem:
    def __init__(self, name, data):
        self.name = name

        mstr = '(.+?), bone-chain (\\d+) to (\\d+)'
        m = re.match(mstr, data)
        if (m):
            pdata = m.groups()[0].strip().split(" ")
            self.effector = [float(i) for i in pdata]
            self.pivot_bone = int(m.groups()[1])
            self.end_bone = int(m.groups()[2])

        else:
            print("Effector Data not found: %s"%data)
            self.effector = None
            self.pivot_bone = None
            self.end_bone = None

    def __str__(self):
        return("Eff: [%s], bones %d to %d"%(str(self.effector), self.pivot_bone, self.end_bone))
        
parser = argparse.ArgumentParser()

parser.add_argument("--nodots", action = 'store_true', help = "Line only, no dots")
parser.add_argument("--save", type = str, help = "Save to file")
parser.add_argument("--elev", type = float, help = "Elevation", default = 0)
parser.add_argument("--azim", type = float, help = "Azimuth", default = 0)
parser.add_argument("--roll", type = float, help = "Roll", default = 0)
parser.add_argument("--lineplot", action = 'store_true', help = "Draw a skel")
parser.add_argument("--scale", type = int, help = "Scaling factor", default = 1000.0)
parser.add_argument("--figsize", type = int, nargs = 2, help = "Figure size in pixels", default = (1600, 800))

parser.add_argument('--file', type = str)

args = parser.parse_args()

if (args.file is not None):
    with open(args.file) as fp:
        instring = fp.readlines()
else:
    instring = teststring.split("\n")

modes = {'pre-pass rot:' : RootTrans,
         'post-pass rot:' : RootTrans,
         'pre-pass rotations' : Quats,
         'post-pass rotations' : Quats,
         'pre-pass keypoints' : Vecs,
         'post-pass keypoints' : Vecs,
         'Effector position:' : IKProblem
         }

ldata = {}

for line in instring:
    m = re.match("\[NATIVE\] \[debug\] IKTrans: (.+)-pass (.+?) (.+)", line)
    m2 = re.match("\[NATIVE\] \[debug\] IKTrans: Effector position: (.+)", line)
    if (m):
        mode = "%s-pass %s"%(m.groups()[0], m.groups()[1])
        data = m.groups()[2]
        ldata[mode] = modes[mode](mode, data)
        
    elif (m2):
        mode = "Effector position:"
        data = m2.groups()[0]
        ldata[mode] = modes[mode](mode, data.strip())
    else:
        pass
        #print("Bad line: %s"%line)

print(ldata['Effector position:'])

for m in ldata:
    print("%s: %s"%(m, str(ldata[m])))

#    fkt = ForwardKinematics_Torch(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)
fkn = ForwardKinematics(body_34_parts, body_34_tree, 'PELVIS', body_34_tpose)

q_rots = [Quaternion([0,0,0,1]) for q in range(34)]
for i, b in enumerate(BONES_34_COMPACT):
    q_rots[b] = ldata['pre-pass rotations'].quats[i]
        
q_poses = ldata['pre-pass keypoints'].vecs
p_poses = fkn.propagate(q_rots, Position.zero())
problem = ldata['Effector position:']

iksolver = CCDSolver(body_34_parts, body_34_tree, 'PELVIS', q_rots, p_poses)

test_rotations = iksolver.CCD_pass(problem.effector, problem.pivot_bone, problem.end_bone, piv_to_end = True)
new_poses = fkn.propagate(test_rotations, Position.zero())
    
rlist = [p_poses, new_poses]

renderlist = [np.array([p.np() for p in pose]) for pose in rlist]

renderer = Render(renderlist,
                  iksolver.parents,
                  lineplot = args.lineplot,
                  elev = args.elev,
                  azim = args.azim,
                  roll = args.roll,
                  figsize = args.figsize,
                  scale = args.scale,
                  static_bone = problem.pivot_bone,
                  extreme_bone = problem.end_bone,
                  effector = problem.effector)
    

