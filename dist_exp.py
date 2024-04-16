
import matplotlib.pyplot as plt
import numpy as np
import dqrobotics
from dq_cosas import *
import pybullet as p
import copy
import re


def get_quaternion(q):
    return q[-1] + dqrobotics.i_ * q[0] + dqrobotics.j_ * q[1] + dqrobotics.k_ * q[2]

def get_dualQuaternion(q_r, q_t):
    return q_r + 0.5*dqrobotics.E_*q_t*q_r


client = p.connect(p.DIRECT)

p_original = [0.1350236995852392, 0.4888180601513394, 1.2106725454148264]
or_original = [ 1.5289873,   0.78591081, -1.54129301]

p_aux = copy.deepcopy(p_original)

p_aux.append(0.0)
pos_wQ = get_quaternion(p_aux)
orn_w = p.getQuaternionFromEuler(or_original)
orn_wQ = get_quaternion(orn_w)
DQ_w_original = get_dualQuaternion(q_r=orn_wQ, q_t=pos_wQ)
original_DQ_vec = dqrobotics.vec8(DQ_w_original)


DQ_dist = []
euc_dist = []
pos = []

idx = 0

for i in range(1000):
    new_p = copy.deepcopy(p_original)
    new_p[idx] += i / 200
    new_p[1] += i / 200
    new_p[2] += i / 200

    
    euc_d = np.linalg.norm(np.array(new_p) - np.array(p_original))
    # print(p_original)
    # print("--")
    pos.append(new_p[idx])

    new_p.append(0.0)
    pos_wQ = get_quaternion(new_p)
    orn_w = p.getQuaternionFromEuler(or_original)
    orn_wQ = get_quaternion(orn_w)
    DQ_w = get_dualQuaternion(q_r=orn_wQ, q_t=pos_wQ)
    w_DQ_vec = dqrobotics.vec8(DQ_w)

    __, DQ_d, __ = dq_distance(torch.tensor(np.array([w_DQ_vec])), torch.tensor(np.array([original_DQ_vec])))
    
    DQ_dist.append(DQ_d.item())
    euc_dist.append(euc_d)


# Plot mean_reward values vs sorted keys as a line plot with background grid
# Plotting
plt.plot(pos, DQ_dist, '-', label="DQ", color="green")  # 'o' for point markers
plt.plot(pos, euc_dist, '-', label="EUC", color="blue")
plt.xlabel('Position')
plt.ylabel('Euclidean Distance')
plt.legend()  # Add legend based on 'label' in plt.plot
plt.title('Euclidean Distance vs. Position')
plt.grid(True)  # Add grid lines
plt.show()



# Display the plot
plt.show()

'''
Hay mas pendiente en el caso de DQ, lo que puede ayudar a diferenciar entre mejores recompensas
en el caso de que se produzcan movimientos en una direccion
'''









