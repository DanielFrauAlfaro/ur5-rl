import pybullet as p
import pybullet_data as pd
import os

p.connect(p.DIRECT)
name_in = "/home/rajat/sbpl_ws/src/TMOMO/models/007_tuna_fish_can/textured_simple.obj"
name_out = "textured_simple_vhacd.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log)
