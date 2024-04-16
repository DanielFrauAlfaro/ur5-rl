from gymnasium.envs.registration import register

register(
     id="ur5_rl/Ur5Env-v0",
     entry_point="ur5_rl.envs:UR5Env")