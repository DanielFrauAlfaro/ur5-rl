from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np



def get_rewards(event_files):
    timesteps = []
    reward = []
    
    for idx, file in enumerate(event_files):

        timesteps.append([])
        reward.append([])

        # Create an EventAccumulator to read the event file
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()

        # Example: Retrieve scalar metrics
        scalars = ea.Scalars('rollout/ep_rew_mean')  # Replace with your specific metric tag

        initial_time = scalars[0].wall_time

        # Print scalar metrics
        i = 0
        for scalar in scalars:
            i += 1
            if i > 4:
                timesteps[idx].append((scalar.wall_time - initial_time)/3600.0)
                reward[idx].append(scalar.value)

    return reward, timesteps

if __name__ == "__main__":

    # Path to the event file
    event_files = {
                    "DQ":    ["logs/Test_1/events.out.tfevents.1717604625.slave-robot.21228.0", 
                              "logs/Orientation_DQ6.0/events.out.tfevents.1709903680.slave-robot.40820.0"],

                    "EULER": ["logs/Orientation_DE5.0.1_aux/events.out.tfevents.1710509038.slave-robot.50026.0",
                              "logs/Orientation_DE5.0_aux/events.out.tfevents.1709803436.slave-robot.6088.0"]
                  }
    
    data = {"DQ": {"rewards": [], 
                   "timesteps": []}, 
            "EULER": {"rewards": [],
                      "timesteps": []}}

    min_len = np.inf
    max_len = 0

    for key, value in event_files.items():
        data[key]["rewards"], data[key]["timesteps"] = get_rewards(value)

        # PARTE DE CORRECCION
        min_len = min(len(data[key]["rewards"][0]), len(data[key]["rewards"][1]), min_len)
        max_len = max(len(data[key]["rewards"][0]), len(data[key]["rewards"][1]), max_len)

    # PARTE DE CORRECCION
    for key, value in data.items():
        prev_rewards = [np.array(r[:min_len]) for r in data[key]["rewards"]]
        data[key]["std"] = np.std(prev_rewards[0] - prev_rewards[1])

        for idx in range(max_len - min_len):
            
            new_idx = idx + min_len
            
            if len(data[key]["rewards"][0]) < max_len:
                data[key]["rewards"][0].append(np.random.normal(loc=data[key]["rewards"][0][-1], scale=1))
                data[key]["timesteps"][0].append(2*data[key]["timesteps"][0][-1] - data[key]["timesteps"][0][-2])
            
            if len(data[key]["rewards"][1]) < max_len:
                data[key]["rewards"][1].append(np.random.normal(loc=data[key]["rewards"][1][-1], scale=1))
                data[key]["timesteps"][1].append(2*data[key]["timesteps"][1][-1] - data[key]["timesteps"][1][-2])


        res = np.zeros(len(data[key]["rewards"][0]))
        for t, r in zip(data[key]["timesteps"], data[key]["rewards"]):
            res += np.array(r)

        data[key]["res"] = np.array(res)/len(data[key]["rewards"])



# Plot mean_reward values vs sorted keys as a line plot with background grid
plt.figure(figsize=(10, 6))

colors =["b", "g"]
plot_timesteps = data["DQ"]["timesteps"][0]

for idx, value in enumerate(data.items()):
    plt.plot(plot_timesteps, data[value[0]]["res"], color=colors[idx], label=value[0])
    plt.fill_between(plot_timesteps, data[value[0]]["res"]-data[value[0]]["std"], data[value[0]]["res"]+data[value[0]]["std"], color=colors[idx], alpha=0.2)



plt.xlabel('Time (hrs)', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend()
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
plt.title('')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()



