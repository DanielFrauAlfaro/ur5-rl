import json
import matplotlib.pyplot as plt

# Read JSON data from file
with open('test_results_withError.json', 'r') as file:
    json_data = json.load(file)

# Sort data by index
sorted_data = sorted(json_data.items(), key=lambda x: x[1]['idx'])

# for elem in sorted_data:
#     print(elem[0])
#     print(type(elem))
#     elem[0] = "AAA"
#     raise

# Extract sorted keys and mean_reward values
sorted_keys = [item[1]["idx"] for item in sorted_data]
mean_rewards = [item[1]['mean_reward'] for item in sorted_data]
d_error = [item[1]['distance_error'] for item in sorted_data]
or_error = [item[1]['orientation_error'] for item in sorted_data]
dq_error = [item[1]['dq_error'] for item in sorted_data]

# Plot mean_reward values vs sorted keys as a line plot with background grid
plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, mean_rewards, marker='o')  # Use marker='o' for data points
plt.xlabel('Steps')
plt.ylabel('Mean Reward')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, d_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Steps')
plt.ylabel('Distance error (m)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, or_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Steps')
plt.ylabel('Orientation error (rad)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, dq_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Steps')
plt.ylabel('DQ error')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('Dual ')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()
