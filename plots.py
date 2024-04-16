import json
import matplotlib.pyplot as plt

# Read JSON data from file
with open('test_results_withError.json', 'r') as file:
    json_data = json.load(file)

# Sort data by index
sorted_data = sorted(json_data.items(), key=lambda x: x[1]['idx'])

# Extract sorted keys and mean_reward values
sorted_keys = [item[0] for item in sorted_data]
mean_rewards = [item[1]['mean_reward'] for item in sorted_data]
d_error = [item[1]['distance_error'] for item in sorted_data]
or_error = [item[1]['orientation_error'] for item in sorted_data]
dq_error = [item[1]['dq_error'] for item in sorted_data]

# Plot mean_reward values vs sorted keys as a line plot with background grid
plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, mean_rewards, marker='o')  # Use marker='o' for data points
plt.xlabel('Model')
plt.ylabel('Mean Reward')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('Mean Reward (Ordered by Index)')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, d_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Model')
plt.ylabel('Distance error (m)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('Distance error (Ordered by Index)')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, or_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Model')
plt.ylabel('Orientation error (º)')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('Orientation Error (Ordered by Index)')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_keys, dq_error, marker='o')  # Use marker='o' for data points
plt.xlabel('Model')
plt.ylabel('DQ error')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.title('Dual Quaternion Error (Ordered by Index)')
plt.grid(True)  # Add background grid
plt.tight_layout()
plt.show()
