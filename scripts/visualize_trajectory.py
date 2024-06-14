import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = pd.read_csv('/media/shapelim/UX960NVMe1/newer-college-dataset/01_short_experiments/ground_truth/registered_poses.csv', delimiter=',')

print(data)
# Plot the x, y, z trajectory
plt.figure(figsize=(10, 7))
plt.plot(data['x'], data['y'], label='xy trajectory')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectories')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()