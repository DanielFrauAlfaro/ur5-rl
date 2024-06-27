import numpy as np
import matplotlib.pyplot as plt

x = ["DQ", "EULER"]
y = [0.7, 0.15]

# Gráfico de barras
fig, ax = plt.subplots()
ax.grid(axis='y', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.bar(x = x[0], height = y[0], ecolor="g", alpha=0.9, zorder=3)
ax.bar(x = x[1], height = y[1], alpha=0.9, zorder=4)
ax.errorbar(x[0], y[0], yerr=0.2, fmt='o', color="k", zorder=6)
ax.errorbar(x[1], y[1], yerr=0.1, fmt='o', color="k", zorder=6)
ax.set_xlabel('Agent', fontsize=15)
ax.set_ylabel('Success rate', fontsize=15)

plt.show()