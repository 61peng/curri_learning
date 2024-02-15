import matplotlib.pyplot as plt
import numpy as np

# Updated datasets
models_tasks_new = ['Mixtral-8x7B\nSciCite', 'Mixtral-8x7B\nSciERC', 'Mixtral-8x7B\nSciNLI', 'Llama2-70B\nSciERC']
performance_random_new = [67.07, 32.95, 43.18, 31.39]
performance_human_new = [67.80, 33.41, 51.56, 32.94]
performance_model_new = [66.04, 32.22, 46.04, 29.12]

# Calculate improvement rates
improvement_human_new = [(h - r) / r * 100 for h, r in zip(performance_human_new, performance_random_new)]
improvement_model_new = [(m - r) / r * 100 for m, r in zip(performance_model_new, performance_random_new)]

# Define bar positions
x_new = np.arange(len(models_tasks_new))
width = 0.25
gap = 0.05
x_adjusted_new = np.array([x_new - width - gap, x_new, x_new + width + gap]).T.flatten()
x_model_line_new = x_adjusted_new[2::3]

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars
rects1_new = ax.bar(x_adjusted_new[::3], performance_random_new, width, label='Random', edgecolor='grey', linewidth=2, fill=False)
rects2_new = ax.bar(x_adjusted_new[1::3], performance_human_new, width, label='Human', edgecolor='blue', linewidth=2, fill=False)
rects3_new = ax.bar(x_adjusted_new[2::3], performance_model_new, width, label='Model', edgecolor='red', linewidth=2, fill=False)

# Adding value labels on top of each bar
for rect in rects1_new + rects2_new + rects3_new:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Primary axis settings
ax.set_ylabel('Performance')
# ax.set_title('Performance and Improvement Rate by Model, Task, and Demonstration Order')
ax.set_xticks(x_new)
ax.set_xticklabels(models_tasks_new)
ax.set_ylim([0, 100])

# Secondary axis for improvement rates
ax2 = ax.twinx()
ax2.plot(models_tasks_new, improvement_human_new, 'b-o', label='Improvement of Human')
ax2.plot(x_model_line_new, improvement_model_new, 'r-s', label='Improvement of Model', linewidth=2, markersize=8)

ax2.set_ylabel('Improvement Rate (%)')
ax2.axhline(0, color='black', linewidth=1)

# Adjusting the improvement rate y-axis limits
improvement_range_new = max(max(improvement_human_new), max(improvement_model_new), abs(min(improvement_human_new)), abs(min(improvement_model_new)))
ax2.set_ylim([-improvement_range_new - 5, improvement_range_new + 5])

# Combine legends from both axes
lines_new, labels_new = ax.get_legend_handles_labels()
lines2_new, labels2_new = ax2.get_legend_handles_labels()
ax2.legend(lines_new + lines2_new, labels_new + labels2_new, loc='upper left')

fig.tight_layout()
plt.savefig('bar_V2.png', dpi=400)
