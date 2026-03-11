import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
datasets = ['MNIST', 'Iris', 'Breast Cancer', 'Ionosphere']
encodings = ['Angle', 'Diagonal', 'IQP', 'Pauli']

# Mean accuracy
means = {
    'Angle':    [0.879, 1.000, 0.879, 0.692],
    'Diagonal': [0.991, 0.991, 0.920, 0.836],
    'IQP':      [0.838, 0.995, 0.750, 0.690],
    'Pauli':    [0.862, 1.000, 0.889, 0.703],
}

# Error bars
errors = {
    'Angle':    [0.056, 0.000, 0.042, 0.035],
    'Diagonal': [0.002, 0.002, 0.000, 0.000],
    'IQP':      [0.033, 0.015, 0.063, 0.062],
    'Pauli':    [0.055, 0.000, 0.022, 0.014],
}

# =========================
# Plot style for academic paper
# =========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

# Colors chosen to be professional and distinguishable
colors = {
    'Angle': '#4C72B0',     # blue
    'Diagonal': '#C44E52',  # red
    'IQP': '#55A868',       # green
    'Pauli': '#8172B3',     # purple
}

# =========================
# Grouped bar positions
# =========================
x = np.arange(len(datasets))
bar_width = 0.18

offsets = [-1.5*bar_width, -0.5*bar_width, 0.5*bar_width, 1.5*bar_width]

for i, enc in enumerate(encodings):
    ax.bar(
        x + offsets[i],
        means[enc],
        width=bar_width,
        label=enc,
        color=colors[enc],
        edgecolor='black',
        linewidth=1.0,
        yerr=errors[enc],
        capsize=6,
        error_kw={
            'elinewidth': 1.2,
            'capthick': 1.2,
            'ecolor': 'black'
        }
    )

# =========================
# Axes formatting
# =========================
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel('accuracy', fontdict={'fontsize': 20})
ax.set_xlabel('datasets', fontdict={'fontsize': 20})

# You can adjust the y range depending on journal style
ax.set_ylim(0.0, 1.05)

# Light horizontal grid only
ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Legend
legend = ax.legend(
    # title='Encoding Type',
    frameon=True,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.01),
    ncol=4,
    columnspacing=1.5,
    handletextpad=0.6,
    borderaxespad=0.3
)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

# Remove top/right spines for cleaner academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout for publication
plt.tight_layout()

# Save figure
plt.savefig('./results/benchmark_accuracy_barplot.png', dpi=600, bbox_inches='tight')
plt.savefig('./results/benchmark_accuracy_barplot.pdf', bbox_inches='tight')

# plt.show()