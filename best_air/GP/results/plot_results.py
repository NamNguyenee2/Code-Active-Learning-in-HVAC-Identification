import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 15,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7.5,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = 'ini10_0.2'
N_PLOT = 288

AL_STRATEGIES = [
    ('GS',  r'$\mathrm{GS}_{\mathrm{GP}}$',  '#a65628', '-'),
    ('IDW', r'$\mathrm{IDW}_{\mathrm{GP}}$', '#984ea3', '-'),
    ('US',  'MV',  '#f781bf', '-'),
    ('IVR', 'IVR', '#377eb8', '-'),
    ('FS',  'FI',  '#17becf', '-'),
    ('IG',  'SE',  '#e41a1c', '-'),
    ('PI',  'PI',  '#4daf4a', '-'),
    ('UCB', 'UCB', '#ff7f00', '-'),
]

BASELINES = [
    ('PL', 'PL', '#222222', '--'),
]

def load_rmse(strategy):
    path = os.path.join(RESULTS_DIR, f'RMSE_{strategy}_GP_{CONFIG}.csv')
    return pd.read_csv(path).iloc[:, 0].values


def plot_learning_curves():
    fig_w = 160 / 50.4
    fig_h = fig_w * 0.20

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(N_PLOT) * 5 / 60

    for key, label, color, ls in BASELINES:
        ax.plot(x, load_rmse(key)[:N_PLOT], color=color, linestyle=ls,
                linewidth=1.0, label=label, zorder=2)

    for key, label, color, ls in AL_STRATEGIES:
        ax.plot(x, load_rmse(key)[:N_PLOT], color=color, linestyle=ls,
                linewidth=1.2, label=label, zorder=3)

    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('RMSE (°C)')
    ax.set_xlim(0, (N_PLOT - 1) * 5 / 60)

    n_init, ramp = CONFIG.lstrip('ini').split('_')
    ax.set_title(f'GP model  |  Initial points: {n_init},  Ramp: {ramp}')
    ax.yaxis.grid(True, linewidth=0.5, color='#dddddd', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(ncol=3, loc='upper right', frameon=True,
              framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout(pad=0.4)

    base = os.path.join(RESULTS_DIR, f'learning_curves_GP_{CONFIG}')
    fig.savefig(base + '.pdf')
    plt.show()


def _nice_ticks(hour_max, step):
    ticks = list(np.arange(0, hour_max, step))
    if not np.isclose(ticks[-1], hour_max):
        ticks.append(hour_max)
    ticks[0] = 0
    return ticks


def plot_learning_curves_panels():
    x = np.arange(N_PLOT) * 5 / 60
    hour_full = x[-1]

    panels = [
        ('0–2h',  2.0,        0.5),
        ('0–12h', 12.0,       2.0),
        ('0–24h', hour_full,  4.0),
    ]

    fig_w = 160 / 25.4
    panel_h = fig_w * 0.35


    fig, axes = plt.subplots(3, 1, figsize=(fig_w, panel_h * 3))

    rmse = {}
    for key, _, color, ls in BASELINES + AL_STRATEGIES:
        rmse[key] = load_rmse(key)[:N_PLOT]

    for i, (ax, (panel_label, hour_max, step)) in enumerate(zip(axes, panels)):
        for key, label, color, ls in BASELINES:
            ax.plot(x, rmse[key], color=color, linestyle=ls,
                    linewidth=1.0, label=label, zorder=2)

        for key, label, color, ls in AL_STRATEGIES:
            ax.plot(x, rmse[key], color=color, linestyle=ls,
                    linewidth=1.2, label=label, zorder=3)

        ax.set_xlim(0, hour_max)
        ticks = _nice_ticks(hour_max, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{round(t, 1):g}' for t in ticks])

        ax.set_ylabel('RMSE (°C)')
        ax.set_title(panel_label, fontsize=11, loc='left')
        ax.yaxis.grid(True, linewidth=0.5, color='#dddddd', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == len(axes) - 1:
            ax.set_xlabel('Time (hour)')

        if i == 2:
            ax.legend(ncol=3, loc='upper right', frameon=True,
                       framealpha=0.9, edgecolor='#cccccc')

    n_init, ramp = CONFIG.lstrip('ini').split('_')
    ramp = float(ramp)
    title = (
        rf'GP model  |  Initial points: {n_init},  '
        rf'Ramp constraints for $T^s$: {ramp * 40:g}$^\circ$C, '
        rf'$\dot{{m}}$: {ramp * 100:g}%'
    )
    fig.suptitle(title, y=0.99)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 1.01))

    base = os.path.join(RESULTS_DIR, f'learning_curves_GP_{CONFIG}_panels')
    fig.savefig(base + '.pdf')
    plt.show()


if __name__ == '__main__':
    plot_learning_curves_panels()
