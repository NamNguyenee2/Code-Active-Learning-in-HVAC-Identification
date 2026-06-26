import os
import pandas as pd

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = 'ini2_0.05'
N_PLOT = 288
DECIMALS = 3

AL_STRATEGIES = [
    ('GS',  r'$\mathrm{GS}_{\mathrm{GP}}$'),
    ('IDW', r'$\mathrm{IDW}_{\mathrm{GP}}$'),
    ('US',  'MV'),
    ('IVR', 'IVR'),
    ('FS',  'FI'),
    ('IG',  'SE'),
    ('UCB', 'UCB'),
    ('PI',  'PI'),
]

BASELINES = [
    ('PL', 'PL'),
]

WINDOWS = [
    ('0--2h',  24),
    ('0--12h', 144),
    ('0--24h', 288),
]


def load_rmse(strategy):
    path = os.path.join(RESULTS_DIR, f'RMSE_{strategy}_GP_{CONFIG}.csv')
    return pd.read_csv(path).iloc[:, 0].values


def build_table():
    rows = []
    for key, label in BASELINES + AL_STRATEGIES:
        rmse = load_rmse(key)[:N_PLOT]
        row = {'Strategy': label}
        for win_label, n_steps in WINDOWS:
            window = rmse[:n_steps]
            row[(win_label, 'Mean')] = window.mean()
            row[(win_label, 'Last')] = window[-1]
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Strategy')
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Window', 'Metric'])
    return df.T


def to_latex(df):
    strategies = list(df.columns)
    al_labels = [label for _, label in AL_STRATEGIES]
    col_spec = 'll' + 'c' * len(strategies)

    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append('\\small')
    n_init, ramp = CONFIG.lstrip('ini').split('_')
    ramp = float(ramp)
    caption = (
        rf'\normalsize GP model \textbar{{}} '
        rf'Initial points: {n_init}, Ramp constraints for $T^s$: {ramp * 40:g}$^\circ$C, '
        rf'$\dot{{m}}$: {ramp * 100:g}\%'
    )
    lines.append(f'\\caption{{{caption}}}')
    lines.append(r'\vspace{0.2cm}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')

    header = ' & '.join(strategies)
    lines.append(f'Window & Metric & {header} \\\\')
    lines.append('\\midrule')

    windows = [win_label for win_label, _ in WINDOWS]
    for wi, window in enumerate(windows):
        for mi, metric in enumerate(('Mean', 'Last')):
            row = df.loc[(window, metric)]
            best = row[al_labels].min()
            cells = []
            for strategy in strategies:
                text = f'{row[strategy]:.{DECIMALS}f}'
                if strategy in al_labels and row[strategy] == best:
                    text = f'\\textbf{{{text}}}'
                cells.append(text)
            window_cell = f'\\multirow{{2}}{{*}}{{{window}}}' if mi == 0 else ''
            lines.append(f'{window_cell} & {metric} & ' + ' & '.join(cells) + ' \\\\')
        if wi != len(windows) - 1:
            lines.append('\\midrule')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


if __name__ == '__main__':
    df = build_table()
    print()
    print(to_latex(df))
