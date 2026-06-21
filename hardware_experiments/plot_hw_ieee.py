#!/usr/bin/env python3
"""
plot_hw_ieee.py — IEEE-Style Hardware Experiment Plotter
=========================================================
Reads all CSV files from hardware_exp/logs/ and generates publication-ready
plots comparing PID, SAC-Nominal, and SAC-DR controllers under different
fault conditions.

Usage:
    cd ~/Fault_Tolerant
    python3 hardware_exp/plot_hw_ieee.py                        # all logs
    python3 hardware_exp/plot_hw_ieee.py --fault 1.0_1.0_1.0_1.0  # filter by fault
    python3 hardware_exp/plot_hw_ieee.py --save                 # save as PDF/PNG

Author: Ayush   Date: 2026-06-21
"""

import argparse
import os
import glob

import matplotlib
matplotlib.use('Agg')   # headless-safe; switch to TkAgg if you want interactive
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── IEEE column dimensions ──
FIG_W_IN = 7.16          # IEEE double-column width
FIG_H_IN = 4.5
DPI      = 300

# ── Color palette (matches software plot_ieee.py style) ──
COLORS = {
    'pid':         '#E53935',   # Red
    'sac_nominal': '#1E88E5',   # Blue
    'sac_dr':      '#43A047',   # Green
}
LABELS = {
    'pid':         'PID',
    'sac_nominal': 'SAC (Nominal)',
    'sac_dr':      'SAC-DR',
}
LINESTYLES = {
    'pid':         '-',
    'sac_nominal': '--',
    'sac_dr':      '-.',
}

FAULT_LABELS = {
    '1.0_1.0_1.0_1.0': 'No Fault',
    '1.0_1.0_0.7_1.0': '30% Loss — Motor 2',
    '0.8_0.8_0.8_0.8': '20% Loss — All Motors',
    '0.6_0.6_0.6_0.6': '40% Loss — All Motors',
}


def load_logs(log_dir: str, fault_filter: str = None) -> list[pd.DataFrame]:
    """Load all CSVs from log_dir, optionally filtered by fault string."""
    pattern = os.path.join(log_dir, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'[!] No CSV files found in {log_dir}')
        return []

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if fault_filter and df['fault'].iloc[0] != fault_filter:
                continue
            df['_file'] = os.path.basename(f)
            dfs.append(df)
            print(f'  Loaded: {os.path.basename(f)}  ({len(df)} rows)')
        except Exception as e:
            print(f'  [WARN] Could not load {f}: {e}')
    return dfs


def plot_time_series(dfs: list[pd.DataFrame], out_dir: str, save: bool):
    """
    Time-series hover error plot for each unique fault scenario.
    One figure per fault value.
    """
    if not dfs:
        print('[!] No data to plot.')
        return

    all_faults = sorted(set(df['fault'].iloc[0] for df in dfs))

    for fault in all_faults:
        fault_dfs = [df for df in dfs if df['fault'].iloc[0] == fault]
        fault_label = FAULT_LABELS.get(fault, f'Fault: {fault}')

        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

        for df in fault_dfs:
            ctrl = df['controller'].iloc[0]
            color = COLORS.get(ctrl, '#555555')
            label = LABELS.get(ctrl, ctrl)
            ls    = LINESTYLES.get(ctrl, '-')

            ax.plot(df['time_s'], df['hover_error'],
                    color=color, lw=1.2, ls=ls,
                    label=label, alpha=0.85)

        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Hover Error (m)', fontsize=9)
        ax.set_title(f'Hardware Flight — {fault_label}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        fname = f'hw_timeseries_{fault}.png'
        if save:
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            print(f'  Saved → {path}')
        else:
            plt.show()
        plt.close(fig)


def plot_bar_comparison(dfs: list[pd.DataFrame], out_dir: str, save: bool):
    """
    Bar chart: mean hover error per controller per fault scenario.
    Publication-ready IEEE style.
    """
    if not dfs:
        return

    # Aggregate stats
    records = []
    for df in dfs:
        ctrl  = df['controller'].iloc[0]
        fault = df['fault'].iloc[0]
        errors = df['hover_error'].values
        records.append({
            'controller': ctrl,
            'fault':      fault,
            'mean_err':   np.mean(errors),
            'std_err':    np.std(errors),
        })

    stats = pd.DataFrame(records)
    faults = sorted(stats['fault'].unique())
    controllers = ['pid', 'sac_nominal', 'sac_dr']
    controllers = [c for c in controllers if c in stats['controller'].values]

    x = np.arange(len(faults))
    n = len(controllers)
    width = 0.22

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN - 0.5))

    for i, ctrl in enumerate(controllers):
        sub = stats[stats['controller'] == ctrl].set_index('fault')
        means = [sub.loc[f, 'mean_err'] if f in sub.index else 0.0 for f in faults]
        stds  = [sub.loc[f, 'std_err']  if f in sub.index else 0.0 for f in faults]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width,
                      label=LABELS.get(ctrl, ctrl),
                      color=COLORS.get(ctrl, '#999'),
                      yerr=stds, capsize=3,
                      error_kw={'elinewidth': 0.8, 'ecolor': 'black'})
        # Value labels on bars
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{m:.3f}', ha='center', va='bottom',
                        fontsize=6.5, rotation=90)

    fault_tick_labels = [FAULT_LABELS.get(f, f) for f in faults]
    ax.set_xticks(x)
    ax.set_xticklabels(fault_tick_labels, fontsize=8, rotation=12, ha='right')
    ax.set_ylabel('Mean Hover Error (m)', fontsize=9)
    ax.set_title('Hardware: Controller Comparison Under Motor Faults',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    fname = 'hw_bar_comparison.png'
    if save:
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print(f'  Saved → {path}')
    else:
        plt.show()
    plt.close(fig)


def print_summary(dfs: list[pd.DataFrame]):
    """Print a clean text summary table."""
    if not dfs:
        return
    print('\n' + '═' * 65)
    print(f'  {"Controller":<14} {"Fault":<25} {"Mean Err":>9} {"Std":>7} {"Max":>7}')
    print('─' * 65)
    for df in sorted(dfs, key=lambda d: (d['fault'].iloc[0], d['controller'].iloc[0])):
        ctrl  = df['controller'].iloc[0]
        fault = df['fault'].iloc[0]
        errs  = df['hover_error'].values
        print(f'  {ctrl:<14} {FAULT_LABELS.get(fault, fault):<25} '
              f'{np.mean(errs):>9.4f} {np.std(errs):>7.4f} {np.max(errs):>7.4f}')
    print('═' * 65 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Plot hardware experiment results')
    parser.add_argument('--fault', default=None,
                        help='Filter by fault string, e.g. 1.0_1.0_0.7_1.0')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to hardware_exp/plots/ instead of showing')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir    = os.path.join(script_dir, 'logs')
    out_dir    = os.path.join(script_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n  Loading logs from: {log_dir}')
    dfs = load_logs(log_dir, fault_filter=args.fault)
    if not dfs:
        print('  No data found. Run hw_logger.py first.')
        return

    print_summary(dfs)
    plot_time_series(dfs, out_dir, save=args.save)
    plot_bar_comparison(dfs, out_dir, save=args.save)
    print('  Done.\n')


if __name__ == '__main__':
    main()
