import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import argparse

def plot_ieee(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Configure IEEE style settings
    # Font: Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # Font sizes
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 9
    
    # IEEE Double Column Width is ~7.16 inches. 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 5.0), dpi=300)
    
    # Line configurations for distinction
    # Using color + linestyle to be readable even in grayscale
    styles = {
        'PID': {'color': 'blue', 'linestyle': '-', 'linewidth': 1.5, 'label': 'PID Controller'},
        'SAC_Nominal': {'color': 'green', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Nominal SAC'},
        'SAC_DR': {'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Domain-Randomized SAC'}
    }
    
    # Group by agent
    agents = df['agent'].unique()
    
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        
        # Calculate time (assuming 48 Hz control freq as per custom_hover_env.py)
        time = agent_data['step'] / 48.0
        
        # Z-Error
        ax1.plot(time, agent_data['z_error'], 
                 color=styles[agent]['color'], 
                 linestyle=styles[agent]['linestyle'], 
                 linewidth=styles[agent]['linewidth'],
                 label=styles[agent]['label'])
                 
        # XY Drift (Euclidean distance from origin)
        xy_drift = np.sqrt(agent_data['x']**2 + agent_data['y']**2)
        ax2.plot(time, xy_drift, 
                 color=styles[agent]['color'], 
                 linestyle=styles[agent]['linestyle'], 
                 linewidth=styles[agent]['linewidth'],
                 label=styles[agent]['label'])

    # Format Top Panel (Z-Error)
    ax1.set_ylabel('Altitude Error (m)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.axhline(0, color='gray', linewidth=1.0, alpha=0.5)
    
    # Adjust Y-axis tick spacing to 0.2 (as requested in a previous conversation)
    # Finding min/max
    max_z = df['z_error'].max()
    min_z = df['z_error'].min()
    # Ensure 0.2 ticks are used nicely
    yticks1 = np.arange(np.floor(min_z*5)/5, np.ceil(max_z*5)/5 + 0.2, 0.2)
    ax1.set_yticks(yticks1)
    
    # Format Bottom Panel (XY Drift)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Horizontal Drift (m)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    # Put legend at the top of the first plot, spanning columns
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    
    plt.tight_layout()
    # Adjust top to make room for legend
    plt.subplots_adjust(top=0.88)
    
    # Save the figure
    out_name = csv_file.replace('.csv', '_ieee_plot.png')
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0.05)
    print(f"IEEE style plot saved to {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IEEE style plots from experiment CSV")
    parser.add_argument("--csv", type=str, default="(0.741,0.741,1,1).csv", help="Path to the experiment CSV file")
    args = parser.parse_args()
    
    plot_ieee(args.csv)
