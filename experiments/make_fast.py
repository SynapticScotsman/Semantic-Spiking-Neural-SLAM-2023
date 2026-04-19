with open('experiments/slam_map_new.py', 'r') as f:
    code = f.read()

# Reduce simulation time from 60 to 10 seconds for a fast demonstration
code = code.replace('T = 60', 'T = 10')
# Must increase high pass freq to avoid ValueError when T decreases
code = code.replace('high=.05', 'high=0.1')

# Avoid blocking GUI calls, save to PNG instead
code = code.replace('plt.show()', "plt.savefig('slam_plot_env.png')")
code = code.replace('fig.show()', "fig.savefig('slam_plot_results.png')\nplt.close('all')")

with open('experiments/slam_map_fast.py', 'w') as f:
    f.write(code)
