'''
Generic plotting tool:

Example use cases:
plot_curves('active_learning',title='Active learner learning curve', xlabel='# points',ylabel='%', ActiveLearner=[1,2,3,4])
plot_curves('comparison_learning',title='Active learner vs Passive learning curve', xlabel='# points',ylabel='%', ActiveLearner=[1,2,3,4], PassiveLearner=[1,2,3,4])
plot_curves('potatoes',title='Cumulative # potatoes eaten', xlabel='Days', ylabel='Potatoes', GreenPotatoes=[1,2,3,4])
'''

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import MaxNLocator

def plot_curves(file_name, title = '', xlabel = '', ylabel = '', **kwargs):
    fig = host_subplot(111)
    for name, value in kwargs.items():
      fig.plot(value, label = name)
    plt.legend()
    x_ax = fig.axes.get_xaxis()  ## Get X axis
    x_ax.set_major_locator(MaxNLocator(integer=True))
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=16)
    plt.savefig(file_name+'.png')