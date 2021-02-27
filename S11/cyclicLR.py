import numpy as np
import matplotlib.pyplot as plt

def Cyclic_LR_Plot(lr_min, lr_max, total_iterations, stepsize):
    """
    Function for plotting a cyclic LR profile given the min, max LR and the total number of iterations, alognwith stepsize
    """
    num_cycles = np.floor(1 + (total_iterations / (2 * stepsize)))
    iterations = np.arange(1,total_iterations+1)
    cycles     = [np.floor(1 + (iteration / (2 * stepsize))) for iteration in iterations]
    ratio      = [abs((iteration/stepsize) - (2*cycle) + 1) for iteration,cycle in zip(iterations,cycles)]
    lr_t       = [(lr_min + (lr_max-lr_min) * (1-x)) for x in ratio]
    fig = plt.figure(figsize=(12,8)) 
    ax = fig.add_subplot(111)
    ax.set_title('LR v/s iterations for Cyclic LR', fontsize=18)
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.plot(lr_t,color='b')

    # Makinf upper and lower bounds
    ax.plot([lr_max]*len(iterations),color='r')
    ax.plot([lr_min]*len(iterations),color='g')
    max_str = f'LR_Max={lr_max}'
    min_str = f'LR_Min={lr_min}'
    ax.text(-100, lr_max+((lr_max-lr_min)*0.02), max_str, style='italic', fontsize=12,
        bbox={'facecolor': 'red', 'alpha': 0.2, 'pad': 5})
    ax.text(-100, lr_min-((lr_max-lr_min)*0.02), min_str, style='italic', fontsize=12,
        bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 5})
    
    # Annotation for stepsize
    plt.vlines(x=0, ymin=lr_min + (lr_max+lr_min)*0.38, ymax=lr_max - (lr_max+lr_min)*0.38)
    plt.vlines(x=stepsize, ymin=lr_min + (lr_max+lr_min)*0.38, ymax=lr_max - (lr_max+lr_min)*0.38)
    plt.hlines(y=(lr_max+lr_min)/2, xmin=0, xmax=stepsize)
    step_str = f'StepSize={stepsize}'
    ax.text(0.25*stepsize,((lr_max+lr_min)/2 +((lr_max-lr_min)*0.02)), step_str, fontsize=12,
        bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 1})
    
    # Annotation for Cycle
    plt.vlines(x=0, ymin=lr_min + (lr_max+lr_min)*0.68, ymax=lr_min + (lr_max+lr_min)*0.78)
    plt.vlines(x=2*stepsize, ymin=lr_min + (lr_max+lr_min)*0.68, ymax=lr_min + (lr_max+lr_min)*0.78)
    plt.hlines(y=(lr_min + (lr_max+lr_min)*0.73), xmin=0, xmax=2*stepsize)
    cycle_str = f'1 Cycle={2*stepsize}'
    ax.text(0.7*stepsize,(lr_min + (lr_max+lr_min)*0.74), cycle_str, fontsize=12,
        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 1})
    my_dpi = 100
    fig.savefig('S11/images/cyclicLRschedule.png',figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    
