# Libraries and Classes
import numpy as np
import random
from Agent import Agent
from VDControl import VDControl
import time
import matplotlib.pyplot as plt


#Define some functions
def plot_both_velocities():
    fig, ax = plt.subplots()
    plt.title('sharon and diana final epoch velocity')
    ax.plot(sharon.controller.velocity_path,label='sharon')
    ax.plot(diana.controller.velocity_path,label='diana')
    plt.xlabel('time (10ms)')
    plt.ylabel('velocity')
    plt.legend()
    plt.show()