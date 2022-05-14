# Libraries and Classes
import numpy as np
import FACL
from Agent import Agent
from VDControl import VDControl
import time
#
# This driver program is used for training an agent and then playing a series of games
# First the fuzzy inference (FIS) system is setup (number of membership functions etc)
# Then the type of training is selected: either fuzzy Q learning or fuzzy actor critic
# The controller is made and then plugged into an agent object. It is trained for a # of epochs
# Infomation like the time it took to train and the reward plot are shown after. Finally a new
# agent and controller object a made using the saved data and then they play a game 100 times, recording
# the outcome of each game (success of fail)

# This driver program looks at 

# General Fuzzy Parameters
state = [0, 0] # start position on the grid. make random later
state_max = [10, 10] # max values of the grid [x,y]
state_min = [-10, -10] # smallest value of the grid [x,y]
num_of_mf = [9, 9] # breaking up the state space (grid in this case) into 29 membership functions



########## TRAINING SECTION ###############

start = time.time() # used to see how long the training time took
FACLcontroller = VDControl(state, state_max, state_min, num_of_mf) #create the FACL controller
sharon = Agent(FACLcontroller) # create the agent with the above controller
#print out all the rule sets
print("rules:")
print(sharon.controller.rules)
for i in range(3000):
    # self.controller.reset()
    # for i in range(self.training_iterations_max):
    #     self.controller.iterate_train()
    #     if (self.controller.distance_from_target() < self.controller.r):  ##change to a check capture / completion function later
    #         self.success += 1
    #         break
    # self.controller.updates_after_an_epoch()
    # self.reward_total.append(self.reward_sum_for_a_single_epoch())
    sharon.run_one_epoch()
    if (i % 100 == 0):
        print(i)
        print("time:", time.time()-start)
        print("xy path",sharon.controller.path) #numerical values of path
        #print("input, ut:", sharon.controller.input)

end = time.time()
print('total train time : ', end-start)
print(' total num of successes during training : ', sharon.success)

# Print the path that our agent sharon took in her last epoch
print("xy path",sharon.controller.path) #numerical values of path
print("input, ut:" , sharon.controller.input)
sharon.print_path() #graph
sharon.print_reward_graph()

sharon.save_epoch_training_info() #save all the important info from our training sesh


