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
state = [2, 2] # start position on the grid. make random later
state_max = [10, 10] # max values of the grid [x,y]
state_min = [-10, -10] # smallest value of the grid [x,y]
num_of_mf = [9, 9] # breaking up the state space (grid in this case) into 29 membership functions



########## TRAINING SECTION ###############
# two agents: sharon and diane

start = time.time() # used to see how long the training time took
Sharon_FACLcontroller = VDControl(state, state_max, state_min, num_of_mf) #create the FACL controller
Diana_FACLcontroller = VDControl([0,0],state_max,state_min,num_of_mf)
sharon = Agent(Sharon_FACLcontroller) # create the agent with the above controller
diana = Agent(Diana_FACLcontroller)

#print out all the rule sets
print("rules:")
print(sharon.controller.rules)

for i in range(1500):
    sharon.controller.reset()
    diana.controller.reset()
    for j in range(sharon.training_iterations_max):
        # sharon.controller.iterate_train()
        # diana.controller.iterate_train()
        if (sharon.controller.state[0] < 10):  ##change to a check capture / completion function later
            sharon.controller.iterate_train()
        else:
            if (sharon.record_success_flag == 0):
                sharon.success += 1
                sharon.record_success_flag = 1


        if (diana.controller.state[0] < 10):  ##change to a check capture / completion function later
            diana.controller.iterate_train()
        else:
            if(diana.record_success_flag == 0):
                diana.success+=1
                diana.record_success_flag = 1
        if (sharon.controller.state[0] >= 10 and diana.controller.state[0]>=10):
            break
    sharon.controller.updates_after_an_epoch()
    sharon.record_success_flag = 0
    sharon.reward_total.append(sharon.reward_sum_for_a_single_epoch())
    diana.controller.updates_after_an_epoch()
    diana.record_success_flag = 0
    diana.reward_total.append(diana.reward_sum_for_a_single_epoch())
    # sharon.run_one_epoch()
    if (i % 100 == 0):
        print(i)
        print("time:", time.time()-start)
        print("xy path of sharon",sharon.controller.path) #numerical values of path
    if (i % 100 == 0):
        print(i)
        print("time:", time.time() - start)
        print("xy path of diana", diana.controller.path)  # numerical values of path
        #print("input, ut:", sharon.controller.input)

end = time.time()
print('total train time : ', end-start)
print(' total num of successes during training : ', sharon.success)

# Print the path that our agent sharon took in her last epoch
#print("xy path",sharon.controller.path) #numerical values of path
print("input, ut:" , sharon.controller.input)
# sharon.print_path()
sharon.print_reward_graph()
diana.print_reward_graph()
sharon.save_epoch_training_info() #save all the important info from our training sesh


