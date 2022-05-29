# Libraries and Classes
import numpy as np
import random
from Agent import Agent
from VDControl import VDControl
import time
import matplotlib.pyplot as plt
import os.path as osp

#
# This driver program is used for training an agent and then playing a series of games
# First the fuzzy inference (FIS) system is setup (number of membership functions etc)
# Then the type of training is selected: either fuzzy Q learning or fuzzy actor critic
# The controller is made and then plugged into an agent object. It is trained for a # of epochs
# Infomation like the time it took to train and the reward plot are shown after. Finally a new
# agent and controller object a made using the saved data and then they play a game 100 times, recording
# the outcome of each game (success of fail)
def plot_both_velocities():
    fig, ax = plt.subplots()
    plt.title('sharon and diana final epoch velocity')
    ax.plot(sharon.controller.velocity_path,label='sharon')
    ax.plot(diana.controller.velocity_path,label='diana')
    plt.xlabel('time (10ms)')
    plt.ylabel('velocity')
    plt.legend()
    fig.savefig('velocity plot.png')  # save the figure to file
    plt.show()
    # plt.close()
# This driver program looks at

# General Fuzzy Parameters
state = [2,0,0,0] # Sharon position, Sharon Velocity, Diana position, Diana Velcoity
state_max = [12, 10, 12, 10] # max values of the grid [x,y]
state_min = [-12, -5, -12, -5]  # smallest value of the grid [x,y]
num_of_mf = [7, 5, 7, 5]  # breaking up the state space (grid in this case) into 29 membership functions
number_of_ties = 0
starting_pos = [0,  1,  2, 3, 4]
tol = 0.75
########## TRAINING SECTION ###############
# two agents: sharon and diane

start = time.time() # used to see how long the training time took
Sharon_FACLcontroller = VDControl([2,0,0,0], state_max, state_min, num_of_mf) #create the FACL controller
Diana_FACLcontroller = VDControl([0,0,2,0],state_max,state_min,num_of_mf)
sharon = Agent(Sharon_FACLcontroller) # create the agent with the above controller
diana = Agent(Diana_FACLcontroller)

#print out all the rule sets
#print("rules:")
#print(sharon.controller.rules)

rolling_success_counter = 0
cycle_counter = 0
for i in range(10000):
    sharon.controller.reset()
    diana.controller.reset()

    # Start at new positions each time
    sharon_start = random.choice(starting_pos)
    diana_start = random.choice(starting_pos)
    sharon.controller.state[0] = sharon_start
    diana.controller.state[0] = diana_start
    sharon.controller.state[2] = diana_start
    diana.controller.state[2] = sharon_start
    diana.controller.distance_away_from_target_t = diana.controller.distance_from_target()
    sharon.controller.distance_away_from_target_t = sharon.controller.distance_from_target()
    cycle_counter += 1
    for j in range(sharon.training_iterations_max):
        # sharon.controller.iterate_train()
        # diana.controller.iterate_train()
        if (sharon.controller.state[0] < sharon.controller.finish_line and diana.controller.state[0] <  diana.controller.finish_line):  ##if both havent crossed the finish line, train


            sharon.controller.generate_noise()
            diana.controller.generate_noise()

            # Step 3 :  calculate the necessary action
            sharon.controller.calculate_ut()
            diana.controller.calculate_ut()

            # Step 4: calculate the value function at current iterate/time step
            sharon.controller.v_t = sharon.controller.calculate_vt(sharon.controller.phi) #  v_t = sum of self.phi[l] * self.zeta[l]
            diana.controller.v_t = diana.controller.calculate_vt(diana.controller.phi)

            # Step 5: update the state of the system
            max = 3
            min = -3
            if (sharon.controller.u_t > max):
                sharon.controller.u_t = max
            elif (sharon.controller.u_t < min):
                sharon.controller.u_t = min
            if (diana.controller.u_t > max):
                diana.controller.u_t = max
            elif (diana.controller.u_t < min):
                diana.controller.u_t = min

            sharon.controller.a = (1 / sharon.controller.m) * (sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])
            sharon.controller.state[1] = sharon.controller.state[1] + sharon.controller.a * sharon.controller.dt
            diana.controller.a = (1 / diana.controller.m) * (
                        diana.controller.u_t - diana.controller.b * diana.controller.state[1])
            diana.controller.state[1] = diana.controller.state[1] + diana.controller.a * diana.controller.dt
            for t in range(10):
                sharon.controller.state[0] = sharon.controller.state[0] + sharon.controller.state[1] * sharon.controller.dt
                sharon.controller.state[1] = sharon.controller.state[1] + sharon.controller.a * sharon.controller.dt
                sharon.controller.a = (1 / sharon.controller.m) * (
                            sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])

            for t in range(10):
                diana.controller.state[0] = diana.controller.state[0] + diana.controller.state[
                    1] * diana.controller.dt
                diana.controller.state[1] = diana.controller.state[1] + diana.controller.a * diana.controller.dt
                diana.controller.a = (1 / diana.controller.m) * (
                        diana.controller.u_t - diana.controller.b * diana.controller.state[1])



            sharon.controller.state[2] = diana.controller.state[0]
            sharon.controller.state[3] = diana.controller.state[1]
            diana.controller.state[2] = sharon.controller.state[0]
            diana.controller.state[3] = sharon.controller.state[1]

            sharon.controller.v = sharon.controller.state[1]
            sharon.controller.update_path(sharon.controller.state)
            sharon.controller.update_v_path(sharon.controller.state[1])
            sharon.controller.update_input_array(sharon.controller.u_t)
            diana.controller.v = diana.controller.state[1]
            diana.controller.update_path(diana.controller.state)
            diana.controller.update_v_path(diana.controller.state[1])
            diana.controller.update_input_array(diana.controller.u_t)

            #sharon.controller.update_state()
            sharon.controller.phi_next = sharon.controller.update_phi()
            #diana.controller.update_state()
            diana.controller.phi_next = diana.controller.update_phi()

            # Step 6: get reward,
            # sharon.controller.reward = sharon.controller.get_reward()
            # sharon.controller.reward = diana.controller.get_reward()
            sharon.controller.distance_away_from_target_t_plus_1 = sharon.controller.distance_from_target()
            diana.controller.distance_away_from_target_t_plus_1 = diana.controller.distance_from_target()
            if (sharon.controller.state[0] >= sharon.controller.finish_line):
                sharon.controller.reward = 50
                if(diana.controller.state[0]>=diana.controller.finish_line-tol):
                    diana.controller.reward = 50
                else:
                    diana.controller.reward = 0
                sharon.controller.reward = sharon.controller.reward*0.8 + 0.2*(sharon.controller.reward+diana.controller.reward)
                diana.controller.reward = diana.controller.reward*0.8 + 0.2*(sharon.controller.reward+diana.controller.reward)

            elif(diana.controller.state[0]>=diana.controller.finish_line):
                diana.controller.reward = 50
                if (sharon.controller.state[0] >= sharon.controller.finish_line - tol):
                    sharon.controller.reward = 50
                else:
                    sharon.controller.reward=0
                sharon.controller.reward = sharon.controller.reward * 0.8 + 0.2 * (
                            sharon.controller.reward + diana.controller.reward)
                diana.controller.reward = diana.controller.reward * 0.8 + 0.2 * (
                            sharon.controller.reward + diana.controller.reward)
            else:
                sharon.controller.reward =  (sharon.controller.distance_away_from_target_t - sharon.controller.distance_away_from_target_t_plus_1)
                diana.controller.reward =  (diana.controller.distance_away_from_target_t - diana.controller.distance_away_from_target_t_plus_1)
            # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
            sharon.controller.distance_away_from_target_t = sharon.controller.distance_away_from_target_t_plus_1
            diana.controller.distance_away_from_target_t = diana.controller.distance_away_from_target_t_plus_1

            sharon.controller.update_reward_graph(sharon.controller.reward)
            diana.controller.update_reward_graph(diana.controller.reward)

            # Step 7: Calculate the expected value for the next step
            sharon.controller.v_t_1 = sharon.controller.calculate_vt(sharon.controller.phi_next) # self.phi[l] * self.zeta[l]
            diana.controller.v_t_1 = diana.controller.calculate_vt(diana.controller.phi_next)

            # Step 8: calculate the temporal difference
            #No VD
            sharon.controller.calculate_prediction_error()
            diana.controller.calculate_prediction_error()

            # Regular VD
            # sharon.controller.temporal_difference = (sharon.controller.reward+diana.controller.reward) + sharon.controller.gamma * (sharon.controller.v_t_1 + diana.controller.v_t_1) - (sharon.controller.v_t+diana.controller.v_t)
            # diana.controller.temporal_difference = (diana.controller.reward+sharon.controller.reward) + diana.controller.gamma * (diana.controller.v_t_1+sharon.controller.v_t_1) - (diana.controller.v_t+sharon.controller.v_t)

            # A weighted version of VD
            # w=0.75
            # sharon.controller.temporal_difference = (
            #                                                     w*sharon.controller.reward + (1-w)*diana.controller.reward) + sharon.controller.gamma * (
            #                                                     w*sharon.controller.v_t_1 + (1-w)*diana.controller.v_t_1) - (
            #                                                     w*sharon.controller.v_t + (1-w)*diana.controller.v_t)
            # diana.controller.temporal_difference = (
            #                                                    w*diana.controller.reward + (1-w)*sharon.controller.reward) + diana.controller.gamma * (
            #                                                    w*diana.controller.v_t_1 + (1-w)*sharon.controller.v_t_1) - (
            #                                                    w*diana.controller.v_t + (1-w)*sharon.controller.v_t)

            # Step 9: update the actor and critic functions
            sharon.controller.update_zeta() # update the critic
            sharon.controller.update_omega() # update the actor
            diana.controller.update_zeta()  # update the critic
            diana.controller.update_omega()  # update the actor

            sharon.controller.phi = sharon.controller.phi_next
            diana.controller.phi = diana.controller.phi_next

        else: #if an agent has crossed the line
            if (sharon.controller.state[0] >= sharon.controller.finish_line and diana.controller.state[
                0] >= diana.controller.finish_line):
                number_of_ties += 1
                rolling_success_counter+=1
                # plot_both_velocities()
                break
            if (sharon.controller.state[0] >= sharon.controller.finish_line):
                sharon.success += 1
                if(diana.controller.state[0]>=diana.controller.finish_line-tol):
                    sharon.success-=1
                    number_of_ties+=1
                    rolling_success_counter+=1
                    # plot_both_velocities()
                break
            if (diana.controller.state[0] >= diana.controller.finish_line):
                diana.success += 1
                if (sharon.controller.state[0] >= sharon.controller.finish_line - tol):
                    diana.success -= 1
                    number_of_ties += 1
                    rolling_success_counter +=1
                    # plot_both_velocities()
                break

            break


    sharon.controller.updates_after_an_epoch()
    sharon.reward_total.append(sharon.reward_sum_for_a_single_epoch())
    diana.controller.updates_after_an_epoch()
    diana.reward_total.append(diana.reward_sum_for_a_single_epoch())

    # check to see if we should stop training based on a rolling counter
    # if we hit 2k consecutive successful training rounds, then stop training
    if(rolling_success_counter != cycle_counter):
        cycle_counter=0
        rolling_success_counter=0

    if (rolling_success_counter >= 1000):
        print('number of epochs trained: ', i)
        break
    # print out some stats as it trains every so often
    if (i % 1000 == 0):
        print(i)
        print("time:", time.time()-start)
        print("xy path of sharon",sharon.controller.path[len(sharon.controller.path)-1]) #numerical values of path
        print("xy path of diana", diana.controller.path[len(diana.controller.path)-1])  # numerical values of path
        print('length of game', len(diana.controller.path))
        print('sigma ', diana.controller.sigma)
        print('sharon wins : ', sharon.success)
        print('diana wins ', diana.success)
        print('ties ', number_of_ties)
        print('number of consecutive ties in a row', rolling_success_counter)

        #print("input, ut:", sharon.controller.input)

end = time.time()
print('total train time : ', end-start)
print(' total num of successes during training for sharon : ', sharon.success)
print(' total num of successes during training for diana : ', diana.success)
print('total number of ties', number_of_ties)
# Print the path that our agent sharon took in her last epoch
#print("xy path",sharon.controller.path) #numerical values of path
print("input, ut:" , sharon.controller.input)

fig, ax = plt.subplots()
plt.title('sharon and diana sum of rewards per epoch')
ax.plot(sharon.reward_total,label='sharon')
ax.plot(diana.reward_total,label='diana')
plt.xlabel('epoch')
plt.ylabel('total rewards per epoch')
plt.legend()
plt.show()
fig.savefig('reward plot.png')   # save the figure to file
plt.close(fig)    # close the figure window
plot_both_velocities()

#Save the training data

userdoc = osp.join(osp.expanduser("~"),'/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/sharon')
np.savetxt(osp.join(userdoc, "%s.csv" % 'critic_weights'),sharon.controller.zeta)
np.savetxt(osp.join(userdoc, "%s.txt" % 'actor_weights'),sharon.controller.omega)
userdoc = osp.join(osp.expanduser("~"),'/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/diana')
np.savetxt(osp.join(userdoc, "%s.csv" % 'critic_weights'),diana.controller.zeta)
np.savetxt(osp.join(userdoc, "%s.txt" % 'actor_weights'),diana.controller.omega)
sharon.success = 0
diana.success = 0
number_of_ties=0
sharon.controller.sigma = 0.15
diana.controller.sigma = 0.15

#########################################################
#        Run a series of games with the trained agent
#########################################################

for i in range(100):
    sharon.controller.reset()
    diana.controller.reset()
    sharon_start = random.choice(starting_pos)
    diana_start = random.choice(starting_pos)
    sharon.controller.state[0] = sharon_start
    diana.controller.state[0] = diana_start
    sharon.controller.state[2] = diana_start
    diana.controller.state[2] = sharon_start
    diana.controller.distance_away_from_target_t = diana.controller.distance_from_target()
    sharon.controller.distance_away_from_target_t = sharon.controller.distance_from_target()
    for j in range(sharon.training_iterations_max):

        if (sharon.controller.state[0] < sharon.controller.finish_line and diana.controller.state[0] <  diana.controller.finish_line):  ##if both havent crossed the finish line, train
            sharon.controller.generate_noise()
            diana.controller.generate_noise()
            # Step 1 :  calculate the necessary action
            sharon.controller.calculate_ut()
            diana.controller.calculate_ut()

            # Step 2: update the state of the system
            max = 3
            min = -3
            if (sharon.controller.u_t > max):
                sharon.controller.u_t = max
            elif (sharon.controller.u_t < min):
                sharon.controller.u_t = min
            if (diana.controller.u_t > max):
                diana.controller.u_t = max
            elif (diana.controller.u_t < min):
                diana.controller.u_t = min

            sharon.controller.a = (1 / sharon.controller.m) * (sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])
            sharon.controller.state[1] = sharon.controller.state[1] + sharon.controller.a * sharon.controller.dt
            diana.controller.a = (1 / diana.controller.m) * (
                        diana.controller.u_t - diana.controller.b * diana.controller.state[1])
            diana.controller.state[1] = diana.controller.state[1] + diana.controller.a * diana.controller.dt
            for t in range(10):
                sharon.controller.state[0] = sharon.controller.state[0] + sharon.controller.state[
                    1] * sharon.controller.dt
                sharon.controller.state[1] = sharon.controller.state[1] + sharon.controller.a * sharon.controller.dt
                sharon.controller.a = (1 / sharon.controller.m) * (
                        sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])

            for t in range(10):
                diana.controller.state[0] = diana.controller.state[0] + diana.controller.state[
                    1] * diana.controller.dt
                diana.controller.state[1] = diana.controller.state[1] + diana.controller.a * diana.controller.dt
                diana.controller.a = (1 / diana.controller.m) * (
                        diana.controller.u_t - diana.controller.b * diana.controller.state[1])


            sharon.controller.state[2] = diana.controller.state[0]
            sharon.controller.state[3] = diana.controller.state[1]
            diana.controller.state[2] = sharon.controller.state[0]
            diana.controller.state[3] = sharon.controller.state[1]

            sharon.controller.phi_next = sharon.controller.update_phi()
            sharon.controller.phi = sharon.controller.phi_next
            diana.controller.phi_next = diana.controller.update_phi()
            diana.controller.phi = diana.controller.phi_next

        else:  # if an agent has crossed the line
            print('\n')
            print('game played : ' , i)
            print('sharon start: ', sharon_start)
            print('sharon finish', sharon.controller.state[0])
            print('diana start: ', diana_start)
            print('diana finish', diana.controller.state[0])
            # plot_both_velocities()
            if (sharon.controller.state[0] >= sharon.controller.finish_line and diana.controller.state[
                0] >= diana.controller.finish_line):
                number_of_ties += 1
                # plot_both_velocities()
                break
            if (sharon.controller.state[0] >= sharon.controller.finish_line):
                sharon.success += 1
                if (diana.controller.state[0] >= diana.controller.finish_line - tol):
                    sharon.success -= 1
                    number_of_ties += 1
                    # plot_both_velocities()
                break
            if (diana.controller.state[0] >= diana.controller.finish_line):
                diana.success += 1
                if (sharon.controller.state[0] >= sharon.controller.finish_line - tol):
                    diana.success -= 1
                    number_of_ties += 1
                    # plot_both_velocities()
                break


print('GAME STATS ')
print('sharon wins : ', sharon.success)
print('diana wins ', diana.success)
print('ties ', number_of_ties)

# print('last game played')
# print('sharon start: ', sharon_start)
# print('sharon finish', sharon.controller.state[0])
# print('diana start: ', diana_start)
# print('diana finish', diana.controller.state[0])
# sharon.save_epoch_training_info() #save all the important info from our training sesh



#Print out the reward plots combined
# fig, ax = plt.subplots()
# plt.title('sharon and diana rewards per epoch')
# ax.plot(sharon.reward_total,label='sharon')
# ax.plot(diana.reward_total,label='diana')
# plt.xlabel('epoch')
# plt.ylabel('total rewards per epoch')
# plt.legend()
# plt.show()
#
# sharon.print_reward_graph()
# diana.print_reward_graph()

# plot_both_velocities()
# sharon.print_velocity_graph()
# diana.print_velocity_graph()
