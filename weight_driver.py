#### This file is to basically to run a bunch of different weights and see the results
import numpy as np
import random
from Agent import Agent
from VDControl import VDControl
import time
import matplotlib.pyplot as plt
import os.path as osp

beginning = time.time()

# Define the function
def run_epoch_and_games(w, directory_path, training_text_name, num_games, num_epochs):
    print("w = ", w)
    print("time", time.time() - beginning)
    # General Fuzzy Parameters
    state = [2, 0, 0, 0]  # Sharon position, Sharon Velocity, Diana position, Diana Velcoity
    state_max = [12, 10, 20, 10]  # max values of the grid [x,y]
    state_min = [-12, -10, -20, -10]  # smallest value of the grid [x,y]
    num_of_mf = [8, 5, 8, 5]  # breaking up the state space (grid in this case) into 29 membership functions
    number_of_ties = 0
    starting_pos = [0, 1, 2, 3]
    # state_max = [12, 10, 20, 10]  # max values of the grid [x,y]
    # state_min = [-12, -5, -20, -5]  # smallest value of the grid [x,y]
    # num_of_mf = [8, 5, 8, 5]  # breaking up the state space (grid in this case) into 29 membership functions
    # number_of_ties = 0
    # starting_pos = [0, 1, 2, 3]
    tol = 0.8
    ########## TRAINING SECTION ###############
    # two agents: sharon and diane

    start = time.time()  # used to see how long the training time took
    Sharon_FACLcontroller = VDControl([2, 0, 2, 0], state_max, state_min, num_of_mf)  # create the FACL controller
    Diana_FACLcontroller = VDControl([0, 0, 2, 0], state_max, state_min, num_of_mf)
    sharon = Agent(Sharon_FACLcontroller)  # create the agent with the above controller
    diana = Agent(Diana_FACLcontroller)

    # File I/Os
    f = open(training_text_name, "a")
    #d = open('Games')


    rolling_success_counter = 0
    cycle_counter = 0
    for i in range(num_epochs):
        sharon.controller.reset()
        diana.controller.reset()

        # Start at new positions each time
        sharon_start = random.choice(starting_pos)
        diana_start = random.choice(starting_pos)
        sharon.controller.state[0] = sharon_start
        diana.controller.state[0] = diana_start
        sharon.controller.state[2] = sharon_start - diana_start
        diana.controller.state[2] = diana_start - sharon_start
        diana.controller.distance_away_from_target_t = diana.controller.distance_from_target()
        sharon.controller.distance_away_from_target_t = sharon.controller.distance_from_target()
        cycle_counter += 1
        for j in range(sharon.training_iterations_max):
            # sharon.controller.iterate_train()
            # diana.controller.iterate_train()
            if (sharon.controller.state[0] < sharon.controller.finish_line and diana.controller.state[
                0] < diana.controller.finish_line):  ##if both havent crossed the finish line, train

                sharon.controller.generate_noise()
                diana.controller.generate_noise()

                # Step 3 :  calculate the necessary action
                sharon.controller.calculate_ut()
                diana.controller.calculate_ut()

                # Step 4: calculate the value function at current iterate/time step
                sharon.controller.v_t = sharon.controller.calculate_vt(
                    sharon.controller.phi)  # v_t = sum of self.phi[l] * self.zeta[l]
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

                sharon.controller.a = (1 / sharon.controller.m) * (
                            sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])
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

                sharon.controller.state[2] = sharon.controller.state[0] - diana.controller.state[0]
                sharon.controller.state[3] = diana.controller.state[1]
                diana.controller.state[2] = diana.controller.state[0] - sharon.controller.state[0]
                diana.controller.state[3] = sharon.controller.state[1]

                sharon.controller.v = sharon.controller.state[1]
                sharon.controller.update_path(sharon.controller.state)
                sharon.controller.update_v_path(sharon.controller.state[1])
                sharon.controller.update_input_array(sharon.controller.u_t)
                diana.controller.v = diana.controller.state[1]
                diana.controller.update_path(diana.controller.state)
                diana.controller.update_v_path(diana.controller.state[1])
                diana.controller.update_input_array(diana.controller.u_t)

                if(diana.controller.state[0] <-12 or sharon.controller.state[0]<-12):
                    print(i," pos state went below")
                    break
                if (diana.controller.state[1] < -10 or sharon.controller.state[1] < -10):
                    print(i," vel state went below")
                    break
                if (diana.controller.state[1] >10 or sharon.controller.state[1] > 10):
                    print(i," vel state went above")
                    break
                if (diana.controller.state[3] >20 or sharon.controller.state[3] > 20):
                    print(i," delta state went above")
                    break
                if (diana.controller.state[3] <-20 or sharon.controller.state[3] <-20):
                    print(i," delta state went below")
                    break
                # sharon.controller.update_state()
                sharon.controller.phi_next = sharon.controller.update_phi()
                # diana.controller.update_state()
                diana.controller.phi_next = diana.controller.update_phi()

                # Step 6: get reward,
                # sharon.controller.reward = sharon.controller.get_reward()
                # sharon.controller.reward = diana.controller.get_reward()
                sharon.controller.distance_away_from_target_t_plus_1 = sharon.controller.distance_from_target()
                diana.controller.distance_away_from_target_t_plus_1 = diana.controller.distance_from_target()
                if (sharon.controller.state[0] >= sharon.controller.finish_line):
                    sharon.controller.reward = -15  # -10
                    if (diana.controller.state[0] >= diana.controller.finish_line - tol):
                        diana.controller.reward = 50
                        sharon.controller.reward = 50
                    else:
                        diana.controller.reward = 0
                    sharon.controller.reward = sharon.controller.reward * 0.5 + 0.5 * (
                                sharon.controller.reward + diana.controller.reward)
                    diana.controller.reward = diana.controller.reward * 0.5 + 0.5 * (
                                sharon.controller.reward + diana.controller.reward)

                elif (diana.controller.state[0] >= diana.controller.finish_line):
                    diana.controller.reward = -15  # -10
                    if (sharon.controller.state[0] >= sharon.controller.finish_line - tol):
                        sharon.controller.reward = 50
                        diana.controller.reward = 50
                    else:
                        sharon.controller.reward = 0
                    sharon.controller.reward = sharon.controller.reward * 0.5 + 0.5 * (
                            sharon.controller.reward + diana.controller.reward)
                    diana.controller.reward = diana.controller.reward * 0.5 + 0.5 * (
                            sharon.controller.reward + diana.controller.reward)
                else:
                    #w = 0.3
                    sharon.controller.reward = w * (
                                sharon.controller.distance_away_from_target_t - sharon.controller.distance_away_from_target_t_plus_1) + (
                                                           1 - w) * np.exp(
                        -(sharon.controller.state[2]/0.25) ** 2)
                    diana.controller.reward = w * (
                                diana.controller.distance_away_from_target_t - diana.controller.distance_away_from_target_t_plus_1) + (
                                                          1 - w) * np.exp(
                        -(diana.controller.state[2]/0.25) ** 2)
                # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
                sharon.controller.distance_away_from_target_t = sharon.controller.distance_away_from_target_t_plus_1
                diana.controller.distance_away_from_target_t = diana.controller.distance_away_from_target_t_plus_1

                sharon.controller.update_reward_graph(sharon.controller.reward)
                diana.controller.update_reward_graph(diana.controller.reward)

                # Step 7: Calculate the expected value for the next step
                sharon.controller.v_t_1 = sharon.controller.calculate_vt(
                    sharon.controller.phi_next)  # self.phi[l] * self.zeta[l]
                diana.controller.v_t_1 = diana.controller.calculate_vt(diana.controller.phi_next)

                # Step 8: calculate the temporal difference
                # No VD
                sharon.controller.calculate_prediction_error()
                diana.controller.calculate_prediction_error()

                # Step 9: update the actor and critic functions
                sharon.controller.update_zeta()  # update the critic
                sharon.controller.update_omega()  # update the actor
                diana.controller.update_zeta()  # update the critic
                diana.controller.update_omega()  # update the actor

                sharon.controller.phi = sharon.controller.phi_next
                diana.controller.phi = diana.controller.phi_next

            else:  # if an agent has crossed the line
                if (sharon.controller.state[0] >= sharon.controller.finish_line and diana.controller.state[
                    0] >= diana.controller.finish_line):
                    number_of_ties += 1
                    rolling_success_counter += 1
                    # plot_both_velocities()
                    break
                if (sharon.controller.state[0] >= sharon.controller.finish_line):
                    sharon.success += 1
                    if (diana.controller.state[0] >= diana.controller.finish_line - tol):
                        sharon.success -= 1
                        number_of_ties += 1
                        rolling_success_counter += 1
                        # plot_both_velocities()
                    break
                if (diana.controller.state[0] >= diana.controller.finish_line):
                    diana.success += 1
                    if (sharon.controller.state[0] >= sharon.controller.finish_line - tol):
                        diana.success -= 1
                        number_of_ties += 1
                        rolling_success_counter += 1
                        # plot_both_velocities()
                    break

                break

        sharon.controller.updates_after_an_epoch()
        sharon.reward_total.append(sharon.reward_sum_for_a_single_epoch())
        diana.controller.updates_after_an_epoch()
        diana.reward_total.append(diana.reward_sum_for_a_single_epoch())

        # check to see if we should stop training based on a rolling counter
        # if we hit 2k consecutive successful training rounds, then stop training
        if (rolling_success_counter != cycle_counter):
            cycle_counter = 0
            rolling_success_counter = 0

        if (rolling_success_counter >= 100):
            sharon.controller.sigma = 0.3
            diana.controller.sigma = 0.3
        if (rolling_success_counter >= 250):
            sharon.controller.sigma = 0.1
            diana.controller.sigma = 0.1
        if (rolling_success_counter >= 1000):
            f.write("num of epochs trained: ")
            f.write(str(i))
            f.write("\n")
            break
        # print out some stats as it trains every so often
        if (i % 100 == 0):
            print('epoch ', i)
            print("ties ", number_of_ties)
            print('sharon w : ', sharon.success)
            print('diana w : ', diana.success)
            f.write("\n")
            f.write("Epoch : ")
            f.write(str(i))
            f.write("\n")
            f.write("time :")
            f.write(str(time.time()-start))
            f.write("\n")
            # print("xy path of sharon",
            #       sharon.controller.path[len(sharon.controller.path) - 1])  # numerical values of path
            # print("xy path of diana", diana.controller.path[len(diana.controller.path) - 1])  # numerical values of path
            f.write('length of game : ')
            f.write(str(len(diana.controller.path)))
            f.write("\n")
            # print('sigma ', diana.controller.sigma)
            # print('sharon wins : ', sharon.success)
            # print('diana wins ', diana.success)
            # print('ties ', number_of_ties)
            # print('number of consecutive ties in a row', rolling_success_counter)
            f.write('Diana wins : ')
            f.write(str(diana.success))
            f.write("\n")
            f.write('Sharon wins : ')
            f.write(str(sharon.success))
            f.write("\n")
            f.write('Ties : ')
            f.write(str(number_of_ties))
            f.write("\n")
            f.write('Number of consecutive Ties : ')
            f.write(str(rolling_success_counter))
            f.write("\n")
            # print("input, ut:", sharon.controller.input)

    end = time.time()
    f.write('total train time : ')
    f.write(str( end - start))
    f.write("\n")
    f.write(' total num of successes during training for sharon : ')
    f.write( str(sharon.success))
    f.write("\n")
    f.write(' total num of successes during training for diana : ')
    f.write(str(diana.success))
    f.write("\n")
    f.write('total number of ties : ')
    f.write(str(number_of_ties))
    # Print the path that our agent sharon took in her last epoch
    # print("xy path",sharon.controller.path) #numerical values of path
    # print("input, ut:", sharon.controller.input)

    fig, ax = plt.subplots()
    plt.title('sharon and diana sum of rewards per epoch')
    ax.plot(sharon.reward_total, label='sharon')
    ax.plot(diana.reward_total, label='diana')
    plt.xlabel('epoch')
    plt.ylabel('total rewards per epoch')
    plt.legend()
    # plt.show()
    fig.savefig('reward plot.png')  # save the figure to file
    plt.close(fig)  # close the figure window
    # plot_both_velocities()

    # Save the training data

    userdoc = osp.join(osp.expanduser("~"), directory_path)
    np.savetxt(osp.join(userdoc, "%s.csv" % 'sharon_critic_weights'), sharon.controller.zeta)
    np.savetxt(osp.join(userdoc, "%s.txt" % 'sharon_actor_weights'), sharon.controller.omega)
    userdoc = osp.join(osp.expanduser("~"), directory_path)
    np.savetxt(osp.join(userdoc, "%s.csv" % 'diana_critic_weights'), diana.controller.zeta)
    np.savetxt(osp.join(userdoc, "%s.txt" % 'diana_actor_weights'), diana.controller.omega)
    sharon.success = 0
    diana.success = 0
    number_of_ties = 0
    sharon.controller.sigma = 0.15
    diana.controller.sigma = 0.15

    #########################################################
    #        Run a series of games with the trained agent
    #########################################################

    for i in range(num_games):
        sharon.controller.reset()
        diana.controller.reset()
        sharon_start = random.choice(starting_pos)
        diana_start = random.choice(starting_pos)
        sharon.controller.state[0] = sharon_start
        diana.controller.state[0] = diana_start
        sharon.controller.state[2] = sharon_start - diana_start
        diana.controller.state[2] = diana_start - sharon_start
        diana.controller.distance_away_from_target_t = diana.controller.distance_from_target()
        sharon.controller.distance_away_from_target_t = sharon.controller.distance_from_target()
        for j in range(sharon.training_iterations_max):

            if (sharon.controller.state[0] < sharon.controller.finish_line and diana.controller.state[
                0] < diana.controller.finish_line):  ##if both havent crossed the finish line, train
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

                sharon.controller.a = (1 / sharon.controller.m) * (
                            sharon.controller.u_t - sharon.controller.b * sharon.controller.state[1])
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

                sharon.controller.state[2] = sharon.controller.state[0] - diana.controller.state[0]
                sharon.controller.state[3] = diana.controller.state[1]
                diana.controller.state[2] = diana.controller.state[0] - sharon.controller.state[0]
                diana.controller.state[3] = sharon.controller.state[1]

                sharon.controller.phi_next = sharon.controller.update_phi()
                sharon.controller.phi = sharon.controller.phi_next
                diana.controller.phi_next = diana.controller.update_phi()
                diana.controller.phi = diana.controller.phi_next

            else:  # if an agent has crossed the line
                f.write('\n')
                f.write('game played : ')
                f.write(str(i))
                f.write('\n')
                f.write('sharon start: ')
                f.write(str(sharon_start))
                f.write('\n')
                f.write('sharon finish point')
                f.write(str(sharon.controller.state[0]))
                f.write('\n')
                f.write('diana start: ')
                f.write(str(diana_start))
                f.write('\n')
                f.write('diana finish point')
                f.write(str(diana.controller.state[0]))
                f.write('\n')
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

    ### print to a file
    f.write('GAME STATS \n')
    f.write('sharon wins : ')
    f.write(str( sharon.success))
    f.write('diana wins : ')
    f.write(str(diana.success))
    f.write('ties ')
    f.write(str(number_of_ties))
    f.close()


# run_epoch_and_games(0.1, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.1", "training_01.txt", 500, 15000)
# run_epoch_and_games(0.2, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.2", "training_02.txt", 500, 15000)
run_epoch_and_games(0.6, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.6", "training_065_18.txt", 500, 20000)
run_epoch_and_games(0.8, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.8", "training_081_18.txt", 500, 20000)
run_epoch_and_games(0.7, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.7", "training_072_18.txt", 500, 20000)
run_epoch_and_games(0.9, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.9", "training_093_18.txt", 500, 20000)
run_epoch_and_games(0.5, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.5", "training_054_18.txt", 500, 20000)
# run_epoch_and_games(0.8, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.8", "training_08.txt", 500, 15000)
# run_epoch_and_games(0.9, "/Users/rachelhaighton/PycharmProjects/Value-Decomposition-Hallway/w 0.9", "training_09.txt", 500, 15000)