from FACL import FACL
import numpy as np
from numpy import savetxt

# This class inherits FACL and implements the :
# reward function
# state update
# saves the path the agent is taking in a given epoch
# resets the game after an epoch


class VDControl(FACL):

    def __init__(self, state, max, min, num_mf):
        self.state = state.copy()
        self.path = state.copy()
        self.velocity_path = 0
        self.input = 0
        self.initial_position = state.copy()
        self.finish_line = 10  # these will eventually be in the game class and passed into the actor
        self.r = 1 #radius of the finish line/territory
        self.a = 0 # acceleration
        self.v = 0 # velocity, m/s
        self.m = 1 # mass, kg
        self.b = 0.1 # viscosity, newton-second per square metre
        self.dt = 0.01
        self.distance_away_from_target_t_plus_1 = 0 #this gets set later
        self.distance_away_from_target_t = self.distance_from_target()
        self.reward_track =[] # to keep track of the rewards
        FACL.__init__(self, max, min, num_mf) #explicit call to the base class constructor
        self.fuzzy_info_max = max
        self.fuzzy_info_min = min
        self.fuzzy_info_nmf = num_mf

    def get_reward(self):
        self.distance_away_from_target_t_plus_1 = self.distance_from_target()
        if (self.state[0]  >= self.finish_line ):
            r = 10

        else:
            r = 10*(self.distance_away_from_target_t - self.distance_away_from_target_t_plus_1)
        # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
        self.distance_away_from_target_t = self.distance_away_from_target_t_plus_1
        self.update_reward_graph(r)
        return r

    def update_state(self):
        # self.state[0] = self.state[0] + self.v * np.cos(self.u_t)
        # self.state[1] = self.state[1] + self.v * np.sin(self.u_t)
        # self.update_path(self.state)
        #
        max = 3
        min = -3
        if(self.u_t>max):
            self.u_t = max
        elif(self.u_t<min):
            self.u_t = min

        self.a = (1 / self.m) * (self.u_t - self.b * self.v)
        self.state[1] = self.state[1] + self.a * self.dt
        # self.state[1] = self.v
        # self.state[0] = self.state[0] + self.v * self.dt
        # self.state[1] = self.state[1] + self.v * self.dt
        for t in range(10):
            self.state[0] = self.state[0] + self.state[1] * self.dt
            self.state[1] = self.state[1] + self.a * self.dt

        self.v=self.state[1]
        self.update_path(self.state)
        self.update_v_path(self.v)
        self.update_input_array(self.u_t)
        pass

    def reset(self):
        # Edited for each controller
        self.state = self.initial_position.copy()
        self.path = []
        self.path = self.initial_position.copy()
        self.reward_track = []
        self.distance_away_from_target_t = self.distance_from_target()
        self.input = 0
        self.a = 0
        self.v = 0
        self.velocity_path = []
        self.velocity_path = 0
        pass

    def update_path(self, state):
        self.path = np.vstack([self.path, state])
        pass
    def update_v_path(self, state):
        self.velocity_path = np.vstack([self.velocity_path, state])
        pass
    def update_input_array(self, u):
        self.input = np.vstack([self.input, u])
        pass
    def update_reward_graph(self, r):
        self.reward_track.append(r)

    def distance_from_target(self):
        distance_away_from_target = np.sqrt(
            (self.state[0] - self.finish_line) ** 2 + (self.state[1] - self.finish_line) ** 2)
        return distance_away_from_target

    def save(self):
        # save the actor weight list
        savetxt('actor_weights.csv', self.omega, delimiter=',')
        # save the critic weight list
        savetxt('critic_weights.csv', self.zeta, delimiter=',')
        # save the fuzzy system information
        # savetxt('fuzzy_info.txt',self.fuzzy_info)
        np.savetxt("fuzzy_info.txt",self.fuzzy_info_max, fmt='%1.3f', newline="\n")
        with open("fuzzy_info.txt", "a") as f:
             np.savetxt(f, self.fuzzy_info_min, fmt='%1.3f', newline="\n")
             np.savetxt(f, self.fuzzy_info_nmf,fmt='%1.3f', newline="\n")
        savetxt('u_t.csv', self.input, delimiter=',')
        pass
    def load(self):
        self.omega = np.loadtxt('actor_weights.csv', delimiter=',')
        self.zeta = np.loadtxt('critic_weights.csv', delimiter=',')