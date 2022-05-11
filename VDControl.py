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
        self.territory_coordinates = [45,45]  # these will eventually be in the game class and passed into the actor
        self.r = 1 #radius of the finish line/territory
        self.a = 0 # acceleration
        self.v = 0 # velocity, m/s
        self.m = 1 # mass, kg
        self.b = 0.00001 # viscosity, newton-second per square metre
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
        if (abs(self.state[0]  - self.territory_coordinates[0]) <= self.r and abs(self.state[1] - self.territory_coordinates[1]) <= self.r):
            r = 100

        else:
            r = (self.distance_away_from_target_t - self.distance_away_from_target_t_plus_1)
        # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
        self.distance_away_from_target_t = self.distance_away_from_target_t_plus_1
        # heading_desired = np.arctan( (self.territory_coordinates[1] - self.state[1]) / (self.territory_coordinates[0] - self.state[0]))
        # heading_error = heading_desired - self.u_t
        # r = 6*np.exp(-(heading_error/0.5)**2)-3
        # print('reward', 'exp(-', heading_error, '/0.5)^2)-3 = ',r )
        self.update_reward_graph(r)
        return r

    def update_state(self):
        # self.state[0] = self.state[0] + self.v * np.cos(self.u_t)
        # self.state[1] = self.state[1] + self.v * np.sin(self.u_t)
        # self.update_path(self.state)
        a = self.a
        v = 100*self.u_t
        #
        # self.a = 100*self.u_t #(1 / self.m) * (100*self.u_t - self.b * v)
        # self.v = v + a * self.dt
        self.state[0] = self.state[0] + v * self.dt
        self.state[1] = self.state[1] + v * self.dt
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
            (self.state[0] - self.territory_coordinates[0]) ** 2 + (self.state[1] - self.territory_coordinates[1]) ** 2)
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

        pass
    def load(self):
        self.omega = np.loadtxt('actor_weights.csv', delimiter=',')
        self.zeta = np.loadtxt('critic_weights.csv', delimiter=',')