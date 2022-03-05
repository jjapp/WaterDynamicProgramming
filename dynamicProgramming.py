import numpy as np


class DynamicProgram:
    """A class to solve dynamic programming problems that have a complete model"""
    def __init__(self, state_array, action_array, reward_space, transition_matrix, convergence_factor, discount_rate):
        self.state_space = state_array
        self.action_space = action_array
        self.r_space = reward_space
        self.transition = transition_matrix
        self.convergence_factor = convergence_factor
        self.v_matrix = np.zeros((len(self.state_space)))
        self.q_matrix = np.zeros((len(self.state_space)*len(self.action_space)))
        self.gamma = discount_rate

    def get_actions(self, system_state):
        """Cycles through the reward space and returns a list with all rewards for actions in that state"""
        reward_list = []
        for row in self.r_space:
            if row[0] == system_state:
                reward_list.append(row[2])
        return reward_list

    def itervalue(self):
        delta = 1000
        while delta > self.convergence_factor:
            temp_v = np.copy(self.v_matrix)
            for i in range(len(self.state_space)):
                reward_list = self.get_actions(self.state_space[i])
                v_prime_list = []
                for j in reward_list:
                    for k in range(len(self.v_matrix)):
                        value = self.transition[i, k] * (j + self.gamma * self.v_matrix[i])
                        v_prime_list.append(value)
                v_prime_max = max(v_prime_list)
                temp_v[i] = v_prime_max
            delta = np.subtract(temp_v, self.v_matrix)
            delta = np.absolute(delta)
            delta = np.amax(delta)
            self.v_matrix = np.copy(temp_v)











