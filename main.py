import numpy as np
import itertools


# helper function
import dynamicProgramming


def get_reward(state, action):
    if state == (1, 1) and action == 1:
        return 2
    elif state == (1, 0) and action == 1:
        return 1
    elif state == (1, 1) and action == 0:
        return -2
    elif state == (1, 0) and action == 1:
        return -1
    elif state == (0, 1) and action == 1:
        return 2
    elif state == (0, 0) and action == 1:
        return 2
    elif state == (0, 1) and action == 0:
        return -2
    else:
        return -1


# system description
transition_matrix = np.matrix([[0.64, 0.16, 0.04, 0.16], [0.16, 0.64, 0.16, 0.04],
                               [0.04, 0.16, 0.64, 0.16], [0.16, 0.04, 0.16, 0.64]])

state_matrix = [(1, 1), (1, 0), (0, 0), (0, 1)]

action_space = [1, 0]

state_action = np.array(list(itertools.product(state_matrix, action_space)), dtype=object)

q_array = []

for row in list(state_action):
    reward_list = [row[0], row[1], get_reward(row[0], row[1])]
    q_array.append(reward_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model1 = dynamicProgramming.DynamicProgram(state_matrix, action_space, q_array, transition_matrix, 0.1, 0.8)
    model2 = dynamicProgramming.DynamicProgram(state_matrix, action_space, q_array, transition_matrix, 0.1, 0.5)
    model1.itervalue()
    model2.itervalue()
    print("model 1", model1.v_matrix, model2.v_matrix)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
