import tensorflow as tf
import itertools
import numpy as np
from tensorflow.keras import layers, models

class TrainModel(object):
    def __init__(self):
        self.model = model

    def predict(state_elements_array):
        return self.model.predict(state_elements_array)

class DQN(object):
    def __init__(self,
                valid_actions):
        self.train_model = TrainModel()
        self.valid_actions = valid_actions

    def deep_q_learning(self,
                        env,
                        epsilon_start,
                        epsilon_end,
                        epsilon_decy_steps,
                        q_estimator,
                        update_target_estimator_every,
                        num_episode
                        ):
        
        total_t = 0
        epsilons = np.linespace(epsilon_start, epsilon_end, epsilon_decy_steps)

        for episode_i in range(num_episode):
            state = env.reset()

            for t in itertools.count():
                epsilon = epsilons[min(total_t, epsilon_decy_steps - 1)]

                #if total_t % update_target_estimator_every == 0:
                    # copy all parameters as backup

                action_probs = self.make_epsilon_greedy_policy(len(self.valid_actions), state, epsilon)

                # np.random.choice means that we get a random action from 0~len(action_probs) with probs action_probs
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                next_state, reward, done = env.step(self.valid_actions[action])
                # state[:,:,1:] means the 2nd~last elements in 3rd dimension

                total_t += 1


    def make_epsilon_greedy_policy(action_num, observation, epsilon):
        A = np.ones(action_num, dtype = float) * epsilon / action_num
        """
        predict actually gets q_values table which records all correspondence between actions and states
        observation means current "state elements array", so expand_dims actually makes current state as a "state array"
        syntax [0] at end gets q_values of latest state correspond to various actions

        """
        q_values = self.train_model.predict(observation)

        """
        We take the best action with highest score
        """
        best_action_idx = np.argmax(q_values)

        """
        epsilon represents some kind of exploring probility, for each transition(transit from one state to another)
        the probility to choose a random action instead of the best action is equal to (1 - epsilon)
        purpose of this operation is to explore as many unknown cases as possible, as number of episode increases,
        epsilon will decrease as well, which makes (1 - epsilon) increasing, which means we will increase the probility
        to use "best action" instead of exploring unknown cases at late stage.
        """
        A[best_action_idx] += (1.0 - epsilon)
        return A

