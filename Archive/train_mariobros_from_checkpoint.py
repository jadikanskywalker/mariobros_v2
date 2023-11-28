#adapted from https://www.kaggle.com/code/danieldreher/vanilla-dqn-cartpole-tensorflow-2-3/notebook 
import gymnasium as gym
import numpy as np
import tensorflow as tf
import random as rand
import cv2
import collections
from collections import deque
from tensorflow import keras
import datetime

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

    
def make_env(env, obs_type="grayscale"):
    env = ProcessFrame84(env)
    return ScaledFloatFrame(env)

#the replay buffer contains episode transitions in the
#  order the episode is generated
#The Python collections deque has a pointer to the next
#  and previous element for all elements
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.replay_memory = deque(maxlen=buffer_size)    

    #add one transition to the replay buffer
    def add(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    #take a random sample from the replay buffer for training
    def sample(self, batch_size):
        if batch_size <= len(self.replay_memory):
            return rand.sample(self.replay_memory, batch_size)
        else:
            assert False

    #Python magic method to enable len to be used on a replay buffer object
    def __len__(self):
        return len(self.replay_memory)
    
    #Class to implement an epsilon decay schedule
#epsilon starts high and then reduces by a decay factor
#The decay factor can be changed according to how many training iterations
#  are completed
class EpsilonSchedule():
    def __init__(self, final_epsilon=0.1, pre_train_steps=10, final_exploration_step=100):
        self.pre_train_steps = pre_train_steps
        self.final_exploration_step = final_exploration_step
        self.final_epsilon = final_epsilon
        self.decay_factor = self.pre_train_steps/self.final_exploration_step
        self.epsilon = 1
    
    def value(self, t):
        if t > self.pre_train_steps:
            self.decay_factor = (t - self.pre_train_steps)/self.final_exploration_step
            self.epsilon = 1-self.decay_factor
            self.epsilon = max(self.final_epsilon, self.epsilon)
            return self.epsilon
        else:
            return 1
        
#define the neural network model using keras
class DQN(keras.Model):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        # self.input_layer = keras.layers.InputLayer(input_shape=input_shape)
        # self.hidden_layers = []
        
                        # tf.keras.layers.InputLayer(input_shape=input_shape),
        print(input_shape)
        self.net = tf.keras.Sequential(
            [
                # tf.keras.layers.Input(shape=input_shape), #Input shape will be a 210 x 160 rgb image (change 3 to 1 and double check the dimensions of the input image if grayscale is used
                keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(32, 10, strides = 2, activation = "gelu", padding = "same"), #32 10 x 10 filters with a stride length of 2 and a padding of 0s around the edges of the image
                tf.keras.layers.Conv2D(32, 10, activation = "gelu"), #32 10 x 10 filters with a stride length of 1
                tf.keras.layers.Conv2D(64, 10, activation = "gelu"), #64 10 x 10 filters with a stride length of 1
                tf.keras.layers.Conv2D(64, 5, activation = "gelu"), #64 5 x 5 filters with a stride length of 1
                tf.keras.layers.MaxPooling2D(5), #Max Pooling using a 5 x 5 filter
                #Max pooling takes the highest value in the filter an makes it the value of a smaller "image," unlike the convolutional layers, the filter does not overlap itself
                tf.keras.layers.Flatten(), #Converts the data up to this point into a 1D array
                tf.keras.layers.Dense(256, activation = "gelu"), #Single regression layer for a small boost to feature extraction
                tf.keras.layers.Dense(num_actions, activation = "softmax") #Determines the class
                #Softmax activation will bind the results to a range of [0, 1], with the sum of all nodes equaling 1
                #This allows the final layer to present a probability of each action
            ]
        )

        # self.hidden_layers.append(keras.layers.Dense(64, activation='relu'))
        # self.hidden_layers.append(keras.layers.Dense(32, activation='relu'))
        # self.output_layer = keras.layers.Dense(units=num_actions, activation='linear')

    #forward pass of the model
    @tf.function
    def call(self, inputs):
        # z = self.input_layer(inputs)
        # for l in self.hidden_layers:
        #     z = l(z)
        # q_vals = self.output_layer(z)
        q_vals = self.net(inputs)
        return q_vals   
        
class Agent:
    def __init__(self, env, gamma=0.9, batch_size=64, lr=0.001,
                 max_episodes = 500, max_steps_per_episode=2000,
                 steps_until_sync=20, choose_action_frequency=1,
                 pre_train_steps = 1, final_exploration_step = 100):
        
        self.env = env
        self.input_shape = list(env.observation_space.shape)
        self.num_actions = env.action_space.n
        # dqn is used to predict Q-values to decide which action to take
        self.dqn = DQN(self.input_shape, self.num_actions)
        #build is used for subclassed models and takes the input shape as an argument
        #build builds the model
        self.dqn.build(tf.TensorShape([None, *self.input_shape]))
        
        # second DQN to predit the future reward of Q(s',a)
        # dqn_target is used to predict the future reward
        self.dqn_target = DQN(self.input_shape, self.num_actions)
        self.dqn_target.build(tf.TensorShape([None, *self.input_shape]))


        self.batch_size = batch_size
        #stochastic gradient method, lr is learning rate
        self.optimizer = tf.optimizers.legacy.Adam(lr)
        #discount factor
        self.gamma = gamma
        #to fill up the replay buffer
        self.pre_train_steps = pre_train_steps
        self.final_exploration_step = final_exploration_step
        self.replay_buffer = ReplayBuffer(max_episodes*max_steps_per_episode)
        self.epsilon_schedule = EpsilonSchedule(final_epsilon=0.2, 
                pre_train_steps=self.pre_train_steps,
                final_exploration_step=self.final_exploration_step)
        #steps until the target dqn is updated with the current dqn
        self.steps_until_sync = steps_until_sync
        #choose a new action every action frequency steps
        self.choose_action_frequency = choose_action_frequency
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        #loss function of mean squared error
        self.loss_function = tf.keras.losses.MSE
        self.episode_reward_history = []

    #predict the q values
    def predict_q(self, inputs):
        return self.dqn(inputs)

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            # explore
            return np.random.choice(self.num_actions)
        else:
            # exploit
            return np.argmax(self.predict_q(np.expand_dims(states, axis=0))[0])

    #copy dqn into dqn_target
    def update_target_network(self):
        self.dqn_target.set_weights(self.dqn.get_weights())

    #take a training step
    def train_step(self):
        #take a random sample from the replay buffer
        mini_batch = self.replay_buffer.sample(self.batch_size)
        #unzip the random sample into separate vectors
        observations_batch, action_batch, reward_batch, next_observations_batch, done_batch = map(np.array,zip(*mini_batch))
        #record operations for automatic differentiation
        with tf.GradientTape() as tape:
            #watch the trainable variables
            dqn_variables = self.dqn.trainable_variables
            tape.watch(dqn_variables)
            #compute the rewards of the next state
            future_rewards = self.dqn_target(tf.convert_to_tensor(next_observations_batch, dtype=tf.float32))
            next_action = tf.argmax(future_rewards, axis=1)
            #find the sum of elements across the columns of the tensor
            #one_hot is used to mask out any q values that are unneeded
            target_q = tf.reduce_sum(tf.one_hot(next_action, self.num_actions) * future_rewards, axis=1)
            #update the future rewards eliminating any states that were terminal states
            target_q = (1 - done_batch) * self.gamma * target_q + reward_batch
            #do the same for the current state
            predicted_q = self.dqn(tf.convert_to_tensor(observations_batch, dtype=tf.float32))
            predicted_q = tf.reduce_sum(tf.one_hot(action_batch, self.num_actions) * predicted_q, axis=1)
            #find the loss between the tartget and the predicted
            loss = self.loss_function(target_q, predicted_q)   
        # Backpropagate the loss
        gradients = tape.gradient(loss, dqn_variables)
        self.optimizer.apply_gradients(zip(gradients, dqn_variables))
        #return the loss
        return loss

    def train(self):
        episode = 0
        total_step = 0
        episode_step = 0
        state, info = self.env.reset()
        loss = 0
        last_hundred_rewards = deque(maxlen=100)

        while episode < self.max_episodes:
            current_state, info = self.env.reset()
            done = False
            action = 0
            episode_reward = 0
            episode_step = 0
            epsilon = self.epsilon_schedule.value(total_step)

            while not done:
                #control the number of times a new action is chosen
                if total_step % self.choose_action_frequency == 0:
                    if len(self.replay_buffer) > self.batch_size:
                        action = self.get_action(current_state, epsilon)
                    else:
                        action = self.get_action(current_state, 1.0)

                num_lives = info["lives"]

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                reward /= 800
                if info["lives"] < num_lives: # penalize agent when life lost
                    reward -= 0.33
                
                self.replay_buffer.add(current_state, action, reward, next_state, done)
                episode_reward += reward
                
                #train the dqn if enough data samples are available
                if total_step > self.pre_train_steps and len(self.replay_buffer) > self.batch_size:
                    loss = self.train_step()
                #control how often the target dqn is updated in order to foster stability in the target q value
                if total_step % self.steps_until_sync == 0:
                    self.update_target_network()
                                    
                #end of step
                total_step += 1
                episode_step += 1
                current_state = next_state
                
            # end of episode
            self.episode_reward_history.append(episode_reward)
            last_hundred_rewards.append(episode_reward)
            mean_episode_reward = np.mean(last_hundred_rewards)
            #show the average reward
            if episode % 20 == 0:
                print('\n' + f'Episode {episode} (Step {total_step}) - Moving Avg Reward: {mean_episode_reward:.3f} Loss: {loss:.5f} Epsilon: {epsilon:.3f}')
                self.dqn_target.save_weights("./checkpoints/ep_" + str(episode))
            else:
                print("*", end="")
            #stop training if the mean of the last 100 rewards is nearing 200
            if mean_episode_reward >= 195:
                print(f'Task solved after {episode} episodes! (Moving Avg Reward: {mean_episode_reward:.3f})')
                return                
            episode += 1
            
    def load_weights(self, pathname):
        self.dqn.load_weights(pathname)

GAME = "ALE/MarioBros-v5"
    

#Train the agent
env = make_env(gym.make(GAME, mode=4))
print("Action space: {}".format(env.action_space))
print("Action space size: {}".format(env.action_space.n))
observation, info = env.reset()
print("Observation space shape: {}".format(observation.shape))
print("Environment spec: ", env.spec)

agent = Agent(env, gamma=0.999, batch_size=64, lr=0.0007, max_episodes=1000,
              max_steps_per_episode=2000,
              steps_until_sync=20, choose_action_frequency=1,
              pre_train_steps = 1, final_exploration_step = 700_000)
agent.load_weights("./checkpoints_v3/ep_20")
agent.train()

env.close()

#use the DQN
env = make_env(gym.make(GAME, render_mode="rgb_array"))
observation, info = env.reset()

# Create a VideoWriter object.
video_writer = cv2.VideoWriter('test_output_' + str(datetime.datetime.now().date()) + '.avi', cv2.VideoWriter_fourcc(*'AVI'), 20.0, (160, 210), isColor=True)

#show the steps the agent takes using the optimal policy table
for i in range(10):
    observation, info = env.reset()
    terminated = truncated = False
    rewards = 0
    while not terminated and not truncated:
        #find max policy
        Q_values = agent.predict_q(np.expand_dims(observation, axis=0))
        action = np.argmax(Q_values[0])
        observation, reward, terminated, truncated, info = env.step(action)
        video_writer.write(np.uint8(np.reshape(env.render(), (210, 160, 3))))
        rewards += reward
    print('Total reward is: '+str(rewards))
env.close()

# Close the VideoWriter object.
video_writer.release()