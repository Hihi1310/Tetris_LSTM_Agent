
# Functional package
import numpy as np
import tensorflow as tf
from agent import TetrisLSTMAgent
# ENV package
import gymnasium as gym

# Create and wrap the environment
env = gym.make("ALE/Tetris-v5",render_mode="human", frameskip = 4)
state_size = (env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])
action_size = env.action_space.n
frames = []
taken_actions = []
states = []

#change this base on where you save the downloaded model
agent = TetrisLSTMAgent(state_size, action_size)
model_path = "tetris_lstm_model.h5"
agent.model.load_weights(model_path)

# Test the trained model
env.reset()
state = env.step(env.action_space.sample())[0]
state = np.array(state, dtype=float)
state = np.reshape(state, [1, 1, *state_size])
for t in range(5000):

    action = np.argmax(agent.model.predict(state, verbose=0))
    taken_actions.append(action)

    next_state, reward, done, _, info = env.step(action)
    next_state = np.array(next_state)
    states.append(next_state)

    state = np.reshape(next_state, [1, 1, *state_size])

    env.render()
    if done:
        break

# env.close()
