# Functional package
import numpy as np
import tensorflow as tf
from agent import TetrisLSTMAgent

# ENV package
import gymnasium as gym

# Util package
from tqdm import tqdm
import json


# Initialize the agent
tf.keras.backend.clear_session()
env = gym.make("ALE/Tetris-v5", frameskip=4)

state_size = (env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])
action_size = env.action_space.n

agent = TetrisLSTMAgent(state_size, action_size)

# Training parameters
EPISODES = 50
BATCH_SIZE = 32
EPISODE_MAX_STEPS = 5000

# Main training loop
for e in tqdm(range(EPISODES)):
    env.reset()
    start_state, reward, done, _ , info= env.step(env.action_space.sample())
    state= start_state
    print(f'\n\nthis is episode {e}')
    step_counter = 0
    bonus_reward = 1
    for time in tqdm(range(EPISODE_MAX_STEPS)):  # Limit each episode number of steps
        #get action and next state with model
        action = agent.act(state)
        next_state, reward, done, _ , info= env.step(action)

        agent.remember(state, action, reward+bonus_reward, next_state, done)

        state = next_state
        if done:
            print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon}")
            break
    
        # Train the model using agent memory
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
            
            bonus_reward += time/100.

            step_counter=0
    
        step_counter += 1
        
    # Save the trained model every episode
    agent.model.save('tetris_lstm_model_checkpoint.h5')
    
    # Log the result after each episode
    with open('train_log.json', 'w') as f:
            json.dump(agent.train_history, f)