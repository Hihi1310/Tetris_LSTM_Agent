# Functional package
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Util package
import random
import os
import json
from tqdm import tqdm
from collections import deque
from typing import Tuple, Deque

#Vizualized package
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


# Define the LSTM-based RL model
class TetrisLSTMAgent():
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.train_history = {'loss':[]}

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            layers.Input(shape=(None, *self.state_size)),
            layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D(2, 2)),
            layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D(2, 2)),
            layers.TimeDistributed(tf.keras.layers.Flatten()),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(self.action_size),
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self,
                 state: np.ndarray,
                 action: int,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool) -> None:
        '''
        Save the experience to the memory
        '''
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray):
        '''
        The agent return an action based on the state. The action can be the
        a model prediction or random action base on epsilon (exploration rate)
        '''
        # Choose between random action or model action
        if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)
        else:
          state = np.reshape(state, [1, 1, *self.state_size])
          act_values = self.model.predict(state, verbose=0)
          return np.argmax(act_values[0])
        
    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def replay(self, batch_size: int):
        '''
        Train the model base using the saved memory on the environment
        '''
        minibatch = random.sample(self.memory, batch_size)
        state_batch = []
        Q_value_batch = []
        # have the model learning from mini batch by replaying it
        for state, action, reward, next_state, done in minibatch:
            # state = np.reshape(state, [1, 1, self.state_size])
            # next_state = np.reshape(next_state, [1, 1, self.state_size])
            state_batch.append(state)
            # set model target as the Q-value from the action
            target = reward
            if not done:
                next_reward = np.amax(
                               self.model.predict(np.expand_dims(next_state, axis=(0, 1)), verbose=0)[0]
                          )
                target = (reward + self.gamma * next_reward)
            # set the action value to target
            target_f = self.model.predict(np.expand_dims(state, axis=(0, 1)), verbose=0)[0]
            target_f[action] = target
            # Add Q_value to training batch
            Q_value_batch.append(target_f)

        # train model
        state_batch = np.array(state_batch)
        Q_value_batch = np.array(Q_value_batch)

        state_batch = np.expand_dims(state_batch, axis=1)
        Q_value_batch = np.expand_dims(Q_value_batch, axis=1)

        hist = self.model.fit(state_batch, Q_value_batch, epochs=1, verbose=1, batch_size=batch_size)
        self.train_history['loss'] += hist.history['loss']
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_model(self, plot_name: str='model.png', save_to: str=''):
      plot_model(self.model, to_file=os.path.join(save_to, plot_name), show_shapes=True, show_layer_names=True)
      model_img = plt.imread(plot_name)
      plt.imshow(model_img)
      plt.axis('off')
      plt.figure(figsize=(10, 10))
      plt.show()