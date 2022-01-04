import os

import numpy as np
from tensorflow import keras as tf_keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from collections import deque

# Modified Tensorboard Class
class ModifiedTensorBoard(TensorBoard):

    # Consturctor to set up initial step and writer
    def __init__(self, writer_start_step, model_name, **kwargs):
        super().__init__(**kwargs)
        self.step = writer_start_step
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.model_name = model_name
        self._log_write_dir = os.path.join(self.log_dir, self.model_name)

    # Override set model to prevent default log writer creation
    def set_model(self, model):
        pass

    # Override func to save logs with our predefined step numbers instead
    # of saving logs for every .fit() call at the 0th step
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Override to do nothing as we train for only one batch.
    def on_batch_end(self, batch, logs=None):
        pass

    # Overriden to not close the writer
    def on_train_end(self, logs=None):
        pass

    # Custom method for saving metrics
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs_dict, step_index):
        with self.writer.as_default():
            for name, value in logs_dict.items():
                tf.summary.scalar(name, value, step=step_index)
                self.step += 1
                self.writer.flush()


class DQNAgent:
    def __init__(self,
                 model_name,
                 replay_memory_size,
                 log_dir,
                 min_replay_memory_size,
                 input_shape=env.OBSERVATION_SPACE_VALUES,
                 writer_start_step=1):
        # main model # Gets trained every ste[
        self.input_shape = input_shape
        self.model = self.create_model()
        self.log_dir = F'logs/{model_name}'
        self.min_replay_size = min_replay_memory_size

        # target model. This is what we predict against every step
        self.target_model = self.create_model(input_shape)
        self.target_model.set_weights(self.model.get_weights())

        # Contains the last n steps for training
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_tens_board = ModifiedTensorBoard(writer_start_step, model_name, log_dir=log_dir)

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(MaxPool2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=0.0001, metrics=['accuracy']))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.asarray(state).reshape(-1, *state.shape)/255)[0]

    # Trains main network every step during the episode
    def train(self, terminal_state, step):
        # Start training only after certain number of samples are saved
        if len(self.replay_memory) < self.min_replay_size:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.asarray([trainsition[0] for transition in minibatch])/255
        current_q_list = self.model.predict(current_states)
        