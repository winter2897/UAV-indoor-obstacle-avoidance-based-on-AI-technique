import numpy as np
import gym


import gym_airsim


import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input, Activation
from keras.optimizers import Adam
import keras.backend as K
import keras
from keras import regularizers 
from keras.layers.merge import add

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from callbacks import FileLogger

from keras.callbacks import History

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='airsim-v1')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
INPUT_SHAPE = (30, 100)
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 144, 256, 3
WINDOW_LENGTH = 1
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
# input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


# model = Sequential()
# model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape, data_format = "channels_first"))
# model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
# model.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())

def resnet8(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
  INPUT_SHAPE = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
  x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same', data_format='channels_last')(INPUT_SHAPE)
  x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)
      # First residual block: RES BLOCK 1
  x2 = keras.layers.normalization.BatchNormalization()(x1)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
       kernel_initializer="he_normal",
       kernel_regularizer=regularizers.l2(1e-4))(x2)

  x2 = keras.layers.normalization.BatchNormalization()(x2)
  x2 = Activation('relu')(x2)
  x2 = Conv2D(32, (3, 3), padding='same',
       kernel_initializer="he_normal",
       kernel_regularizer=regularizers.l2(1e-4))(x2)

  x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
  x3 = add([x1, x2])

      # Second residual block: RES BLOCK 2
  x4 = keras.layers.normalization.BatchNormalization()(x3)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizers.l2(1e-4))(x4)

  x4 = keras.layers.normalization.BatchNormalization()(x4)
  x4 = Activation('relu')(x4)
  x4 = Conv2D(64, (3, 3), padding='same',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4))(x4)

  x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
  x5 = add([x3, x4])

      # Third residual block: RES BLOCK 3
  x6 = keras.layers.normalization.BatchNormalization()(x5)
  x6 = Activation('relu')(x6)
  x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4))(x6)

  x6 = keras.layers.normalization.BatchNormalization()(x6)
  x6 = Activation('relu')(x6)
  x6 = Conv2D(128, (3, 3), padding='same',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(1e-4))(x6)

  x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
  x7 = add([x5, x6])

      #Flatten 
  x8 = Flatten()(x7)
  x8 = Activation('relu')(x8)
  x8 = Dropout(0.5)(x8)
  
  x9 = Dense(512, activation = "relu")(x8)
  x9 = keras.layers.normalization.BatchNormalization()(x9)
  x9 = Dropout(0.5)(x9)
  result =  Dense(3, activation = 'softmax')(x9)

  model = Model(inputs=[INPUT_SHAPE], outputs=[result])
  print(model.summary())
  return model

model = resnet8(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

if args.mode == "train":
    train = True
elif args.mode == "test":
    train = False
else:
    exit(-1)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)                        #reduce memmory


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)

class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        return np.squeeze(batch, axis=1)

dqn = DQNAgent(model=model, processor=CustomProcessor(), nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=True, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])

<<<<<<< HEAD

if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'RLModel_{}.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=100)]
    
    dqn.fit(env, callbacks=callbacks, nb_steps=10000, visualize=False, verbose=2, log_interval=100)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights('RLWeight_{}.h5f'.format(args.env_name), overwrite=True)


else:

    dqn.load_weights('RLWeight_{}.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=10, visualize=False)
=======
if __name__ == "__main__":
    if train:
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        
        
        log_filename = 'RLModel_{}.json'.format(args.env_name)
        callbacks = [FileLogger(log_filename, interval=100)]
        
        dqn.fit(env, callbacks=callbacks, nb_steps=10000, visualize=False, verbose=2, log_interval=100)
        
        
        # After training is done, we save the final weights.
        dqn.save_weights('RLWeight_{}.h5f'.format(args.env_name), overwrite=True)


    else:

        dqn.load_weights('RLWeight_{}.h5f'.format(args.env_name))
        dqn.test(env, nb_episodes=10, visualize=False)
        
>>>>>>> 47a1e0b0f7e2f8b41d493fee255d23e4c30fcad3
