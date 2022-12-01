from gym_airsim.AirSimEnv import AirSimEnv

from gym.envs.registration import register

register(
    id='airsim-v1',
    entry_point='gym_airsim:AirSimEnv',
)