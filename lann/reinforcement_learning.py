import ale_py
import gymnasium as gym

from collections import namedtuple
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

gym.register_envs(ale_py)

"""Transition represents a transition from one state to another. It is implemented as a 
name tuple with five fields.

Args:
    state: The current state
    action: The chosen action
    next_state: The resulting state
    reward: The reward achieved from the transition
    done: Whether or not the transition resulted in either a termination (or truncation if 
        RLTrainer is initialized with reward_when_truncted)
"""
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done'])

def make_env(id:str, resolution=(84, 84), color=False, num_frames_stacked=4):
    environment = gym.make(id)
    environment = ResizeObservation(environment, resolution)
    if not color:
        environment = GrayscaleObservation(environment)
    environment = FrameStackObservation(environment, num_frames_stacked)
    print(f'Create environment for {id}')
    return environment


