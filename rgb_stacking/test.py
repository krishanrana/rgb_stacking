
from absl import app
from rgb_stacking import environment
from absl import flags
from dm_robotics.manipulation.props.rgb_objects import rgb_object
import numpy as np
import pdb
import cv2
from rgb_stacking.utils import policy_loading
from dm_control import viewer


class RandomAgent:
  """An agent that emits uniform random actions."""

  def __init__(self, action_spec):
    self._spec = action_spec
    self._shape = action_spec.shape
    self._range = action_spec.maximum - action_spec.minimum

  def step(self, unused_timestep):
    action = (np.random.rand(*self._shape) - 0.5) * self._range
    np.clip(action, self._spec.minimum, self._spec.maximum, action)
    return action.astype(self._spec.dtype)


def run_episode_and_render(env, policy) :

    while True:
        state = policy.initial_state()
        timestep = env.reset()
        while not timestep.last():
            #action = policy.step(timestep)
            (action, _), state = policy.step(timestep, state)

            timestep = env.step(action)
            obs = timestep.observation['basket_front_right/pixels']
            rew = timestep.reward
            done = timestep.last()

            success = (rew == 1.0)
            print(timestep.observation['gripper/joints/angle'])


            obs = cv2.resize(obs, (96,96), interpolation = cv2.INTER_AREA)
            cv2.namedWindow('rgb',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('rgb', 500,500)
            cv2.imshow('rgb', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)

            #rendered_images.append(env.physics.render(camera_id='main_camera'))
    return 


def main(argv):
    #env = environment.rgb_stacking(object_triplet="rgb_test_random")
    #env = environment.rgb_stacking(object_triplet='rgb_test_triplet4', observation_set=environment.ObservationSet.VISION_ONLY)
    env = environment.rgb_stacking(object_triplet='rgb_test_triplet4', observation_set=environment.ObservationSet.ALL, use_sparse_reward=True)
    
    #agent = RandomAgent(env.action_spec())

    policy_path = "../assets/saved_models/mpo_state_rgb_test_triplet4"
    agent = policy_loading.policy_from_path(policy_path)

    #agent = policy_loading.StatefulPolicyCallable(agent)
    #viewer.launch(env, policy=agent)

    run_episode_and_render(env, agent)

    env.close()
    return

if __name__ == "__main__":
    app.run(main)