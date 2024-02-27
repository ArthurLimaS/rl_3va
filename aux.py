import gymnasium as gym

def intersection_v1(STACK_SIZE = 4):

    env = gym.make("intersection-v1", render_mode='rgb_array')

    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": STACK_SIZE,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1,
        }
    }

    env.unwrapped.configure(config)

    return env