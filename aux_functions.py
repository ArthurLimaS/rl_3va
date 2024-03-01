import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def intersection_v1(STACK_SIZE = 4, SCALING = 1):

    env = gym.make("intersection-v1", render_mode='rgb_array')

    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": STACK_SIZE,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": SCALING,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": False,
            "lateral": True
        },
    }

    env.unwrapped.configure(config)

    return Monitor(env)

def render_model(model, env):
    obs, info = env.reset()

    for _ in range(3):
        while True:
            env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    env.close()