import gym
import highway_env
import time
import numpy as np
env = gym.make("acc-v0", speed_limit=-1)
for i in range(10):
    env.reset()
    done = False
    while not done:
        action =[np.random.uniform(-1,1.)]
        obs, reward, done, info = env.step(action)
        #print("Obs:")
        #print(obs)
        env.render()
        time.sleep(0.1)