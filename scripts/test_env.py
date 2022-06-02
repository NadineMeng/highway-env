import gym
import highway_env
import time
import numpy as np
env = gym.make("acc-v0", speed_limit=-1)
for i in range(1):
    env.reset()
    done = False
    while not done:
        #action =[np.random.uniform(-1,1.)]
        print('secode:1',env.vehicle.speed)
        action =[1]
        #print(action)
        obs, reward, done, info = env.step(action)
        print('secode:2',env.vehicle.speed)
        #print("Obs:")
        #print(obs)
        env.render()
        #time.sleep(0.01)