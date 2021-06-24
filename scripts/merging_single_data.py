import gym
import highway_env


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":

    env = gym.make("mergecooperative-v0", observation="LIST")
    obs=None

    for _ in range(2000):
        action = env.action_type.actions_indexes["IDLE"]

        obs, reward, done, info = env.step(action)
        print(obs)
        env.render()
        #plt.imshow(env.render(mode="rgb_array"))
        #plt.show()
        if done:
            env.reset()
