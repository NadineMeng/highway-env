import gym
import highway_env
from stable_baselines.gail import generate_expert_traj

# ==================================
#        Main script
# ==================================



if __name__ == "__main__":

    env = gym.make("mergecooperative-v0", observation="LIST")
    obs=None
    def dummy_expert(_obs):
        """
        Random agent. It samples actions randomly
        from the action space of the environment.

        :param _obs: (np.ndarray) Current observation
        :return: (np.ndarray) action taken by the expert
        """
        return env.action_space.sample()

    generate_expert_traj(dummy_expert, 'dummy_expert_merge', env, n_episodes=10)

    # for _ in range(2000):
    #     action = env.action_type.actions_indexes["IDLE"]
    #
    #     obs, reward, done, info = env.step(action)
    #     print(obs)
    #     env.render()
    #     #plt.imshow(env.render(mode="rgb_array"))
    #     #plt.show()
    #     if done:
    #         env.reset()
