import gym
import highway_env
from stable_baselines.gail import generate_expert_traj
import argparse

# ==================================
#        Main script
# ==================================



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coop', type=float, default=0.0)
    parser.add_argument('--render', default=False, action='store_true')

    args = parser.parse_args()
    env = gym.make("mergecooperative-v1", observation="LIST", cooperative_prob=args.coop, force_render=args.render)
    obs=None
    def dummy_expert(_obs):
        """
        Random agent. It samples actions randomly
        from the action space of the environment.

        :param _obs: (np.ndarray) Current observation
        :return: (np.ndarray) action taken by the expert
        """
        return env.action_space.sample()
    exp_name = "expert_data_coop_{}".format(args.coop)
    generate_expert_traj(dummy_expert, exp_name, env, n_episodes=10)

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
