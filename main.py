import numpy as np
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.rewards import reward_nets


SEED = 1

def run_gail(env, expert, epochs=5, trajectories=120, eval_episodes=100, train_steps=500_000):
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=trajectories),
        rng=np.random.default_rng(SEED),
    )
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=epochs,
        seed=SEED,
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    # evaluate the learner before training
    env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, eval_episodes, return_episode_rewards=True,
    )

    # train the learner and evaluate again
    gail_trainer.train(train_steps)  # Train for 800_000 steps to match expert.
    env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, eval_episodes, return_episode_rewards=True,
    )

    # print("mean reward before training:", np.mean(learner_rewards_before_training))
    # print("mean reward after training:", np.mean(learner_rewards_after_training))
    print(
        "Rewards before training:",
        np.mean(learner_rewards_before_training),
        "+/-",
        np.std(learner_rewards_before_training),
    )
    print(
        "Rewards after training:",
        np.mean(learner_rewards_after_training),
        "+/-",
        np.std(learner_rewards_after_training),
    )
    return np.mean(learner_rewards_before_training), np.mean(learner_rewards_after_training)

def run_bc(env, expert, epochs=1, trajectories=10, eval_episodes=100):
    reward, _ = evaluate_policy(expert, env, n_eval_episodes=eval_episodes)
    print("Expert reward:", reward)

    rng = np.random.default_rng(SEED)
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=trajectories),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    print(
        f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    """
    )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, eval_episodes)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=epochs)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, eval_episodes)
    print(f"Reward after training: {reward_after_training}")
    return reward_before_training, reward_after_training


if __name__ == "__main__":
    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=np.random.default_rng(),
        n_envs=8,
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env)
        ],  # needed for computing rollouts later
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals/CartPole-v0",
        venv=env,
    )

    trajectories = 50
    epochs = 1
    eval_episodes = 10
    print("------------Starting BC------------")
    bc_reward_before_training, bc_reward_after_training = run_bc(env, expert, epochs, trajectories, eval_episodes)
    print("------------Starting GAIL------------")
    gail_reward_before_training, gail_reward_after_training = run_gail(env, expert, epochs, trajectories, eval_episodes, train_steps=500_000)
    print("------------Results------------")
    print("BC reward before training: ", bc_reward_before_training)
    print("BC reward after training: ", bc_reward_after_training)
    print("GAIL reward before training: ", gail_reward_before_training)
    print("GAIL reward after training: ", gail_reward_after_training)

    print("done!")





