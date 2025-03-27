import gymnasium as gym
import utils
from agent import SACAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
import wandb

warnings.simplefilter("ignore")

environments = [
    "BipedalWalker-v3",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "LunarLanderContinuous-v2",
    "HumanoidStandup-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v3",
    "Walker2d-v4",
]


def make_env(env_name):
    """Returns a function that creates an environment instance."""
    return lambda: gym.make(env_name)


def run_sac(env_name, n_games=10000, norm_flow=False, wandb_key=None, num_envs=8):
    env = gym.vector.AsyncVectorEnv([make_env(env_name) for _ in range(num_envs)])

    agent = SACAgent(
        env_name,
        env.single_observation_space.shape,
        env.single_action_space,
        tau=5e-3,
        reward_scale=10,
        batch_size=256,
        norm_flow=norm_flow,
    )

    if wandb_key:
        with open(wandb_key, "r") as f:
            wandb_key = f.read().strip()
        wandb.login(key=wandb_key)
        run_name = f"{env_name}-{'NF' if norm_flow else 'SAC'}"
        wandb.init(project="NF-SAC", name=run_name, config=locals())

    best_avg_score = -float("inf")
    scores = []
    metrics = []
    episode_scores = np.zeros(num_envs)
    states, _ = env.reset()

    while len(scores) < n_games:

        actions = np.array([agent.choose_action(state) for state in states])
        next_states, rewards, term, trunc, _ = env.step(actions)

        for j in range(num_envs):
            episode_scores[j] += rewards[j]
            agent.store_transition(
                states[j],
                actions[j],
                rewards[j],
                next_states[j],
                term[j] | trunc[j],
            )
            if term[j] or trunc[j]:  # If an episode ends
                scores.append(episode_scores[j])  # Save final score
                episode_scores[j] = 0  # Reset score for new episode

                avg_score = np.mean(scores[:-100] if len(scores) > 100 else scores)
                best_score = max(scores) if scores else 0

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    agent.save_checkpoints()

                results = {
                    "score": scores[-1],
                    "average_score": avg_score,
                    "best_score": best_score,
                }

                metrics.append(results)

                if wandb_key:
                    wandb.log(results)

                print(
                    f"[{env_name} Episode {len(scores):04}/{n_games}]  Score = {scores[-1]}  Average Score = {avg_score:7.4f}",
                    end="\r",
                )

            if len(scores) >= n_games:
                break

            agent.learn()

        states = next_states

    return history, metrics, best_avg_score, agent


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()

    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        env = gym.make(env_name, render_mode="rgb_array")

        frames = []
        total_reward = 0

        state, _ = env.reset()
        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            state = next_state
            total_reward += reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    utils.save_animation(
        best_frames, f"environments/{env_name}-norm={str(agent.norm_flow)}.gif"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "-n",
        "--n_games",
        default=1000,
        type=int,
        help="Number of episodes (games) to run during training",
    )
    parser.add_argument(
        "--norm_flow", action="store_true", help="Use normalizing flow-based policy"
    )
    parser.add_argument(
        "--wandb_key", type=str, default=None, help="Path to WandB API key text file"
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        history, metrics, best_score, trained_agent = run_sac(
            args.env, args.n_games, args.norm_flow, args.wandb_key
        )
        utils.plot_running_avg(history, args.env)
        df = pd.DataFrame(metrics)
        df.to_csv(f"metrics/{args.env}_metrics.csv", index=False)
        save_best_version(args.env, trained_agent)
    else:
        for env_name in environments:
            history, metrics, best_score, trained_agent = run_sac(
                env_name, args.n_games, args.norm_flow, args.wandb_key
            )
            utils.plot_running_avg(history, env_name)
            df = pd.DataFrame(metrics)
            df.to_csv(f"metrics/{env_name}_metrics.csv", index=False)
            save_best_version(env_name, trained_agent)
