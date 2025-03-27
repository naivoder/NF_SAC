import gymnasium as gym
import utils
from agent import SACAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
import wandb
from tqdm import tqdm

warnings.simplefilter("ignore")

environments = [
    "Pendulum-v1",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "MountainCarContinuous-v0",
    "Reacher-v5",
    "Swimmer-v3",
    "Pusher-v5",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "Hopper-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Ant-v4",
    "Humanoid-v4",
    "HumanoidStandup-v4",
]


def make_env(env_name):
    """Returns a function that creates an environment instance."""
    return lambda: gym.make(env_name)


def run_sac(args):
    env = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])

    agent = SACAgent(
        args.env,
        env.single_observation_space.shape,
        env.single_action_space,
        tau=5e-3,
        reward_scale=10,
        batch_size=256,
        norm_flow=args.norm_flow,
        num_flows=args.num_flows,
    )

    if args.norm_flow:
        save_str = f"NF-{args.num_flows}"
    else:
        save_str = "SAC"

    if args.wandb_key:
        with open(args.wandb_key, "r") as f:
            wandb_key = f.read().strip()
        wandb.login(key=wandb_key)
        run_name = f"{args.env}-{save_str}"
        wandb.init(project="NF-SAC", name=run_name, config=locals())

    best_avg_score = -float("inf")
    scores = []
    episode_scores = np.zeros(args.num_envs)
    states, _ = env.reset()

    while len(scores) < args.n_games:

        actions = np.array([agent.choose_action(state) for state in states])
        next_states, rewards, term, trunc, _ = env.step(actions)

        for j in range(args.num_envs):
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

                avg_score = np.mean(scores[-100:] if len(scores) > 100 else scores)
                best_score = max(scores) if scores else 0

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    agent.save_checkpoints()

                if args.wandb_key:
                    wandb.log(
                        {
                            "score": scores[-1],
                            "average_score": avg_score,
                            "best_score": best_score,
                        }
                    )

                print(
                    f"[{args.env} Episode {len(scores):04}/{args.n_games}]  Score = {scores[-1]:.2f}  Average Score = {avg_score:7.2f}",
                    end="\r",
                )

            if len(scores) >= args.n_games:
                break

        agent.learn()

        states = next_states

    if args.wandb_key:
        wandb.finish()
    return agent, save_str


def save_best_version(env_name, agent, save_str, seeds=10):
    agent.load_checkpoints()

    best_total_reward = float("-inf")
    best_frames = None

    for _ in tqdm(range(seeds)):
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

    utils.save_animation(best_frames, f"environments/{env_name}-{save_str}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "--n_games",
        default=2000,
        type=int,
        help="Number of episodes (games) to run during training",
    )
    parser.add_argument(
        "--num_envs",
        default=8,
        type=int,
        help="Number of environments to run in parallel",
    )
    parser.add_argument(
        "--norm_flow", action="store_true", help="Use normalizing flow-based policy"
    )
    parser.add_argument(
        "--num_flows", default=2, type=int, help="Number of flows in the policy"
    )
    parser.add_argument(
        "--wandb_key", type=str, default=None, help="Path to WandB API key text file"
    )
    args = parser.parse_args()

    for fname in ["environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        trained_agent, save_str = run_sac(args)
        save_best_version(args.env, trained_agent, save_str)
    else:
        for env_name in environments:
            args.env = env_name
            trained_agent, save_str = run_sac(args)
            save_best_version(env_name, trained_agent, save_str)
