import os
import pickle

import gymnasium as gym
import neat

import sys

def run(net, episodes=10, render=True, camera_distance=4.0):
    fitnesses = []

    if render:
        env = gym.make('InvertedDoublePendulum-v5', render_mode='human')
    else:
        env = gym.make('InvertedDoublePendulum-v5')

    try:
        for episode in range(episodes):
            observation, info = env.reset()

            if episode == 0 and render and hasattr(env.unwrapped, 'mujoco_renderer'):
                renderer = env.unwrapped.mujoco_renderer
                if renderer.viewer is not None:
                    renderer.viewer.cam.distance = camera_distance

            fitness = 0.0
            step = 0

            while True:
                step += 1
                action = net.activate(observation)

                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward

                if terminated or truncated:
                    break

            fitnesses.append(fitness)
            print(f"Episode {episode + 1}: steps={step}, fitness={fitness:.2f}")

    finally:
        env.close()

    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)

    print(f"\nResults over {episodes} episodes:")
    print(f"  Average fitness: {avg_fitness:.2f}")
    print(f"  Max fitness: {max_fitness:.2f}")
    print(f"  Min fitness: {min_fitness:.2f}")

    return fitnesses


def load_and_test(genome_path, config_path, episodes=10, render=True, camera_distance=4.0):

    config = neat.Config.read_from_file(config_path)



    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    print('Loaded genome:')
    print(genome)

    # Create the network
    net = neat.FeedForwardNetwork.create(genome, config)

    # Test the network
    return run(net, episodes=episodes, render=render, camera_distance=camera_distance)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    result_path = "results/2026-04-13-11-51"
    config_path = os.path.join(local_dir, result_path + "/config_feedforward.txt")
    genome_path = os.path.join(local_dir, result_path + '/winner-feedforward.pickle')

    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=20, render=True, camera_distance=4.0)