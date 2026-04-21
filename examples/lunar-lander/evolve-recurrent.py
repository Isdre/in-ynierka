import multiprocessing
import os

import pickle
import datetime
import shutil
import numpy as np
import gymnasium as gym
import neat
import matplotlib
matplotlib.use('TkAgg')
import visualize

runs_per_net = 10
max_steps = 1000


def eval_genome(genome, config):
    net = neat.RecurrentNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make('LunarLander-v3', continuous=True)
        observation, info = env.reset()

        fitness = 0.0
        for step in range(max_steps):
            action = net.activate(observation)
            action = np.array(action)

            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        env.close()
        fitnesses.append(fitness)

    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_recurrent.txt')
    config = neat.Config.read_from_file(config_path)

    inv = neat.InnovationTracker(config.num_inputs, config.num_outputs)
    pop = neat.Population(config, inv)
    pop.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()

    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, config.generation)

    today = datetime.datetime.now()

    output_folder = f"results/{today.year}-{today.month:02d}-{today.day:02d}-{today.hour:02d}-{today.minute:02d}"

    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(config_path, output_folder + "/config_recurrent.txt")

    with open(output_folder+'/winner-recurrent.pickle', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename=output_folder+"/recurrent-fitness.svg")
    visualize.plot_species(stats, view=True, filename=output_folder+"/recurrent-speciation.svg")

    output_path = output_folder+"/recurrent-winner.svg"

    neat.graphs.visualize_genome(winner, filename=output_path, show=True)


if __name__ == '__main__':
    run()