import multiprocessing
import os

import pickle
import datetime
import shutil



import gymnasium as gym
import neat
import matplotlib
matplotlib.use('TkAgg')
import visualize

runs_per_net = 10
max_steps = 10000


def eval_genome(genome, config):
    net = neat.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make('InvertedDoublePendulum-v5')
        observation, info = env.reset()

        fitness = 0.0
        for step in range(max_steps):
            action = net.activate(observation)

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
    config_path = os.path.join(local_dir, 'config_feedforward.txt')
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
    shutil.copy(config_path, output_folder + "/config_feedforward.txt")

    with open(output_folder + '/winner-feedforward.pickle', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename=output_folder+"/feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename=output_folder+"/feedforward-speciation.svg")

    neat.graphs.visualize_genome(winner, filename=output_folder+"/feedforward-winner.svg", show=True)


if __name__ == '__main__':
    run()