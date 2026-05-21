import os

import pickle
import datetime
import shutil

import cart_pole
import neat
import matplotlib
matplotlib.use('TkAgg')
from neat import visualize
import multiprocessing

runs_per_net = 5
simulation_seconds = 60.0
run_times: int = 20

def eval_genome(genome, config):
    net = neat.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            fitness = sim.t

            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

        fitnesses.append(fitness)

    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    for id, genome in genomes.items():
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward.txt')

    today = datetime.datetime.now()
    output_folder = f"results/{today.year}-{today.month:02d}-{today.day:02d}-{today.hour:02d}-{today.minute:02d}"
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(config_path, os.path.join(output_folder, "config_feedforward.txt"))

    all_runs_stats = []

    for run_idx in range(run_times):
        config = neat.Config.read_from_file(config_path)

        print(f"\n{'=' * 20} RUN {run_idx + 1}/{run_times} {'=' * 20}\n")

        inv = neat.InnovationTracker(config.num_inputs, config.num_outputs)
        pop = neat.Population(config, inv)

        pop.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = pop.run(pe.evaluate, config.generation)

        with open(os.path.join(output_folder, f'winner-feedforward-run{run_idx}.pickle'), 'wb') as f:
            pickle.dump(winner, f)

        visualize.plot_stats(stats, ylog=True, view=False, filename=output_folder + f'/feedforward-run{run_idx}-fitness.svg')
        visualize.plot_species(stats, view=False, filename=output_folder + f'/feedforward-run{run_idx}-speciation.svg')
        all_runs_stats.append(stats)

    print("\nTworzenie zbiorczych wykresów...")
    visualize.plot_aggregated_stats(
        all_runs_stats,
        view=False,
        filename=os.path.join(output_folder, "aggregated-fitness.svg")
    )


if __name__ == '__main__':
    run()
