import os
import pickle

import cart_pole
import neat
import matplotlib
matplotlib.use('TkAgg')
import visualize
import multiprocessing

runs_per_net = 5
simulation_seconds = 60.0


def eval_genome(genome, config):
    net = neat.RecurrentNetwork.create(genome, config)

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
    config_path = os.path.join(local_dir, 'config_recurrent.txt')
    config = neat.Config.read_from_file(config_path)

    inv = neat.InnovationTracker(config.num_inputs, config.num_outputs)
    pop = neat.Population(config, inv)
    pop.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()

    pop.add_reporter(stats)

    #pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(eval_genomes, config.generation)

    with open('winner-reccurrent.pickle', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="reccurrent-fitness.svg")
    visualize.plot_species(stats, view=True, filename="reccurrent-speciation.svg")

    output_path = "reccurrent-winner.svg"

    neat.graphs.visualize_genome(winner, filename=output_path, show=True)


if __name__ == '__main__':
    run()
