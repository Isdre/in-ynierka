import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(1,len(statistics.most_fit_genomes)+1)
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="Średnia")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="±1 Odchylenie std.")
    plt.plot(generation, best_fitness, 'r-', label="Najlepsze")

    plt.title("Średnie i najlepsze przystosowanie w każdej generacji")
    plt.xlabel("Generacje")
    plt.ylabel("Przystosowanie")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view and plt.get_backend() != 'agg':
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    t_values = [t for t, I, v, u in spikes]
    v_values = [v for t, I, v, u in spikes]
    u_values = [u for t, I, v, u in spikes]
    I_values = [I for t, I, v, u in spikes]

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(3, 1, 2)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(3, 1, 3)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig



def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(1, num_generations+1), *curves)

    plt.title("Skumulowany rozmiar gatunków w każdej generacji")
    plt.ylabel("Rozmiar gatunku")
    plt.xlabel("Generacje")

    if view:
        plt.show()
    else:
        plt.savefig(filename)

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        if show_disabled:
            warnings.warn("show_disabled has no effect when prune_unused is True")

        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    for n in genome.neurons.keys():
        if n in inputs or n in outputs:
            continue

        gene_type = type(genome.neurons[n]).__name__
        if gene_type == 'LSTMGene':
            default_color = 'mediumpurple'
        elif gene_type == 'GRUGene':
            default_color = 'orange'
        else:
            default_color = 'lightgreen'

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, default_color)}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input = cg.input_id
            output = cg.output_id
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def plot_aggregated_stats(all_stats, ylog=False, view=False, filename='aggregated_fitness.svg'):
    if plt is None:
        warnings.warn("Ta funkcja wymaga biblioteki matplotlib.")
        return

    max_generations = max(len(stats.most_fit_genomes) for stats in all_stats)

    all_best_fitnesses = []
    all_avg_fitnesses = []

    completion_points = []

    for stats in all_stats:
        best_fit = [c.fitness for c in stats.most_fit_genomes]
        avg_fit = stats.get_fitness_mean()

        run_length = len(best_fit)
        final_best_fitness = best_fit[-1]
        if run_length < max_generations:
            completion_points.append((run_length, final_best_fitness))

        while len(best_fit) < max_generations:
            best_fit.append(best_fit[-1])
        while len(avg_fit) < max_generations:
            avg_fit.append(avg_fit[-1])

        all_best_fitnesses.append(best_fit)
        all_avg_fitnesses.append(avg_fit)

    best_matrix = np.array(all_best_fitnesses)
    avg_matrix = np.array(all_avg_fitnesses)

    mean_of_best = np.mean(best_matrix, axis=0)
    std_of_best = np.std(best_matrix, axis=0)
    mean_of_avg = np.mean(avg_matrix, axis=0)

    generations = range(1, max_generations + 1)

    plt.figure(figsize=(10, 6))

    line_best, = plt.plot(generations, mean_of_best, 'r-', label="Średnie przystosowanie najlepszego osobnika")

    plt.fill_between(generations,
                     mean_of_best - std_of_best,
                     mean_of_best + std_of_best,
                     color='r', alpha=0.2, label="±1 Odchylenie std. najlepszych osobników")

    plt.plot(generations, mean_of_avg, 'b-', label="Średnie przystosowanie całej populacji")

    if completion_points:
        unique_completions = list(set(completion_points))

        x_coords = [p[0] for p in unique_completions]
        y_coords = [p[1] for p in unique_completions]

        plt.vlines(x_coords, ymin=0, ymax=y_coords, colors='black', linestyles='dashed', alpha=0.5)
        from matplotlib.lines import Line2D
        line_vlines = Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
        # Dodajemy ją do listy legend
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        current_handles.append(line_vlines)
        current_labels.append("Zakończenie pojedynczej ewolucji")
        plt.legend(current_handles, current_labels, loc="best")

    else:
        plt.legend(loc="best")
    # -----------------------------------------------------------------------------------

    plt.title(f"Zagregowane statystyki z {len(all_stats)} niezależnych ewolucji")
    plt.xlabel("Generacje")
    plt.ylabel("Fitness")
    plt.grid()

    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view and plt.get_backend() != 'agg':
        plt.show()

    plt.close()


def plot_aggregated_species(all_stats, view=False, filename='aggregated_species.svg'):
    if plt is None:
        warnings.warn("Ta funkcja wymaga biblioteki matplotlib.")
        return

    max_generations = max(len(stats.most_fit_genomes) for stats in all_stats)

    all_species_counts = []

    for stats in all_stats:
        species_sizes = stats.get_species_sizes()

        counts = [sum(1 for size in gen if size > 0) for gen in species_sizes]

        while len(counts) < max_generations:
            counts.append(counts[-1])

        all_species_counts.append(counts)

    counts_matrix = np.array(all_species_counts)
    mean_counts = np.mean(counts_matrix, axis=0)
    std_counts = np.std(counts_matrix, axis=0)

    generations = range(max_generations)

    plt.figure(figsize=(10, 6))

    plt.plot(generations, mean_counts, 'g-', label="Średnia liczba gatunków")

    plt.fill_between(generations,
                     mean_counts - std_counts,
                     mean_counts + std_counts,
                     color='g', alpha=0.2, label="±1 Odchylenie std.")

    plt.title(f"Zagregowana różnorodność gatunków ({len(all_stats)} ewolucji)")
    plt.xlabel("Generacje")
    plt.ylabel("Liczba gatunków w populacji")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig(filename)
    if view and plt.get_backend() != 'agg':
        plt.show()

    plt.close()