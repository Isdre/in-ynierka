from neat.network import FeedForwardNetwork
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.config import Config
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
from neat.innovation import InnovationTracker
import random


class TestGenome:
    def test_create_genome(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(3, 1, 2, 1.0, False, True)

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        for genome_key, genome_value in genome.connections.items():
            assert genome_value.enabled is True

        config = Config()
        config.input_keys = [1]
        config.output_keys = [2]
        config.connections = genome.connections.keys()

        network = FeedForwardNetwork.create(genome, config)

        assert network is not None

        output = network.activate([0.5])
        assert output == [0.5]

    def test_crossover_genomes(self):
        parent1 = Genome(1, [1], [2])
        parent2 = Genome(2, [1], [2])

        input_neuron1 = NeuronGene(1, 0.5, ActivationFunction('sigmoid'), AggregationFunction('sum'))
        output_neuron1 = NeuronGene(2, 0.5, ActivationFunction('sigmoid'), AggregationFunction('sum'))
        link1 = LinkGene(3, 1, 2, 0.8, False, True)

        input_neuron2 = NeuronGene(1, -0.5, ActivationFunction('tanh'), AggregationFunction('sum'))
        output_neuron2 = NeuronGene(2, -0.5, ActivationFunction('tanh'), AggregationFunction('sum'))
        link2 = LinkGene(3, 1, 2, -0.8, False, True)

        parent1.neurons[1] = input_neuron1
        parent1.neurons[2] = output_neuron1
        parent1.connections[(1, 2)] = link1

        parent2.neurons[1] = input_neuron2
        parent2.neurons[2] = output_neuron2
        parent2.connections[(1, 2)] = link2

        parent1.fitness = 10.0
        parent2.fitness = 10.0

        child = Genome.crossover(parent1, parent2)

        assert child is not None
        assert len(child.neurons) == 2
        assert len(child.connections) == 1

    def test_crossover_fitter_parent_dominates(self):
        parent1 = Genome(1, [1], [2])
        parent2 = Genome(2, [1], [2])

        input_neuron1 = NeuronGene(1, 0.5, ActivationFunction('sigmoid'), AggregationFunction('sum'))
        output_neuron1 = NeuronGene(2, 0.5, ActivationFunction('sigmoid'), AggregationFunction('sum'))
        link1 = LinkGene(1, 1, 2, 0.8, False, True)
        extra_link1 = LinkGene(2, 2, 1, 0.5, False, True)  # dodatkowe połączenie tylko w parent1

        parent1.neurons[1] = input_neuron1
        parent1.neurons[2] = output_neuron1
        parent1.connections[(1, 2)] = link1
        parent1.connections[(2, 1)] = extra_link1

        input_neuron2 = NeuronGene(1, -0.5, ActivationFunction('tanh'), AggregationFunction('sum'))
        output_neuron2 = NeuronGene(2, -0.5, ActivationFunction('tanh'), AggregationFunction('sum'))
        link2 = LinkGene(1, 1, 2, -0.8, False, True)

        parent2.neurons[1] = input_neuron2
        parent2.neurons[2] = output_neuron2
        parent2.connections[(1, 2)] = link2

        parent1.fitness = 20.0
        parent2.fitness = 10.0

        child = Genome.crossover(parent1, parent2)

        assert len(child.connections) == 2

    def test_crossover_raises_on_different_io(self):
        parent1 = Genome(1, [1,2], [3])
        parent2 = Genome(2,  [1], [2])
        parent1.fitness = 1.0
        parent2.fitness = 1.0

        try:
            Genome.crossover(parent1, parent2)
            assert False, "Powinien zostać zgłoszony ValueError"
        except ValueError:
            pass

    def test_mutate_genome_add_neuron(self):
        config = Config()
        config.add_neuron_gene_prob = 1.0

        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(3, 1, 2, 1.0, False, True)
        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        innovation_tracker = InnovationTracker(1, 1)
        genome.mutate_add_neuron(innovation_tracker, config)

        assert len(genome.neurons) == 3
        assert len(genome.connections) == 3

    def test_mutate_add_neuron_disables_old_link(self):
        config = Config()
        config.add_neuron_gene_prob = 1.0

        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(3, 1, 2, 1.0, False, True)
        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        innovation_tracker = InnovationTracker(1, 1)
        genome.mutate_add_neuron(innovation_tracker, config)

        # Oryginalne połączenie powinno być wyłączone
        assert genome.connections[(1, 2)].enabled is False

    def test_mutate_genome_add_connection(self):
        config = Config()

        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron


        innovation_tracker = InnovationTracker(1, 2)
        genome.initialize_random(innovation_tracker, 1, config)
        genome.mutate_add_link(innovation_tracker, config)

    def test_mutate_remove_link(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(3, 1, 2, 1.0, False, True)
        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        genome.mutate_remove_link()

        assert len(genome.connections) == 0

    def test_mutate_remove_link_empty_genome(self):
        genome = Genome(1, [1], [2])
        # Nie powinien zgłosić błędu przy braku połączeń
        genome.mutate_remove_link()
        assert len(genome.connections) == 0

    def test_mutate_remove_neuron(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        hidden_neuron = NeuronGene(3, 0.0, ActivationFunction('relu'), AggregationFunction('sum'))
        link1 = LinkGene(1, 1, 3, 1.0, False, True)
        link2 = LinkGene(2, 3, 2, 1.0, False, True)

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.neurons[3] = hidden_neuron
        genome.connections[(1, 3)] = link1
        genome.connections[(3, 2)] = link2

        genome.mutate_remove_neuron()

        # Neuron ukryty i jego połączenia powinny zostać usunięte
        assert 3 not in list(genome.neurons.keys())
        assert len(genome.connections) == 0

    def test_mutate_remove_neuron_no_hidden(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron

        genome.mutate_remove_neuron()
        assert len(genome.neurons) == 2

    def test_normalize_weights(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(1, 1, 2, 5.0, False, True)  # waga poza zakresem

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        genome.normalize_weights(weight_range=(-1.0, 1.0))

        assert genome.connections[(1, 2)].weight == 1.0

    def test_normalize_weights_negative(self):
        genome = Genome(1, [1], [2])
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(1, 1, 2, -5.0, False, True)  # ujemna waga poza zakresem

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[(1, 2)] = link

        genome.normalize_weights(weight_range=(-1.0, 1.0))

        assert genome.connections[(1, 2)].weight == -1.0

    def test_calculate_genetic_distance_identical(self):
        genome1 = Genome(1, [1], [2])
        genome2 = Genome(2, [1], [2])

        link1 = LinkGene(1, 1, 2, 0.5, False, True)
        link2 = LinkGene(1, 1, 2, 0.5, False, True)

        genome1.connections[(1, 2)] = link1
        genome2.connections[(1, 2)] = link2

        config = Config()
        config.c1 = 1.0
        config.c2 = 1.0
        config.c3 = 0.4
        distance = genome1.calculate_genetic_distance(genome2, config)

        assert distance == 0.0

    def test_calculate_genetic_distance_different_weights(self):
        genome1 = Genome(1, [1], [2])
        genome2 = Genome(2, [1], [2])

        link1 = LinkGene(1, 1, 2, 1.0, False, True)
        link2 = LinkGene(1, 1, 2, 0.0, False, True)

        genome1.connections[(1, 2)] = link1
        genome2.connections[(1, 2)] = link2

        config = Config()
        config.c1 = 1.0
        config.c2 = 1.0
        config.c3 = 0.4

        distance = genome1.calculate_genetic_distance(genome2, config)

        # Różnica wag wynosi 1.0 * c3 = 0.4
        assert distance == 0.4

    def test_calculate_genetic_distance_disjoint(self):
        genome1 = Genome(1, [1], [2])
        genome2 = Genome(2, [1], [2])

        link1 = LinkGene(1, 1, 2, 0.5, False, True)
        link2 = LinkGene(2, 1, 2, 0.5, False, True)

        genome1.connections[(1, 2)] = link1
        genome2.connections[(1, 2)] = link2

        config = Config()
        config.c1 = 1.0
        config.c2 = 1.0
        config.c3 = 0.4

        distance = genome1.calculate_genetic_distance(genome2, config)

        assert distance > 0.0

    def test_initialize_random(self):
        config = Config()

        genome = Genome(1, [1,2], [3,4])
        for i in range(1, 5):
            genome.neurons[i] = NeuronGene(i, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))

        innovation_tracker = InnovationTracker(2, 2)
        genome.initialize_random(innovation_tracker, initial_connections=3, config=config)

        assert len(genome.connections) <= 3