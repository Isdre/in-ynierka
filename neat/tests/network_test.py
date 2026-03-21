from neat.network import FeedForwardNetwork
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.config import Config
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction

class TestNetwork:
    def test_create_activate_one_layer(self):
        genome = Genome(1, 1, 1)
        input_neuron = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link = LinkGene(3, 1, 2, 1.0, False, True)

        genome.neurons[1] = input_neuron
        genome.neurons[2] = output_neuron
        genome.connections[3] = link

        for genome_key, genome_value in genome.connections.items():
            assert genome_value.enabled is True

        config = Config()
        config.input_keys = [1]
        config.output_keys = [2]
        config.connections = genome.connections.keys()

        network = FeedForwardNetwork.create(genome, config)

        assert network is not None

        output = network.activate([0.5])
        # print(genome.neurons[1].response)
        # print(output)

        assert output == [0.5]

    def test_create_activate_two_layers(self):
        genome = Genome(1, 1, 1)
        input_neuron = NeuronGene(1, 1.0, ActivationFunction('linear'), AggregationFunction('sum'))
        inner_neuron = NeuronGene(2, 1.0, ActivationFunction('linear'), AggregationFunction('sum'))
        output_neuron = NeuronGene(3, 1.0, ActivationFunction('linear'), AggregationFunction('sum'))
        link1 = LinkGene(3, 1, 2, 2.0, False, True)
        link2 = LinkGene(4, 2, 3, 2.0, False, True)

        genome.neurons[1] = input_neuron
        genome.neurons[2] = inner_neuron
        genome.neurons[3] = output_neuron

        genome.connections[3] = link1
        genome.connections[4] = link2

        for genome_key, genome_value in genome.connections.items():
            assert genome_value.enabled is True

        config = Config()
        config.input_keys = [1]
        config.output_keys = [3]
        config.connections = genome.connections.keys()

        network = FeedForwardNetwork.create(genome, config)

        assert network is not None

        output = network.activate([0.5])
        # print(genome.neurons[1].response)
        # print(output)

        assert output == [5.0]

