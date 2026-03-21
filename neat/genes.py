import random
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction

class NeuronGene:
    def __init__(self, gene_id, bias: float, activation: ActivationFunction, aggregation: AggregationFunction, response: float = 0.0):
        self.gene_id = gene_id
        self.bias = bias
        self.activation = activation
        self.aggregation = aggregation
        self.response = response

    @staticmethod
    def crossover(neuron1, neuron2):
        if neuron1.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        bias = neuron1.bias
        activation = neuron1.activation
        aggregation = neuron1.aggregation
        response = neuron1.response

        return NeuronGene(neuron1.gene_id, bias, activation, aggregation, response)

class LinkGene:
    def __init__(self, innovation_id, input_id, output_id, weight, recurrent=False, enabled=True):
        self.innovation_id = innovation_id
        self.input_id = input_id
        self.output_id = output_id
        self.weight = weight
        self.value = 0.0
        self.recurrent = recurrent
        self.enabled = enabled

    def distance(self, other):
        if self.innovation_id != other.innovation_id:
            raise ValueError("Cannot compute distance between links with different IDs")
        return abs(self.weight - other.weight) + (0 if self.enabled == other.enabled else 1.0)

    @staticmethod
    def crossover(link1, link2):
        if link1.innovation_id != link2.innovation_id:
            raise ValueError("Cannot crossover links with different input/output IDs")

        weight = link1.weight
        enabled = True if link1.enabled and link2.enabled else False
        recurrent = link1.recurrent
        return LinkGene(link1.innovation_id, link1.input_id, link1.output_id, weight, recurrent, enabled)