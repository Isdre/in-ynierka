import random
from neat.activation import ActivationFunction

class Gene:
    gene_id = 0

    @staticmethod
    def get_next_id():
        Gene.gene_id += 1
        return Gene.gene_id

class NeuronGene:
    def __init__(self, gene_id, bias: float, activation: ActivationFunction):
        if gene_id is None:
            gene_id = Gene.get_next_id()
        self.gene_id = gene_id
        self.bias = bias
        self.activation = activation

    @staticmethod
    def crossover(neuron1, neuron2):
        if neuron1.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        bias = random.choice([neuron1.bias, neuron2.bias])
        activation = neuron1.activation if random.random() < 0.5 else neuron2.activation

        return NeuronGene(neuron1.id, bias, activation)

class LinkGene:
    def __init__(self, gene_id, input_id, output_id, weight, recurrent=False, enabled=True):
        if gene_id is None:
            gene_id = Gene.get_next_id()
        self.gene_id = gene_id
        self.input_id = input_id
        self.output_id = output_id
        self.weight = weight
        self.value = 0.0
        self.recurrent = recurrent
        self.enabled = enabled

    def calc(self, input_value):
        self.value = input_value * self.weight


    @staticmethod
    def crossover(link1, link2):
        if link1.gene_id != link2.gene_id:
            raise ValueError("Cannot crossover links with different input/output IDs")

        weight = link1.weight if random.random() < 0.5 else link2.weight
        enabled = True if link1.enable and link2.enabled else False

        return LinkGene(link1.gene_id, link1.input_id, link1.output_id, weight, enabled)