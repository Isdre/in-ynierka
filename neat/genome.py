from neat.genes import NeuronGene, LSTMGene, GRUGene, SiTGRUGene, LinkGene
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
import random

class Genome:
    def __init__(self, genome_id, num_inputs, num_outputs):
        self.genome_id = genome_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = {}
        self.connections = {}
        self.fitness = None

    def initialize_random(self, innovation_tracker, initial_connections, config):
        for _ in range(initial_connections):
            self.mutate_add_link(innovation_tracker, config)

    def normalize_weights(self, weight_range=(-1.0, 1.0)):
        for link in self.connections.values():
            link.weight = max(min(link.weight, weight_range[1]), weight_range[0])

    @staticmethod
    def crossover(genome1, genome2):
        if genome1.num_inputs != genome2.num_inputs or genome1.num_outputs != genome2.num_outputs:
            raise ValueError("Cannot crossover individuals with different input/output counts")

        if genome1.fitness > genome2.fitness:
            fitter, other = genome1, genome2
        else:
            fitter, other = genome2, genome1

        child_genome = Genome(None, fitter.num_inputs, fitter.num_outputs)

        for neuron_id in fitter.neurons.keys():
            if neuron_id in other.neurons:
                child_neuron = fitter.neurons[neuron_id].crossover(other.neurons[neuron_id])
            else:
                child_neuron = fitter.neurons[neuron_id]
            child_genome.neurons[neuron_id] = child_neuron

        for link_id in fitter.connections.keys():
            if link_id in other.connections:
                link1 = fitter.connections[link_id]
                link2 = other.connections[link_id]
                child_link = LinkGene.crossover(link1, link2)
            else:
                child_link = fitter.connections[link_id]
            child_genome.connections[link_id] = child_link

        return child_genome

    def mutate(self, config, innovation_tracker):
        a = random.random()
        if a < config.add_link_prob:
            self.mutate_add_link(innovation_tracker, config)
        elif a < config.add_link_prob + config.add_neuron_prob:
            self.mutate_add_neuron(innovation_tracker, config)
        # elif config.add_link_prob + config.add_neuron_prob < a and a < config.add_link_prob + config.add_neuron_prob + config.remove_link_prob:
        #     self.mutate_remove_link()

    def mutate_add_link(self, innovation_tracker, config):
        possible_inputs = list(self.neurons.keys())
        possible_outputs = list(self.neurons.keys())

        for _ in range(100):
            input_id = random.choice(possible_inputs)
            output_id = random.choice(possible_outputs)

            if (input_id != output_id and (input_id, output_id) not in self.connections
                    and output_id not in self.num_inputs and input_id not in self.num_outputs):
                innovation_id = innovation_tracker.get_new_link_innovation_id(input_id, output_id)
                new_link = LinkGene(innovation_id, input_id, output_id, weight=random.uniform(-1.0, 1.0), enabled=True)
                self.connections[innovation_id] = new_link
                return

    def mutate_remove_link(self):
        if not self.connections:
            return

        link_to_remove = random.choice(list(self.connections.keys()))
        del self.connections[link_to_remove]

    def mutate_add_neuron(self, innovation_tracker, config):
        if not self.connections:
            return

        link_to_split = random.choice(list(self.connections.values()))
        if not link_to_split.enabled:
            return

        link_to_split.enabled = False

        new_neuron_id = innovation_tracker.get_new_neuron_id(link_to_split.innovation_id)
        new_neuron = None

        neuron_chosen = random.random()

        if neuron_chosen < config.add_neuron_gene_prob:
            new_neuron = NeuronGene(new_neuron_id, bias=random.uniform(-1.0, 1.0), activation=ActivationFunction('sigmoid'), aggregation=AggregationFunction('sum'))
        elif neuron_chosen < config.add_neuron_gene_prob + config.add_lstm_gene_prob:
            new_neuron = LSTMGene(new_neuron_id, activation=ActivationFunction('tanh'), aggregation=AggregationFunction('sum'), hidden_size=10)
            new_neuron.reset_parameters()
        elif neuron_chosen < config.add_neuron_gene_prob + config.add_lstm_gene_prob + config.add_gru_gene_prob:
            new_neuron = GRUGene(new_neuron_id, activation=ActivationFunction('tanh'), aggregation=AggregationFunction('sum'), hidden_size=10)
            new_neuron.reset_parameters()
        elif neuron_chosen < config.add_neuron_gene_prob + config.add_lstm_gene_prob + config.add_gru_gene_prob + config.add_sitgru_gene_prob:
            new_neuron = SiTGRUGene(new_neuron_id, activation=ActivationFunction('tanh'), aggregation=AggregationFunction('sum'), hidden_size=10)
            new_neuron.reset_parameters()

        self.neurons[new_neuron_id] = new_neuron

        in_to_new_id = innovation_tracker.get_new_link_innovation_id(link_to_split.input_id, new_neuron_id)
        out_from_new_id = innovation_tracker.get_new_link_innovation_id(new_neuron_id, link_to_split.output_id)

        in_to_new_link = LinkGene(in_to_new_id, link_to_split.input_id, new_neuron_id, weight=1.0, enabled=True)
        out_from_new_link = LinkGene(out_from_new_id, new_neuron_id, link_to_split.output_id, weight=link_to_split.weight, enabled=True)

        self.connections[in_to_new_id] = in_to_new_link
        self.connections[out_from_new_id] = out_from_new_link

    def mutate_remove_neuron(self):
        non_io_neurons = [nid for nid in self.neurons if nid > max(self.num_inputs+self.num_outputs)]
        if not non_io_neurons:
            return

        neuron_to_remove_id = random.choice(non_io_neurons)
        del self.neurons[neuron_to_remove_id]

        connections_to_remove = [lid for lid in self.connections if lid[0] == neuron_to_remove_id or lid[1] == neuron_to_remove_id]
        for lid in connections_to_remove:
            del self.connections[lid]

    def calculate_genetic_distance(self, other, config):
        matching_genes = 0
        disjoint_genes = 0
        excess_genes = 0
        weight_diff = 0.0

        self_innovations = {link.innovation_id: link for link in self.connections.values()}
        other_innovations = {link.innovation_id: link for link in other.connections.values()}

        all_ids = set(self_innovations.keys()).union(other_innovations.keys())
        max_self = max(self_innovations.keys(), default=0)
        max_other = max(other_innovations.keys(), default=0)

        for innovation_id in all_ids:
            in_self = innovation_id in self_innovations
            in_other = innovation_id in other_innovations
            if in_self and in_other:
                matching_genes += 1
                weight_diff += abs(self_innovations[innovation_id].weight - other_innovations[innovation_id].weight)
            else:
                if innovation_id > max_self or innovation_id > max_other:
                    excess_genes += 1
                else:
                    disjoint_genes += 1

        avg_weight_diff = weight_diff / matching_genes if matching_genes > 0 else 0.0
        n = len(all_ids) if all_ids else 1
        return (config.c1 * disjoint_genes + config.c2 * excess_genes) / n + config.c3 * avg_weight_diff