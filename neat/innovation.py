import itertools
from collections import defaultdict

class InnovationTracker:
    def __init__(self, num_inputs, num_outputs):
        self.neuron_counter = num_inputs + num_outputs
        self.link_innovations = {}
        self.neuron_innovations = defaultdict(int)
        self.next_innovation_number = 0

    def reset(self):
        self.neuron_counter = 0
        self.link_innovations.clear()
        self.neuron_innovations.clear()
        self.next_innovation_number = 0

    def get_new_neuron_id(self, link_innovation_id):
        neuron_id = self.neuron_innovations.get(link_innovation_id)
        if neuron_id is None:
            self.neuron_counter += 1
            neuron_id = self.neuron_counter
            self.neuron_innovations[link_innovation_id] = neuron_id
        return neuron_id

    def get_new_link_innovation_id(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        innovation_id = self.link_innovations.get(key)
        if innovation_id is None:
            innovation_id = self.next_innovation_number
            self.link_innovations[key] = innovation_id
            self.next_innovation_number += 1
        return innovation_id