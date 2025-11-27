class Genome:
    def __init__(self, genome_id, num_inputs, num_outputs):
        self.genome_id = genome_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = {}
        self.connections = {}
        
    @staticmethod
    def crossover(genome1, genome2, fitness1, fitness2):
        if genome1.num_inputs != genome2.num_inputs or genome1.num_outputs != genome2.num_outputs:
            raise ValueError("Cannot crossover individuals with different input/output counts")

        if fitness1 > fitness2:
            fitter, other = ind1, ind2
        elif fitness2 > fitness1:
            fitter, other = ind2, ind1
        else:
            fitter, other = (ind1, ind2) if random.random() < 0.5 else (ind2, ind1)

        child_genome = Genome(fitter.genome_id, fitter.num_inputs, fitter.num_outputs)

        for neuron_id in fitter.neurons.keys():
            if neuron_id in other.neurons:
                child_neuron = NeuronGene.crossover(fitter.neurons[neuron_id], other.neurons[neuron_id])
            else:
                child_neuron = fitter.neurons[neuron_id]
            child_genome.neurons[neuron_id] = child_neuron

        for link_id in fitter.links.keys():
            if link_id in other.links:
                child_link = LinkGene.crossover(fitter.links[link_id], other.links[link_id])
            else:
                child_link = fitter.links[link_id]
            child_genome.links[link_id] = child_link

        return child_genome

    def mutate_add_link(self, innovation_tracker):
        possible_inputs = list(self.genome.neurons.keys())
        possible_outputs = list(self.genome.neurons.keys())

        input_id = random.choice(possible_inputs)
        output_id = random.choice(possible_outputs)

        if input_id == output_id:
            return

        link_id = (input_id, output_id)
        if link_id in self.genome.links:
            return

        innovation_number = innovation_tracker.get_innovation_number(input_id, output_id)
        new_link = LinkGene(input_id, output_id, weight=random.uniform(-1.0, 1.0), enabled=True)
        self.genome.links[link_id] = new_link

    def mutate_remove_link(self):
        if not self.genome.links:
            return

        link_to_remove = random.choice(list(self.genome.links.keys()))
        del self.genome.links[link_to_remove]

    def mutate_add_neuron(self, innovation_tracker):
        if not self.genome.links:
            return

        link_to_split = random.choice(list(self.genome.links.values()))
        if not link_to_split.enabled:
            return

        link_to_split.enabled = False

        new_neuron_id = innovation_tracker.get_new_neuron_id()
        new_neuron = NeuronGene(new_neuron_id, bias=random.uniform(-1.0, 1.0), activation='sigmoid')
        self.neurons[new_neuron_id] = new_neuron

        in_to_new_id = (link_to_split.input_id, new_neuron_id)
        out_from_new_id = (new_neuron_id, link_to_split.output_id)

        in_to_new_innovation = innovation_tracker.get_innovation_number(link_to_split.input_id, new_neuron_id)
        out_from_new_innovation = innovation_tracker.get_innovation_number(new_neuron_id, link_to_split.output_id)

        in_to_new_link = LinkGene(link_to_split.input_id, new_neuron_id, weight=1.0, enabled=True)
        out_from_new_link = LinkGene(new_neuron_id, link_to_split.output_id, weight=link_to_split.weight, enabled=True)

        self.links[in_to_new_id] = in_to_new_link
        self.links[out_from_new_id] = out_from_new_link

    def mutate_remove_neuron(self):
        non_io_neurons = [nid for nid in self.neurons if nid >= self.num_inputs + self.num_outputs]
        if not non_io_neurons:
            return

        neuron_to_remove_id = random.choice(non_io_neurons)
        del self.neurons[neuron_to_remove_id]

        links_to_remove = [lid for lid in self.links if lid[0] == neuron_to_remove_id or lid[1] == neuron_to_remove_id]
        for lid in links_to_remove:
            del self.links[lid]