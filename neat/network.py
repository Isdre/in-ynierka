from neat.graphs import feed_forward_layers, required_for_output


class FeedForwardNetwork(object):
    def __init__(self, genome, inputs, outputs, node_evals):
        self.genome = genome
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            response = self.genome.neurons[node].calculate_response(node_inputs)
            self.values[node] = response
            self.genome.neurons[node].response = response

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        connections = []
        input_output_keys = []

        for cgk, cgv in genome.connections.items():
            if cgv.enabled:
                connections.append(cgk)
                input_output_keys.append((cgv.input_id, cgv.output_id))

        layers = feed_forward_layers(config.input_keys, config.output_keys, input_output_keys)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = []
                for conn_key in connections:
                    inode = genome.connections[conn_key].input_id
                    onode = genome.connections[conn_key].output_id
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.neurons[node]

                node_evals.append((node, ng.response, inputs))

        return FeedForwardNetwork(genome,config.input_keys, config.output_keys, node_evals)


class RecurrentNetwork(object):
    def __init__(self, genome, inputs, outputs, node_evals):
        self.genome = genome
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0
            for node, _, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0

        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            response = self.genome.neurons[node].calculate_response(node_inputs)
            ovalues[node] = response
            self.genome.neurons[node].response = ovalues[node]

        # print("RecurrentNetwork.activate:  outputs={}".format([ovalues[i] for i in self.output_nodes]))

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        connections = []
        all_conn_keys = []

        for cgk, cgv in genome.connections.items():
            if cgv.enabled:
                connections.append(cgv)
                all_conn_keys.append((cgv.input_id, cgv.output_id))

        required = required_for_output(config.input_keys, config.output_keys, all_conn_keys)

        node_inputs = {}
        for c in connections:
            i, o = c.input_id, c.output_id

            if o not in required and i not in required:
                continue
            node_inputs.setdefault(o, []).append((i, c.weight))

        node_evals = []
        for node_key, inp in node_inputs.items():
            ng = genome.neurons[node_key]
            node_evals.append((node_key, ng.response, inp))

        return RecurrentNetwork(genome, config.input_keys, config.output_keys, node_evals)