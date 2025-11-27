from neat.graphs import feed_forward_layers

class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        # print()
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v
            # print(f"Input node {k} set to {v}")

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
                # print(f"Incoming value from node {i} to node {node} with weight {w}: {self.values[i] * w}")
            # print(" Aggregating inputs:", node_inputs)
            s = agg_func(node_inputs)
            response = act_func(bias + s)
            self.values[node] = response

            # print(f"{s} {bias} {response} {act_func.__name__} {agg_func.__name__}")
            # print(f"Node {node} calculated value {self.values[node]}")

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):

        # Gather expressed connections.
        connections = [cgk for cgk, cgv in genome.connections.items() if cgv.enabled]

        layers = feed_forward_layers(config.input_keys, config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.neurons[node]
                aggregation_function = ng.aggregation
                activation_function = ng.activation
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.input_keys, config.output_keys, node_evals)