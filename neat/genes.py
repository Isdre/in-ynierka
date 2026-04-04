import random
from copy import deepcopy
from typing import Protocol
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
import math

class BaseGene(Protocol):
    def __init__(self, gene_id):
        self.gene_id = gene_id

    def crossover(self, neuron2):
        pass

    def calculate_reponse(self, inputs):
        pass

class NeuronGene(BaseGene):
    def __init__(self, gene_id, bias: float, activation: ActivationFunction, aggregation: AggregationFunction, response: float = 0.0):
        self.gene_id = gene_id
        self.bias = bias
        self.activation = activation
        self.aggregation = aggregation
        self.response = response

    def crossover(self, neuron2):
        if self.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        bias = self.bias
        activation = self.activation
        aggregation = self.aggregation
        response = self.response

        return NeuronGene(self.gene_id, bias, activation, aggregation, response)

    def calculate_response(self, inputs):
        s = self.aggregation.function(inputs)
        self.response = self.activation.function(self.bias + s)
        return self.response

class LSTMGene(BaseGene):
    sigmoid = ActivationFunction('sigmoid')
    tanh = ActivationFunction('tanh')

    def __init__(self, gene_id, activation: ActivationFunction, aggregation: AggregationFunction, hidden_size, weight_ih=None, bias_ih=None, weight_hh=None, bias_hh=None, hidden_state=None, cell_state=None, response: float = 0.0):
        self.gene_id = gene_id

        self.activation = activation
        self.aggregation = aggregation

        self.response = response

        self.hidden_size = hidden_size

        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_state = hidden_state if hidden_state is not None else [0.0] * hidden_size
        self.cell_state = cell_state if cell_state is not None else [0.0] * hidden_size

        if self.weight_ih is None or self.bias_ih is None or self.weight_hh is None or self.bias_hh is None:
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.weight_ih = [random.uniform(-stdv, stdv) for _ in range(4 * self.hidden_size)]
        self.weight_hh = [[random.uniform(-stdv, stdv) for _ in range(4 * self.hidden_size)] for _ in
                          range(self.hidden_size)]
        self.bias_ih = [random.uniform(-stdv, stdv) for _ in range(4 * self.hidden_size)]
        self.bias_hh = [random.uniform(-stdv, stdv) for _ in range(4 * self.hidden_size)]

        self.hidden_state = [0.0] * self.hidden_size
        self.cell_state = [0.0] * self.hidden_size

    def crossover(self, neuron2):
        if self.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        new_weight_ih = deepcopy(self.weight_ih)
        new_weight_hh = deepcopy(self.weight_hh)
        new_bias_ih = deepcopy(self.bias_ih)
        new_bias_hh = deepcopy(self.bias_hh)
        new_cell_state = deepcopy(self.cell_state)
        new_hidden_state = deepcopy(self.hidden_state)

        return LSTMGene(
            self.gene_id,
            self.activation,
            self.aggregation,
            self.hidden_size,
            deepcopy(new_weight_ih),
            deepcopy(new_bias_ih),
            deepcopy(new_weight_hh),
            deepcopy(new_bias_hh),
            deepcopy(new_hidden_state),
            deepcopy(new_cell_state),
            self.response
        )

    def calculate_response(self, inputs):
        x = self.aggregation.function(inputs)

        new_hidden_state = []
        new_cell_state = []

        for j in range(self.hidden_size):
            recurrent_effect = [
                sum(self.hidden_state[k] * self.weight_hh[k][offset + j] for k in range(self.hidden_size)) for offset in
                (0, self.hidden_size, 2 * self.hidden_size, 3 * self.hidden_size)]

            i_val = LSTMGene.sigmoid.function(
                x * self.weight_ih[j] + self.bias_ih[j] + recurrent_effect[0] + self.bias_hh[j]
            )

            f_val = LSTMGene.sigmoid.function(
                x * self.weight_ih[self.hidden_size + j] + self.bias_ih[self.hidden_size + j] + recurrent_effect[1] +
                self.bias_hh[self.hidden_size + j]
            )

            g_val = LSTMGene.tanh.function(
                x * self.weight_ih[2 * self.hidden_size + j] + self.bias_ih[2 * self.hidden_size + j] +
                recurrent_effect[2] + self.bias_hh[2 * self.hidden_size + j]
            )

            o_val = LSTMGene.sigmoid.function(
                x * self.weight_ih[3 * self.hidden_size + j] + self.bias_ih[3 * self.hidden_size + j] +
                recurrent_effect[3] + self.bias_hh[3 * self.hidden_size + j]
            )

            c_new = f_val * self.cell_state[j] + i_val * g_val
            h_new = o_val * LSTMGene.tanh.function(c_new)

            new_cell_state.append(c_new)
            new_hidden_state.append(h_new)

        self.cell_state = new_cell_state
        self.hidden_state = new_hidden_state

        self.response = sum(self.hidden_state)
        return self.response

class GRUGene(BaseGene):
    sigmoid = ActivationFunction('sigmoid')
    tanh = ActivationFunction('tanh')
    
    def __init__(self, gene_id, activation: ActivationFunction, aggregation: AggregationFunction, hidden_size, weight_ih=None, bias_ih=None, weight_hh=None, bias_hh=None, state=None, response: float = 0.0):
        self.gene_id = gene_id

        self.activation = activation
        self.aggregation = aggregation

        self.response = response

        self.hidden_size = hidden_size

        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.state = state

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.weight_ih = [random.uniform(-stdv, stdv) for _ in range(3 * self.hidden_size)]
        self.weight_hh = [[random.uniform(-stdv, stdv) for _ in range(3 * self.hidden_size)] for _ in range(self.hidden_size)]
        self.bias_ih = [random.uniform(-stdv, stdv) for _ in range(3 * self.hidden_size)]
        self.bias_hh = [random.uniform(-stdv, stdv) for _ in range(3 * self.hidden_size)]

        self.state = [0.0] * self.hidden_size

    def crossover(self, neuron2):
        if self.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        return GRUGene(
            self.gene_id,
            self.activation,
            self.aggregation,
            self.hidden_size,
            deepcopy(self.weight_ih),
            deepcopy(self.bias_ih),
            deepcopy(self.weight_hh),
            deepcopy(self.bias_hh),
            deepcopy(self.state),
            self.response
        )

    def calculate_response(self, inputs):
        x = self.aggregation.function(inputs)
        new_state = []

        for j in range(self.hidden_size):
            recurrent_z = sum(self.state[k] * self.weight_hh[k][j] for k in range(self.hidden_size))
            recurrent_r = sum(self.state[k] * self.weight_hh[k][self.hidden_size + j] for k in range(self.hidden_size))

            z_val = GRUGene.sigmoid.function(
                x * self.weight_ih[j] + self.bias_ih[j] + recurrent_z + self.bias_hh[j]
            )
            r_val = GRUGene.sigmoid.function(
                x * self.weight_ih[self.hidden_size + j] + self.bias_ih[self.hidden_size + j] + recurrent_r +
                self.bias_hh[self.hidden_size + j]
            )

            recurrent_n = sum(
                (r_val * self.state[k]) * self.weight_hh[k][2 * self.hidden_size + j]
                for k in range(self.hidden_size)
            )

            n_val = GRUGene.tanh.function(
                x * self.weight_ih[2 * self.hidden_size + j] + self.bias_ih[2 * self.hidden_size + j] + recurrent_n +
                self.bias_hh[2 * self.hidden_size + j]
            )

            h_new = (1 - z_val) * self.state[j] + z_val * n_val
            new_state.append(h_new)

        self.state = new_state
        self.response = sum(self.state)

        return self.response


class SiTGRUGene(BaseGene):
    sigmoid = ActivationFunction('sigmoid')

    def __init__(self, gene_id, activation: ActivationFunction, aggregation: AggregationFunction, hidden_size,
                 weight_ih=None, bias_ih=None, weight_hh=None, bias_hh=None, state=None, response: float = 0.0):
        self.gene_id = gene_id

        self.activation = activation
        self.aggregation = aggregation

        self.response = response

        self.hidden_size = hidden_size

        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.state = state

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.weight_ih = [random.uniform(-stdv, stdv) for _ in range(2 * self.hidden_size)]
        self.weight_hh = [[random.uniform(-stdv, stdv) for _ in range(2 * self.hidden_size)] for _ in
                          range(self.hidden_size)]
        self.bias_ih = [random.uniform(-stdv, stdv) for _ in range(2 * self.hidden_size)]
        self.bias_hh = [random.uniform(-stdv, stdv) for _ in range(2 * self.hidden_size)]

        self.state = [0.0] * self.hidden_size

    def crossover(self, neuron2):
        if self.gene_id != neuron2.gene_id:
            raise ValueError("Cannot crossover neurons with different IDs")

        return SiTGRUGene(
            self.gene_id,
            self.activation,
            self.aggregation,
            self.hidden_size,
            deepcopy(self.weight_ih),
            deepcopy(self.bias_ih),
            deepcopy(self.weight_hh),
            deepcopy(self.bias_hh),
            deepcopy(self.state),
            self.response
        )

    def calculate_response(self, inputs):
        x = self.aggregation.function(inputs)
        new_state = []

        for j in range(self.hidden_size):
            recurrent_z = sum(self.state[k] * self.weight_hh[k][j] for k in range(self.hidden_size))

            z_val = SiTGRUGene.sigmoid.function(
                x * self.weight_ih[j] + self.bias_ih[j] + recurrent_z + self.bias_hh[j]
            )

            n_val = SiTGRUGene.sigmoid.function(
                x * self.weight_ih[self.hidden_size + j] + self.bias_ih[self.hidden_size + j] +
                self.bias_hh[self.hidden_size + j]
            )

            h_new = z_val * self.state[j] + (1 - z_val) * n_val
            new_state.append(h_new)

        self.state = new_state
        self.response = sum(self.state)

        return self.response


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