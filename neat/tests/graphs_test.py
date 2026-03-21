import pytest
from unittest.mock import patch
from neat.graphs import creates_cycle, required_for_output, feed_forward_layers, visualize_genome
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
from neat.innovation import InnovationTracker
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

class TestCreatesCycle:
    def test_self_loop(self):
        assert creates_cycle([], (1, 1)) is True

    def test_no_cycle(self):
        connections = [(1, 2), (2, 3)]
        assert creates_cycle(connections, (1, 3)) is False

    def test_creates_cycle(self):
        connections = [(1, 2), (2, 3)]
        assert creates_cycle(connections, (3, 1)) is True

    def test_empty_connections_no_cycle(self):
        assert creates_cycle([], (1, 2)) is False

    def test_direct_reverse(self):
        connections = [(1, 2)]
        assert creates_cycle(connections, (2, 1)) is True


class TestRequiredForOutput:
    def test_direct_input_to_output(self):
        inputs = [1, 2]
        outputs = [3]
        connections = [(1, 3), (2, 3)]
        required = required_for_output(inputs, outputs, connections)
        assert 3 in required

    def test_hidden_neuron_required(self):
        inputs = [1]
        outputs = [3]
        connections = [(1, 2), (2, 3)]
        required = required_for_output(inputs, outputs, connections)
        assert 2 in required
        assert 3 in required

    def test_unused_neuron_not_required(self):
        inputs = [1, 2]
        outputs = [4]
        connections = [(1, 3), (3, 4)]
        required = required_for_output(inputs, outputs, connections)
        assert 4 in required
        assert 3 in required
        assert 2 not in required

    def test_no_connections(self):
        inputs = [1]
        outputs = [2]
        connections = []
        required = required_for_output(inputs, outputs, connections)
        assert 2 in required


class TestFeedForwardLayers:
    def test_single_layer(self):
        inputs = [1, 2]
        outputs = [3]
        connections = [(1, 3), (2, 3)]
        layers = feed_forward_layers(inputs, outputs, connections)
        assert len(layers) == 1
        assert 3 in layers[0]

    def test_two_layers(self):
        inputs = [1]
        outputs = [3]
        connections = [(1, 2), (2, 3)]
        layers = feed_forward_layers(inputs, outputs, connections)
        assert len(layers) == 2
        assert 2 in layers[0]
        assert 3 in layers[1]

    def test_no_connections(self):
        inputs = [1]
        outputs = [2]
        connections = []
        layers = feed_forward_layers(inputs, outputs, connections)
        assert layers == []

    def test_multiple_inputs_hidden_output(self):
        inputs = [1, 2]
        outputs = [4]
        connections = [(1, 3), (2, 3), (3, 4)]
        layers = feed_forward_layers(inputs, outputs, connections)
        assert 3 in layers[0]
        assert 4 in layers[1]


class TestVisualizeGenome:
    def _make_genome(self, num_inputs=[1,2], num_outputs=[3]):
        genome = Genome(1, num_inputs, num_outputs)
        genome.fitness = 1.0

        for i in num_inputs:
            genome.neurons[i] = NeuronGene(
                i,
                bias=0.0,
                activation=ActivationFunction('linear'),
                aggregation=AggregationFunction('sum')
            )

        for i in num_inputs:
            genome.neurons[i] = NeuronGene(
                i,
                bias=0.0,
                activation=ActivationFunction('sigmoid'),
                aggregation=AggregationFunction('sum')
            )

        tracker = InnovationTracker(num_inputs, num_outputs)
        inn_id = tracker.get_new_link_innovation_id(1, num_outputs[0])
        genome.connections[inn_id] = LinkGene(
            inn_id, 1, num_outputs[0], weight=0.5, enabled=True
        )

        return genome

    @patch("matplotlib.pyplot.show")
    def test_visualize_no_error(self, mock_show):
        genome = self._make_genome()
        visualize_genome(genome, show=False)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_visualize_saves_file(self, mock_show, mock_savefig):
        genome = self._make_genome()
        visualize_genome(genome, filename="test_output.png", show=False)
        mock_savefig.assert_called_once_with("test_output.png", bbox_inches="tight")

    @patch("matplotlib.pyplot.show")
    def test_visualize_with_hidden_neuron(self, mock_show):
        genome = self._make_genome(num_inputs=[1,2], num_outputs=[3])

        hidden_id = 10
        genome.neurons[hidden_id] = NeuronGene(
            hidden_id,
            bias=0.1,
            activation=ActivationFunction('sigmoid'),
            aggregation=AggregationFunction('sum')
        )

        tracker = InnovationTracker(2, 1)
        inn_id = tracker.get_new_link_innovation_id(1, hidden_id)
        genome.connections[inn_id] = LinkGene(
            inn_id, 1, hidden_id, weight=-0.3, enabled=False
        )

        visualize_genome(genome, show=False)

    @patch("matplotlib.pyplot.show")
    def test_visualize_empty_connections(self, mock_show):

        genome = self._make_genome()
        genome.connections = {}
        visualize_genome(genome, show=False)

    @patch("matplotlib.pyplot.show")
    def test_visualize_multiple_outputs(self, mock_show):
        genome = self._make_genome(num_inputs=[1,2,3], num_outputs=[4,5])
        visualize_genome(genome, show=False)

    @patch("matplotlib.pyplot.show")
    def test_visualize_disabled_connection(self, mock_show):
        genome = self._make_genome()
        for link in genome.connections.values():
            link.enabled = False
        visualize_genome(genome, show=False)

    @pytest.mark.visual
    def test_visualize_show_window(self):
        matplotlib.use('Agg')
        genome = self._make_genome(num_inputs=[1,2], num_outputs=[3])

        tracker = InnovationTracker(2, 1)

        genome.mutate_add_link(tracker)
        genome.mutate_add_neuron(tracker)
        genome.mutate_add_link(tracker)

        output_path = "visual_genome_test.png"
        visualize_genome(genome, filename=output_path, show=False)

        assert os.path.exists(output_path), "Plik wizualizacji nie został utworzony"
        #os.startfile(os.path.abspath(output_path))