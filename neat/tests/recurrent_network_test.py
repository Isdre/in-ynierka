from neat.network import RecurrentNetwork
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.config import Config
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction

import pytest


class TestRecurrentNetwork:
    def test_recurrent_simple_flow(self):
        """Test prostego przepływu: In (1) -> Out (2)"""
        genome = Genome(1, 0, 1)
        # Neuron wejściowy i wyjściowy (liniowe, suma, brak biasu)
        genome.neurons[1] = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[2] = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))

        # Połączenie o wadze 2.0
        genome.connections[101] = LinkGene(101, 1, 2, 2.0, False, True)

        config = Config()
        config.input_keys = [1]
        config.output_keys = [2]

        network = RecurrentNetwork.create(genome, config)

        # Pierwsza aktywacja: 0.5 * 2.0 = 1.0
        output = network.activate([0.5])
        assert output == [1.0]

    def test_recurrent_self_loop(self):
        """Test pętli zwrotnej (rekurencji): Neuron (2) łączy się sam ze sobą"""
        genome = Genome(1, 0, 1)
        genome.neurons[1] = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[2] = NeuronGene(2, 0.5, ActivationFunction('linear'), AggregationFunction('sum'))  # Bias 0.5

        # Wejście -> Neuron 2 (waga 1.0)
        genome.connections[101] = LinkGene(101, 1, 2, 1.0, False, True)
        # Neuron 2 -> Neuron 2 (waga 0.5) - Rekurencja!
        genome.connections[102] = LinkGene(102, 2, 2, 0.5, True, True)

        config = Config()
        config.input_keys = [1]
        config.output_keys = [2]

        network = RecurrentNetwork.create(genome, config)

        # KROK 1:
        # Wejście = 1.0. Wartość rekurencyjna z t-1 = 0.0.
        # Wynik = activation(bias + (in * w1) + (prev_out * w2))
        # Wynik = (0.5 + (1.0 * 1.0) + (0.0 * 0.5)) = 1.5
        out1 = network.activate([1.0])
        assert out1 == [1.5]

        # KROK 2:
        # Wejście = 1.0. Wartość rekurencyjna z t-1 (czyli z kroku 1) = 1.5.
        # Wynik = (0.5 + (1.0 * 1.0) + (1.5 * 0.5)) = 0.5 + 1.0 + 0.75 = 2.25
        out2 = network.activate([1.0])
        assert out2 == [2.25]

    def test_recurrent_reset(self):
        """Test sprawdzający, czy reset() poprawnie czyści pamięć sieci"""
        genome = Genome(1, 0, 1)
        genome.neurons[1] = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[2] = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))

        # Neuron 2 zależy od siebie samego
        genome.connections[101] = LinkGene(101, 2, 2, 1.0, True, True)

        config = Config()
        config.input_keys = [1]
        config.output_keys = [2]
        network = RecurrentNetwork.create(genome, config)

        # Aktywujemy, żeby "nabić" stan wewnętrzny
        network.activate([1.0])
        network.reset()

        # Po resecie sieć powinna zachowywać się jak nowa
        # Gdyby nie było resetu, wynik byłby różny od 0.0 (przez rekurencję)
        # Tutaj, ponieważ nie ma wejścia z zewnątrz, powinien być czysty 0.0
        output = network.activate([0.0])
        assert output == [0.0]

    def test_mismatched_input_size(self):
        """Test rzucania błędu przy złej liczbie danych wejściowych"""
        genome = Genome(2, 0, 1)
        config = Config()
        config.input_keys = [1, 2]
        config.output_keys = [3]

        # Tworzymy pustą sieć (tylko dla testu struktury wejść)
        network = RecurrentNetwork(genome, config.input_keys, config.output_keys, [])

        with pytest.raises(RuntimeError):
            network.activate([0.5])  # Oczekuje 2, podano 1