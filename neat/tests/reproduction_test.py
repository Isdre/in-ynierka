import pytest
from unittest.mock import Mock, MagicMock, patch
from neat.reproduction import Reproduction
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
from neat.innovation import InnovationTracker
from neat.reporter import ReporterSet
from neat import Config
from neat.species import SpeciesSet, Species

class TestReproduction:
    def _make_innovation_tracker(self):
        return InnovationTracker(2, 1)

    def _make_reporters(self):
        return ReporterSet()

    def _make_config(self):
        config = Config()
        config.num_inputs = 2
        config.num_outputs = 1
        config.input_keys = [1,2]
        config.output_keys = [3]
        config.population_size = 10
        config.survival_threshold = 0.2
        config.crossover_prob = 0.75
        config.crossover_rate = 0.75
        config.add_link_prob = 0.05
        config.add_neuron_prob = 0.03
        config.remove_link_prob = 0.01

        config.c1 = 1.0
        config.c2 = 1.0
        config.c3 = 0.4
        config.compatibility_threshold = 3.0
        config.compatibility_change = 0.1
        config.stagnation_limit = 15
        return config


    def _make_genome(self, genome_id, fitness=1.0):
        genome = Genome(genome_id, [1,2], [3])
        genome.neurons[1] = NeuronGene(1, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[2] = NeuronGene(2, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.neurons[3] = NeuronGene(3, 0.0, ActivationFunction('linear'), AggregationFunction('sum'))
        genome.connections[(1, 3)] = LinkGene(1, 1, 3, 0.5, False, True)
        genome.connections[(2, 3)] = LinkGene(2, 2, 3, 0.5, False, True)
        genome.fitness = fitness
        return genome

    def _make_reproduction(self):
        reporters = self._make_reporters()
        return Reproduction(reporters)

    def test_reproduction_creation(self):
        reproduction = self._make_reproduction()
        assert reproduction is not None

    def test_create_new_returns_correct_size(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()
        pop_size = 10

        with patch.object(Genome, 'initialize_random', return_value=None):
            population = reproduction.create_new(pop_size, [1,2], [3], tracker)

        assert len(population) == pop_size

    def test_create_new_returns_genomes(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()

        with patch.object(Genome, 'initialize_random', return_value=None):
            population = reproduction.create_new(5, [1,2], [3], tracker)

        for genome in population.values():
            assert isinstance(genome, Genome)

    def test_create_new_unique_ids(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()

        with patch.object(Genome, 'initialize_random', return_value=None):
            population = reproduction.create_new(10, [1,2], [3], tracker)

        ids = list(population.keys())
        assert len(ids) == len(set(ids))

    def test_create_new_correct_inputs_outputs(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()

        with patch.object(Genome, 'initialize_random', return_value=None):
            population = reproduction.create_new(5, [1,2], [3], tracker)

        for genome in population.values():
            assert genome.num_inputs == [1,2]
            assert genome.num_outputs == [3]

    def test_reproduce_returns_correct_size(self):
        reproduction = self._make_reproduction()
        config = self._make_config()
        tracker = self._make_innovation_tracker()
        pop_size = 10

        genomes = {i: self._make_genome(i, fitness=float(i)) for i in range(1, pop_size + 1)}
        species_set = SpeciesSet()
        species_set.speciate(config, genomes, 1)

        with patch.object(Genome, 'mutate', return_value=None):
            new_population = reproduction.reproduce(species_set.species, pop_size, tracker, config)

        assert len(new_population) == pop_size

    def test_reproduce_returns_genomes(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()
        pop_size = 5

        config = self._make_config()

        genomes = {i: self._make_genome(i, fitness=float(i)) for i in range(1, pop_size + 1)}
        species_set = SpeciesSet()
        species_set.speciate(config, genomes, 1)

        with patch.object(Genome, 'mutate', return_value=None):
            new_population = reproduction.reproduce(species_set.species, pop_size, tracker, config)

        for genome in new_population.values():
            assert isinstance(genome, Genome)

    def test_reproduce_unique_ids(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()
        pop_size = 10

        genomes = {i: self._make_genome(i, fitness=float(i)) for i in range(1, pop_size + 1)}
        config = self._make_config()

        species_set = SpeciesSet()
        species_set.speciate(config, genomes, 1)

        with patch.object(Genome, 'mutate', return_value=None):
            new_population = reproduction.reproduce(species_set.species, pop_size, tracker, config)

        ids = list(new_population.keys())
        assert len(ids) == len(set(ids))

    def test_reproduce_with_multiple_species(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()
        pop_size = 10

        genomes1 = {i: self._make_genome(i, fitness=float(i)) for i in range(1, 6)}
        genomes2 = {i: self._make_genome(i, fitness=float(i)) for i in range(6, 11)}

        # self.species_id = species_id
        # self.representative = representative
        # self.members = {}
        # self.fitness = 0.0
        # self.adjusted_fitness = 0.0

        species1 = Species(1, genomes1[1])
        species1.members = genomes1
        species1.fitness = species1.representative.fitness

        species2 = Species(2, genomes2[6])
        species2.members = genomes2
        species2.fitness = species2.representative.fitness

        species_set = SpeciesSet()
        species_set.species = {
            1: species1,
            2: species2
        }

        config = self._make_config()

        with patch.object(Genome, 'mutate', return_value=None):
            new_population = reproduction.reproduce(species_set.species, pop_size, tracker, config)

        assert len(new_population) == pop_size

    def test_reproduce_elitism_preserves_best(self):
        reproduction = self._make_reproduction()
        tracker = self._make_innovation_tracker()
        pop_size = 10

        genomes = {i: self._make_genome(i, fitness=float(i)) for i in range(1, pop_size + 1)}
        best_genome = max(genomes.values(), key=lambda g: g.fitness)
        species_set = SpeciesSet()
        species_set.speciate(self._make_config(), genomes, 1)

        config = self._make_config()

        with patch.object(Genome, 'mutate', return_value=None):
            new_population = reproduction.reproduce(species_set.species, pop_size, tracker, config)

        fitnesses = [g.fitness for g in new_population.values()]
        assert best_genome.fitness in fitnesses