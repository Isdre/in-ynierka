from neat.population import Population
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.config import Config
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction
from neat.innovation import InnovationTracker
from neat.reporter import ReporterSet

class TestPopulation:
    def _make_config(self):
        config = Config()
        config.input_keys = [1,2]
        config.output_keys = [3]
        config.population_size = 10
        config.num_inputs = 2
        config.num_outputs = 1
        config.compatibility_threshold = 3.0
        config.compatibility_change = 0.5
        config.target_species = 5
        config.add_link_prob = 0.05
        config.add_neuron_prob = 0.03
        config.remove_link_prob = 0.01
        config.add_neuron_gene_prob = 1
        config.c1 = 1.0
        config.c2 = 1.0
        config.c3 = 0.4
        config.survival_threshold = 0.2
        config.crossover_prob = 0.75
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

    def test_population_creation(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        assert population is not None

    def test_population_size(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        assert len(population.genomes) == config.population_size

    def test_population_genomes_have_correct_io(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        for genome in population.genomes.values():
            assert genome.num_inputs == config.input_keys
            assert genome.num_outputs == config.output_keys

    def test_population_unique_genome_ids(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        ids = list(population.genomes.keys())
        assert len(ids) == len(set(ids))

    def test_speciate_creates_species(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        for genome in population.genomes.values():
            genome.fitness = 1.0

        population.speciate()

        assert len(population.species) > 0

    def test_speciate_all_genomes_assigned(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        for genome in population.genomes.values():
            genome.fitness = 1.0

        population.speciate()

        assigned = sum(len(s.members.values()) for s in population.species.values())
        assert assigned == len(population.genomes)

    def test_evolve_multiple_generations(self):
        config = self._make_config()
        innovation_tracker = InnovationTracker(2, 1)
        population = Population(config, innovation_tracker)

        def fitness_function(genome, config):
            genome.fitness = 1.0

        def eval_genomes(genomes, config):
            for genome in genomes.values():
                fitness_function(genome, config)

        population.run(eval_genomes, n=5)

        assert population.generation == 5
        assert len(population.genomes) == config.population_size