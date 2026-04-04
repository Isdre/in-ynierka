import copy
import random
from neat.genome import Genome
from neat.genes import NeuronGene, LinkGene
from neat.activation import ActivationFunction
from neat.aggregation import AggregationFunction


class Reproduction:
    def __init__(self, reporters=None):
        self.reporters = reporters
        self.genome_indexer = 0

    def _next_genome_id(self):
        self.genome_indexer += 1
        return self.genome_indexer

    def create_new(self, pop_size, input_keys, output_keys, innovation_tracker, initial_connections=None, config=None):
        population = {}
        for _ in range(pop_size):
            genome_id = self._next_genome_id()
            genome = Genome(genome_id, input_keys, output_keys)

            for i in input_keys:
                genome.neurons[i] = NeuronGene(
                    i,
                    bias=0.0,
                    activation=ActivationFunction('linear'),
                    aggregation=AggregationFunction('sum')
                )

            for i in output_keys:
                genome.neurons[i] = NeuronGene(
                    i,
                    bias=0.0,
                    activation=ActivationFunction('sigmoid'),
                    aggregation=AggregationFunction('sum')
                )

            if initial_connections is not None:
                connections = initial_connections
            else:
                connections = random.randint(1, len(input_keys) * len(output_keys))

            genome.initialize_random(innovation_tracker, connections, config)
            population[genome_id] = genome
        return population

    def reproduce(self, species_set, pop_size, innovation_tracker, config, stagnation=False):
        all_species = list(species_set.values())

        if not all_species:
            raise RuntimeError("Brak gatunków do reprodukcji.")

        for s in all_species:
            species_size = len(s.members)
            for genome in s.members.values():
                raw = genome.fitness if genome.fitness is not None else 0.0
                genome.adjusted_fitness = raw / species_size

        species_fitness_sums = [
            sum(g.adjusted_fitness for g in s.members.values())
            for s in all_species
        ]
        total_adjusted_fitness = sum(species_fitness_sums)

        if total_adjusted_fitness == 0:
            offspring_counts = [pop_size // len(all_species)] * len(all_species)
        else:
            offspring_counts = [
                max(1, int((s_sum / total_adjusted_fitness) * pop_size))
                for s_sum in species_fitness_sums
            ]

        while sum(offspring_counts) < pop_size:
            offspring_counts[random.randrange(len(offspring_counts))] += 1
        while sum(offspring_counts) > pop_size:
            idx = max(range(len(offspring_counts)), key=lambda i: offspring_counts[i])
            offspring_counts[idx] -= 1

        new_population = {}

        for s, count in zip(all_species, offspring_counts):
            if count <= 0:
                continue

            self.cull_species(s, config.survival_threshold)
            survivors = list(s.members.values())

            best_in_species = max(survivors, key=lambda g: g.fitness if g.fitness is not None else 0.0)
            elite_id = self._next_genome_id()
            elite = copy.deepcopy(best_in_species)
            elite.key = elite_id
            new_population[elite_id] = elite
            count -= 1

            crossover_prob = config.crossover_prob if not stagnation else config.stagnation_crossover_prob
            mutation_prob = config.mutation_prob if not stagnation else config.stagnation_mutation_prob

            while count > 0:
                child_id = self._next_genome_id()

                random_value = random.random()

                if len(survivors) >= 2 and random_value < crossover_prob:
                    parent1, parent2 = random.sample(survivors, 2)
                    p1_fit = parent1.fitness if parent1.fitness is not None else 0.0
                    p2_fit = parent2.fitness if parent2.fitness is not None else 0.0

                    if p1_fit < p2_fit:
                        parent1, parent2 = parent2, parent1

                    child = Genome.crossover(parent1, parent2)
                    child.genome_id = child_id
                elif random_value < crossover_prob + mutation_prob:
                    parent = random.choice(survivors)
                    child = copy.deepcopy(parent)
                    child.genome_id = child_id
                    child.fitness = None
                    child.mutate(config, innovation_tracker)
                else:
                    parent = random.choice(survivors)
                    child = copy.deepcopy(parent)
                    child.genome_id = child_id
                    child.fitness = None

                new_population[child_id] = child
                count -= 1

        return new_population

    def cull_species(self, species, survival_threshold):
        num_to_keep = max(1, int(len(species.members) * survival_threshold))
        sorted_members = sorted(
            species.members.items(),
            key=lambda x: x[1].fitness if x[1].fitness is not None else 0.0,
            reverse=True
        )
        species.members = dict(sorted_members[:num_to_keep])