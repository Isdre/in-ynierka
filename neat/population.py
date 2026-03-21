from neat.reporter import ReporterSet
from neat.reproduction import Reproduction
from neat.innovation import InnovationTracker
from neat.species import SpeciesSet
from neat.math_util import mean

class Population:
    def __init__(self, config, innovation_tracker):
        self.config = config
        self.generation = 0
        self.reporters = ReporterSet()
        self.reproduction = Reproduction(self.reporters)
        self.innovation_tracker = innovation_tracker
        self.species = {}
        self.genomes = self.reproduction.create_new(
            config.population_size,
            config.input_keys,
            config.output_keys,
            innovation_tracker
        )
        self.best_genome = None

        self.previous_best_fitness = None
        self.best_fitness = None

    def end_creterion(self):
        if self.config.fitness_threshold is not None and self.previous_best_fitness is not None:
            return abs(self.best_fitness-self.previous_best_fitness) < self.config.fitness_threshold
        return False

    def set_best_genome(self, genome):
        self.best_genome = genome
        self.previous_best_fitness = self.best_fitness
        self.best_fitness = genome.fitness

    def speciate(self):
        species_set = SpeciesSet()
        species_set.speciate(self.config, self.genomes, self.generation)
        self.species = species_set.species

    def get_best_genome(self):
        return max(self.genomes.values(), key=lambda g: g.fitness)

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def run(self, fitness_function, n=None):

        if n is None and (not hasattr(self.config, 'fitness_threshold') or self.config.fitness_threshold is None):
            raise RuntimeError("Must specify a number of generations to run or a fitness threshold.")

        self.genomes = self.reproduction.create_new(
            self.config.population_size,
            self.config.input_keys,
            self.config.output_keys,
            self.innovation_tracker
        )
        self.speciate()

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)
            fitness_function(self.genomes, self.config)

            best = None
            for i, g in self.genomes.items():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g

            self.reporters.post_evaluate(self.config, self.genomes, self.species, best)

            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.set_best_genome(best)

            if self.end_creterion():
                self.reporters.found_solution(self.config, self.generation, best)
                break

            # fv = self.fitness_criterion(g.fitness for g in self.genomes.values())
            # if fv >= self.config.fitness_threshold:
            #     self.reporters.found_solution(self.config, self.generation, best)
            #     break

            self.genomes = self.reproduction.reproduce(self.species, self.config.population_size, self.innovation_tracker, self.config)

            if not self.species:
                self.reporters.complete_extinction()

                if self.config.reset_on_extinction:
                    self.genomes = self.reproduction.create_new(
                        self.config.population_size,
                        self.config.input_keys,
                        self.config.output_keys,
                        self.innovation_tracker
                    )
                else:
                    raise CompleteExtinctionException()

            self.speciate()

            self.reporters.end_generation(self.config, self.genomes, self.species)

            self.generation += 1

        return self.best_genome
