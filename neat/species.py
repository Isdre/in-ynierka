import random

class Species:
    def __init__(self, species_id, representative):
        self.species_id = species_id
        self.representative = representative
        self.members = {}

        self.max_fitness_ever = float('-inf')
        self.stagnation_counter = 0
        self.age = 0


class SpeciesSet:
    def __init__(self):
        self.species = {}
        self._species_indexer = 0

    def _next_species_id(self):
        self._species_indexer += 1
        return self._species_indexer

    def speciate(self, config, genomes, generation):
        global_best_fitness = max((s.max_fitness_ever for s in self.species.values()), default=0.0)

        for s in self.species.values():
            if not s.members:
                continue

            current_best = max((g.fitness if g.fitness is not None else 0.0) for g in s.members.values())

            if current_best > s.max_fitness_ever:
                s.max_fitness_ever = current_best
                s.stagnation_counter = 0
            else:
                s.stagnation_counter += 1
            s.age += 1

        surviving_species = {}
        for sid, s in self.species.items():
            is_stagnant = s.stagnation_counter >= config.stagnation_limit
            is_young = s.age < config.min_species_age
            is_best = s.max_fitness_ever >= global_best_fitness

            if not is_stagnant or is_young or is_best:
                surviving_species[sid] = s

        self.species = surviving_species

        for s in self.species.values():
            s.members = {}

        for genome_id, genome in genomes.items():
            placed = False
            for s in self.species.values():
                distance = genome.calculate_genetic_distance(s.representative, config)
                if distance < config.compatibility_threshold:
                    s.members[genome_id] = genome
                    placed = True
                    break

            if not placed:
                species_id = self._next_species_id()
                new_species = Species(species_id, genome)
                new_species.members[genome_id] = genome
                self.species[species_id] = new_species

        self.species = {sid: s for sid, s in self.species.items() if s.members}

        for s in self.species.values():
            s.representative = random.choice(list(s.members.values()))



    def adjust_compatibility_threshold(self, config):
        target_species = config.target_species
        if len(self.species) > target_species:
            config.compatibility_threshold += config.compatibility_change
        elif len(self.species) < target_species:
            config.compatibility_threshold -= config.compatibility_change