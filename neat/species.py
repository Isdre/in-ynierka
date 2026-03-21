class Species:
    def __init__(self, species_id, representative):
        self.species_id = species_id
        self.representative = representative
        self.members = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0


class SpeciesSet:
    def __init__(self):
        self.species = {}
        self._species_indexer = 0

    def _next_species_id(self):
        self._species_indexer += 1
        return self._species_indexer

    def speciate(self, config, genomes, generation):
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
            s.representative = max(s.members.values(), key=lambda g: g.fitness if g.fitness is not None else 0.0)

    def adjust_compatibility_threshold(self, config):
        target_species = config.target_species
        if len(self.species) > target_species:
            config.compatibility_threshold += config.compatibility_change
        elif len(self.species) < target_species:
            config.compatibility_threshold -= config.compatibility_change