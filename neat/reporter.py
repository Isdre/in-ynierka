class ReporterSet:
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, generation):
        for reporter in self.reporters:
            reporter.start_generation(generation)

    def end_generation(self, config, population, species):
        for reporter in self.reporters:
            reporter.end_generation(config, population, species)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def found_solution(self, config, generation, best):
        for reporter in self.reporters:
            reporter.found_solution(config, generation, best)

    def info(self, msg):
        for reporter in self.reporters:
            reporter.info(msg)

    def warning(self, msg):
        for reporter in self.reporters:
            reporter.warning(msg)


class BaseReporter:
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def found_solution(self, config, generation, best):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


class StdOutReporter(BaseReporter):
    def __init__(self, show_species_detail=False):
        self.show_species_detail = show_species_detail

    def start_generation(self, generation):
        print(f"\n*** Running generation {generation} ***")

    def end_generation(self, config, population, species):
        print(f"Population size: {len(population)}")
        if self.show_species_detail:
            print(f"Number of species: {len(species.keys())}")
            for sid, s in species.items():
                print(f"  Species {sid}: size={len(s.members)}, fitness={s.representative.fitness}")
                print(f"    Representative genome: {s.representative.genome_id}, members: {[m.genome_id for m in s.members.values()]}")

    def found_solution(self, config, generation, best):
        print(f"\nSolution found at generation {generation}")
        print(f"Best genome fitness: {best.fitness:.4f}")

    def info(self, msg):
        print(f"[INFO] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")