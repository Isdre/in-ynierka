from neat.individual import Individual
from neat.genome import Genome

class TestIndividual:
    def test_create_individual(self):
        idv = Individual(Genome(0, 3, 1), 0)