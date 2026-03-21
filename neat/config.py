from neat.genome import Genome

class Config:

    def __init__(self):
        self.num_inputs = None
        self.input_keys = None
        self.num_outputs = None
        self.output_keys = None
        self.connections = None
        self.compatibility_threshold = None
        self.compatibility_change = None
        self.stagnation_limit = None
        self.survival_threshold = None

        self.generation = None

        self.population_size = None

        self.fitness_threshold = None

        self.c1 = None
        self.c2 = None
        self.c3 = None

        self.crossover_prob = None

        self.add_link_prob = None
        self.add_neuron_prob = None
        self.remove_link_prob = None

    @staticmethod
    def parse_value(value):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return value

    @staticmethod
    def read_from_file(filename):
        config = Config()
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if hasattr(config, key):
                    setattr(config, key, Config.parse_value(value))

        config.input_keys = [i for i in range(1, config.num_inputs + 1)]
        config.output_keys = [i for i in range(config.num_inputs + 1, config.num_inputs + config.num_outputs + 1)]

        return config

    @staticmethod
    def write_to_file(config, filename):
        with open(filename, 'w') as f:
            for key in vars(config):
                value = getattr(config, key)
                f.write(f"{key} = {value}\n")