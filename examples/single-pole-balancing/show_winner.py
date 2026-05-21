
import os
import pickle

import neat
from neat import visualize

result_path = "results/example3"

for i in range(20):
    with open(result_path+f'/winner-recurrent-run{i}.pickle', 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, result_path + "/config_recurrent.txt")
    config = neat.Config.read_from_file(config_path)

    neat.graphs.visualize_genome(c, filename=result_path+f'/recurrent-winner-run{i}', show=False)
    # visualize.draw_net(config, c, view=False, filename=result_path+f'/feedforward-winner-run{i}', show_disabled=False)