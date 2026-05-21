
import os
import pickle

import neat
from neat import visualize

result_path = "results/example6"

for i in range(20):
    with open(result_path+f'/winner-recurrent-run{i}.pickle', 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)

    neat.graphs.visualize_genome(c, filename=result_path+f'/recurrent-winner-run{i}', show=False)
    # visualize.draw_net(config, c, view=False, filename=result_path+f'/feedforward-winner-run{i}', show_disabled=False)