import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from neat.genes import LSTMGene, GRUGene
def creates_cycle(connections, test):
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    required = set(outputs)
    s = set(outputs)
    while 1:
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        c = set(b for (a, b) in connections if a in s and b not in s)
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers


def visualize_genome(genome, filename=None, show=True):
    G = nx.DiGraph()

    input_ids = genome.num_inputs
    output_ids = genome.num_outputs

    hidden_ids = [
        nid for nid in genome.neurons.keys()
        if nid not in input_ids and nid not in output_ids
    ]

    for nid, gene in genome.neurons.items():
        gene_Type = type(gene).__name__
        # print(f"Neuron {nid}: type={gene_Type}")

        if nid in input_ids:
            color = "skyblue"
        elif nid in output_ids:
            color = "salmon"
        elif gene_Type == "LSTMGene":
            color = "mediumpurple"
        elif gene_Type == "GRUGene":
            color = "orange"
        elif gene_Type == "SiTGRUGene":
            color = "lightcoral"
        else:
            color = "lightgreen"
        G.add_node(nid, color=color)



    edge_labels = {}
    edge_colors = []
    for link in genome.connections.values():
        in_id = link.input_id
        out_id = link.output_id
        weight = link.weight
        enabled = link.enabled
        inn_id = link.innovation_id
        G.add_edge(in_id, out_id, weight=weight, enabled=enabled)
        status = "1" if enabled else "0"
        edge_labels[(in_id, out_id)] = f"#{inn_id}\nw={weight:.2f} {status}"
        edge_colors.append("black" if enabled else "lightgray")

    pos = {}
    max_count = max(len(input_ids), len(output_ids), len(hidden_ids) or 1)

    def evenly_spaced(ids, x, total):
        n = len(ids)
        step = total / (n + 1)
        return {nid: (x, step * (i + 1)) for i, nid in enumerate(ids)}

    pos.update(evenly_spaced(input_ids, 0, max_count))
    pos.update(evenly_spaced(output_ids, 2, max_count))
    if hidden_ids:
        pos.update(evenly_spaced(hidden_ids, 1, max_count))

    node_colors = []
    for nid in G.nodes():
        node_colors.append(G.nodes[nid]['color'])

    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.1"
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
    )

    table_lines = ["Połączenia:"]
    for link in sorted(genome.connections.values(), key=lambda l: l.innovation_id):
        status = "włączone" if link.enabled else "wyłączone"
        table_lines.append(
            f"  Innow. #{link.innovation_id}: neuron {link.input_id} → {link.output_id}  |  waga={link.weight:.4f}  |  {status}"
        )
    table_text = "\n".join(table_lines)

    fig.text(
        0.01, 0.01, table_text,
        fontsize=7,
        verticalalignment="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8)
    )

    legend = [
        mpatches.Patch(color="skyblue", label="Wejście"),
        mpatches.Patch(color="salmon", label="Wyjście"),
        mpatches.Patch(color="lightgreen", label="Ukryty"),
        mpatches.Patch(color="lightgray", label="Wyłączone połączenie"),
    ]
    # ax.legend(handles=legend, loc="upper left")
    ax.set_title(f"Genom #{genome.genome_id}  |  fitness: {genome.fitness}  |  połączeń: {len(genome.connections)}  |  neuronów: {len(genome.neurons)}")
    ax.axis("off")

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if filename:
        plt.savefig(filename, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)