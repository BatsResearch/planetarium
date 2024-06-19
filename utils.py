from collections import defaultdict
import sqlite3
import yaml

from datasets import Dataset
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx

from planetarium import graph, oracle
import llm_planner as llmp


def apply_template(
    problem: llmp.PlanningProblem,
    domain_prompt: str = "",
    problem_prompt: str = "",
    include_answer: bool = True,
) -> list[dict[str, str]]:
    """Apply problem template to the problem.

    Args:
        problem(llmp.PlanningProblem): The problem to apply the template to.
        domain_prompt (str, optional): How to prompt the domain. Defaults to "".
        problem_prompt (str, optional): How to prompt the problem. Defaults to "".
        include_answer (bool, optional): Whether to include the answer. Defaults to True.

    Returns:
        list[dict[str, str]]: Problem prompt.
    """
    return [
        {
            "role": "user",
            "content": f"{problem_prompt} {problem.natural_language} "
            + f"{domain_prompt}\n{problem.domain}\n",
        },
    ] + (
        [
            {
                "role": "assistant",
                "content": " " + problem.problem,
            },
        ]
        if include_answer
        else []
    )


def strip(text: str, bos_token: str, eos_token: str) -> str:
    return text.removeprefix(bos_token) + eos_token


def _layout(G: graph.PlanGraph, scale: float = 1.0):
    """Position nodes in layers of straight lines.

    Source: https://github.com/networkx/networkx/blob/main/networkx/drawing/layout.py

    Args:
        G (rx.PyDiGraph): A directed graph.
        scale (float, optional): Scale factor for positions. Defaults to 1.

    Returns:
        dict: A dictionary of positions keyed by node.

    """

    center = np.zeros(2)
    if len(G.nodes) == 0:
        return {}

    layers = rx.topological_generations(G.graph)

    pos = None
    nodes = []
    width = len(layers)
    for i, layer in enumerate(layers):
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset
        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)

    # Rescale
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()
    if lim > 0:
        pos *= scale / lim
    pos += center
    # horizontal
    pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos


def _draw(
    G: graph.PlanGraph,
    pos: dict,
    ax: plt.Axes,
    node_size: int = 300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
    font_size: int = 12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    clip_on=True,
):
    """Draw the graph G with Matplotlib.

    Source: Source: https://github.com/networkx/networkx/blob/main/networkx/drawing/nx_pylab.py
    """
    xy = np.asarray([pos[v] for v in G.graph.node_indices()])
    nodes_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
    )
    nodes_collection.set_zorder(2)  # nodes go on top of edges
    # Add node labels:
    labels = [node.node for node in G.nodes]
    for n, label in enumerate(labels):
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=clip_on,
        )

    # plot edges
    edge_pos = np.asarray([(pos[u], pos[v]) for u, v, _ in G.graph.edge_index_map().values()])

    edge_collection = matplotlib.collections.LineCollection(
        edge_pos,
        colors="k",
        linewidths=1.,
        antialiaseds=(1,),
        linestyle="solid",
        alpha=alpha,
    )

    edge_collection.set_zorder(1)  # edges go behind nodes
    ax.add_collection(edge_collection)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_axis_off()


def plot(graph: graph.PlanGraph, reduce: bool = False):
    """Plot a graph representation of the PDDL description.

    Args:
        graph (graph.PlanGraph): The graph to plot.
        already_reduced (bool, optional): Whether the graph is already reduced.
            Defaults to False.
    """
    if reduce:
        graph = oracle.reduce(graph, validate=False)
    # TODO: rx has no multipartite layout
    pos = _layout(graph, scale=-1)

    fig = plt.figure()
    _draw(graph, pos, ax=fig.gca())

    return fig


def load_dataset(config: dict) -> dict[str, Dataset]:
    """Load the dataset from the configuration.

    Args:
        config (dict): The dataset configuration.

    Returns:
        dict[str, Dataset]: The loaded dataset.
    """
    with open(config["splits_path"], "r") as f:
        split_ids_cfg = yaml.safe_load(f)

    splits: set[str] = config.get("splits", {}).keys()
    dataset = {split: defaultdict(list) for split in splits}

    # Connect to database
    conn = sqlite3.connect(config["database_path"])
    c = conn.cursor()

    # load domains
    domains = {}
    c.execute("SELECT name, domain_pddl FROM domains")
    for domain_name, domain_pddl in c.fetchall():
        domains[domain_name] = domain_pddl

    # load problems
    for split in splits:
        queries = []
        split_keys: list[str] = config["splits"][split]
        for split_key in split_keys:
            split_ids = split_ids_cfg
            for key in split_key:
                split_ids = split_ids[key]

            c.execute(
                f"SELECT domain, problem_pddl, natural_language FROM problems WHERE id in ({', '.join(['?'] * len(split_ids))})",
                split_ids,
            )
            queries.extend(c.fetchall())

        for domain, problem_pddl, natural_language in queries:
            dataset[split]["domain"].append(domains[domain])
            dataset[split]["problem"].append(problem_pddl)
            dataset[split]["natural_language"].append(natural_language)

    return {s: Dataset.from_dict(d, split=s) for s, d in dataset.items()}
