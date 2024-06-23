import matplotlib.pyplot as plt
import networkx as nx

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


def plot(graph: graph.PlanGraph, reduce: bool = False):
    """Plot a graph representation of the PDDL description.

    Args:
        graph (graph.PlanGraph): The graph to plot.
        already_reduced (bool, optional): Whether the graph is already reduced.
            Defaults to False.
    """
    if reduce:
        graph = oracle.reduce(graph, validate=False)
    # rx has no plotting functionality

    nx_graph = nx.MultiDiGraph()
    nx_graph.add_edges_from([(u.node, v.node, {"data":edge}) for u, v, edge in graph.edges])

    for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
        for node in nodes:
            nx_graph.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(
        nx_graph,
        align="horizontal",
        subset_key="layer",
        scale=-1,
    )

    fig = plt.figure()
    nx.draw(nx_graph, pos=pos, ax=fig.gca(), with_labels=True)

    return fig
