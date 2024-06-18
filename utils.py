from collections import defaultdict
import sqlite3
import yaml

from datasets import Dataset
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


def strip(text: str, bos_token: str, eos_token: str) -> str:
    return text.removeprefix(bos_token) + eos_token


def plot(graph: graph.SceneGraph, already_reduced: bool = False):
    """Plot a graph representation of the PDDL description.

    Args:
        graph (nx.MultiDiGraph): The graph to plot.
    """
    if not already_reduced:
        graph = oracle.reduce(graph, validate=False)
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        for node in nodes:
            graph.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(
        graph,
        align="horizontal",
        subset_key="layer",
        scale=-1,
    )

    fig = plt.figure()

    nx.draw(graph, pos=pos, ax=fig.gca(), with_labels=True)

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
    dataset: dict[str, dict[str, list]] = {split: defaultdict(list) for split in splits}

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
