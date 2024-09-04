from typing import Callable

import jinja2 as jinja
from pddl.core import Action

from planetarium import graph

from .reduced_graph import ReducedSceneGraph, ReducedProblemGraph
from .oracles import ORACLES


plan_template = jinja.Template(
    """
    {%- for action in actions -%}
    ({{ action.name }} {{ action.parameters | join(", ") }})
    {% endfor %}
    """
)


class DomainNotSupportedError(Exception):
    pass


def reduce(
    graph: graph.SceneGraph,
    domain: str | None = None,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a scene graph to a Directed Acyclic Graph.

    Args:
        graph (graph.SceneGraph): The scene graph to reduce.
        domain (str, optional): The domain of the scene graph.

    Returns:
        ReducedGraph: The reduced problem graph.
    """
    domain = domain or graph.domain
    if oracle := ORACLES.get(domain):
        return oracle.reduce(graph)
    raise DomainNotSupportedError(f"Domain {domain} not supported.")


def inflate(
    scene: ReducedSceneGraph | ReducedProblemGraph,
    domain: str | None = None,
) -> graph.SceneGraph:
    """Inflate a reduced scene graph to a SceneGraph.

    Args:
        scene (ReducedGraph): The reduced scene graph to respecify.
        domain (str | None, optional): The domain of the scene graph. Defaults
            to None.

    Returns:
        graph.SceneGraph: The respecified, inflated scene graph.
    """
    domain = domain or scene._domain
    if oracle := ORACLES.get(domain):
        return oracle.inflate(scene)
    raise DomainNotSupportedError(f"Domain {domain} not supported.")


def fully_specify(
    problem: graph.ProblemGraph,
    domain: str | None = None,
    return_reduced: bool = False,
) -> graph.ProblemGraph | ReducedProblemGraph:
    """Fully specifies a goal state.

    Args:
        problem (graph.ProblemGraph): The problem graph with the goal state to
            fully specify.
        domain (str | None, optional): The domain of the scene graph. Defaults
            to None.
        return_reduced (bool, optional): Whether to return the reduced scene
            graph. Defaults to False.

    Returns:
        graph.ProblemGraph: The fully specified problem graph.
    """
    domain = domain or problem.domain

    if oracle := ORACLES.get(domain):
        return oracle.fully_specify(problem, return_reduced=return_reduced)
    raise DomainNotSupportedError(f"Domain {domain} not supported.")


def plan(problem: graph.ProblemGraph, domain: str | None = None) -> list[Action]:
    """Plans a sequence of actions to solve a problem.

    Args:
        problem (graph.ProblemGraph): The problem to plan for.

    Returns:
        str: The sequence of actions to solve the problem.
    """
    domain = domain or problem.domain
    if oracle := ORACLES.get(domain):
        return oracle.plan(problem)
    raise DomainNotSupportedError(f"Domain {domain} not supported.")


def plan_to_string(actions: list[Action]) -> str:
    """Converts a list of actions to a string.

    Args:
        actions (list[Action]): The list of actions to convert.

    Returns:
        str: The string representation of the actions.
    """
    return plan_template.render(actions=actions)
