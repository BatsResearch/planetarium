import jinja2 as jinja
from pddl.core import Action
import rustworkx as rx

from planetarium import graph

from .reduced_graph import ReducedSceneGraph, ReducedProblemGraph
from .oracles.blocksworld import (
    _reduce_blocksworld,
    _inflate_blocksworld,
    _fully_specify_blocksworld,
    _plan_blocksworld,
)
from .oracles.gripper import (
    _reduce_gripper,
    _inflate_gripper,
    _fully_specify_gripper,
    _plan_gripper,
)
from .oracles.rover import _reduce_rover, _inflate_rover, _fully_specify_rover
from .oracles.rover_single import _reduce_rover_single, _inflate_rover_single, _fully_specify_rover_single


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
    match domain:
        case "blocksworld":
            return _reduce_blocksworld(graph)
        case "gripper":
            return _reduce_gripper(graph)
        case "rover":
            return _reduce_rover(graph)
        case "rover-single":
            return _reduce_rover_single(graph)
        case _:
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
    match domain:
        case "blocksworld":
            return _inflate_blocksworld(scene)
        case "gripper":
            return _inflate_gripper(scene)
        case "rover":
            return _inflate_rover(scene)
        case "rover-single":
            return _inflate_rover_single(scene)
        case _:
            raise DomainNotSupportedError(f"Domain {domain} not supported.")


def fully_specify(
    problem: graph.ProblemGraph,
    domain: str | None = None,
    return_reduced: bool = False,
) -> graph.ProblemGraph:
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

    match domain:
        case "blocksworld":
            reduced_init, reduced_goal = _reduce_blocksworld(problem).decompose()
            fully_specified_goal = _fully_specify_blocksworld(reduced_goal)
        case "gripper":
            reduced_init, reduced_goal = _reduce_gripper(problem).decompose()
            fully_specified_goal = _fully_specify_gripper(
                reduced_init,
                reduced_goal,
            )
        case "rover":
            reduced_init, fully_specified_goal = _fully_specify_rover(
                *problem.decompose()
            )
        case "rover-single":
            reduced_init, fully_specified_goal = _fully_specify_rover_single(
                *problem.decompose()
            )
        case _:
            raise DomainNotSupportedError(f"Domain {domain} not supported.")

    if return_reduced:
        return ReducedProblemGraph.join(reduced_init, fully_specified_goal)
    else:
        init, _ = problem.decompose()
        return graph.ProblemGraph.join(
            init,
            inflate(fully_specified_goal, domain=domain),
        )


def plan(problem: graph.ProblemGraph, domain: str | None = None) -> list[Action]:
    """Plans a sequence of actions to solve a problem.

    Args:
        problem (graph.ProblemGraph): The problem to plan for.

    Returns:
        str: The sequence of actions to solve the problem.
    """
    domain = domain or problem.domain
    problem = fully_specify(problem, domain=domain, return_reduced=True)
    match domain:
        case "blocksworld":
            return _plan_blocksworld(problem)
        case "gripper":
            return _plan_gripper(problem)
        case _:
            raise DomainNotSupportedError(f"Domain {domain} not supported.")


def plan_to_string(actions: list[Action]) -> str:
    """Converts a list of actions to a string.

    Args:
        actions (list[Action]): The list of actions to convert.

    Returns:
        str: The string representation of the actions.
    """
    return plan_template.render(actions=actions)
