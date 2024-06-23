import functools
import rustworkx as rx
import typing

from planetarium import graph


def _preserves_mapping(
    source: graph.PlanGraphNode,
    target: graph.PlanGraphNode,
    mapping: dict,
) -> bool:
    """
    Check if a mapping is preserved between the nodes.

    Parameters:
        source (graph.PlanGraphNode): The source node.
        target (graph.PlanGraphNode): The target node.
        mapping (dict): The mapping between node names.

    Returns:
        bool: True if the mapping preserves names, False otherwise.
    """
    return (
        source.label == graph.Label.CONSTANT
        and target.label == graph.Label.CONSTANT
        and mapping[source.name] == target.name
    )


def _same_typing(source: graph.PlanGraphNode, target: graph.PlanGraphNode) -> bool:
    """
    Check if the typing of two nodes is the same.

    Parameters:
        source (graph.PlanGraphNode): The source node.
        target (graph.PlanGraphNode): The target node.

    Returns:
        bool: True if typings are the same, False otherwise.
    """
    return (
        source.label == graph.Label.CONSTANT
        and target.label == graph.Label.CONSTANT
        and source.typing == target.typing
    )


def _node_matching(
    source: graph.PlanGraphNode,
    target: graph.PlanGraphNode,
    mapping: typing.Optional[dict],
) -> bool:
    """
    Check if two nodes match based on their labels, positions, and typings.

    Parameters:
        source (graph.PlanGraphNode): The source node.
        target (graph.PlanGraphNode): The target node.
        mapping (Optional[dict]): The mapping between node names.

    Returns:
        bool: True if nodes match, False otherwise.
    """
    match (source.label, target.label):
        case (graph.Label.CONSTANT, graph.Label.CONSTANT):
            return _same_typing(source, target) and (
                _preserves_mapping(source, target, mapping) if mapping else True
            )
        case (graph.Label.PREDICATE, graph.Label.PREDICATE):
            # type of predicate should be the same as well
            return source.typing == target.typing
        case _:
            return False


def _edge_matching(
    source: graph.PlanGraphEdge,
    target: graph.PlanGraphEdge,
    attributes: dict[str, str | int | graph.Scene, graph.Label] = {},
) -> bool:
    """
    Check if two edges match based on their attributes.

    Parameters:
        source (graph.PlanGraphEdge): The source edge.
        target (graph.PlanGraphEdge): The target edge.
        attributes (dict): The attributes to match.

    Returns:
        bool: True if edges match, False otherwise.
    """

    def _getattr(obj, attr):
        v = getattr(obj, attr, attributes[attr])
        if v is None:
            v = attributes[attr]
        return v

    return all(_getattr(source, attr) == _getattr(target, attr) for attr in attributes)


def isomorphic(
    source: graph.ProblemGraph | graph.SceneGraph,
    target: graph.ProblemGraph | graph.SceneGraph,
    mapping: typing.Optional[dict] = None,
) -> bool:
    """
    Find all valid isomorphic mappings between nodes of two scene graphs.

    Parameters:
        source (ProblemGraph): The source problem graph.
        target (ProblemGraph): The target problem graph.
        mapping (Optional[dict]): The initial mapping between node names.

    Returns:
        bool: True if there is a valid mapping, False otherwise.
    """
    node_matching = functools.partial(_node_matching, mapping=mapping)
    edge_matching = functools.partial(
        _edge_matching,
        attributes={"position": -1, "predicate": "", "scene": None},
    )

    return rx.is_isomorphic(
        source.graph,
        target.graph,
        node_matcher=node_matching,
        edge_matcher=edge_matching,
    )


def equals(
    source: graph.ProblemGraph,
    target: graph.ProblemGraph,
    is_placeholder: bool = False,
) -> bool:
    """
    Check if there is a valid mapping between problem graphs.

    Parameters:
        source (ProblemGraph | SceneGraph): The initial problem graph.
        target (ProblemGraph | SceneGraph): The goal problem graph.
        is_placeholder (bool): If False, the function will compare the initial
            and goal scene graphs together. If True, the function will compare
            the two initial scene graphs and the two goal scene graphs
            separately.

    Returns:
        bool: True if there is a valid mapping, False otherwise.
    """
    if source == target:
        return True
    if not is_placeholder:
        return isomorphic(source, target)
    else:
        source_init, source_goal = source.decompose()
        target_init, target_goal = target.decompose()

        if source_init == target_init and source_goal == target_goal:
            return True

        valid_init = isomorphic(source_init, target_init)
        valid_goal = isomorphic(source_goal, target_goal)

        return valid_init and valid_goal
