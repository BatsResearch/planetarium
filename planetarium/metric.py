from typing import Any, Callable

import functools
import networkx as nx
import time

from planetarium.graph import Label, SceneGraph, ProblemGraph

Node = dict[str, Any]


def _preserves_mapping(source: Node, target: Node, mapping: dict) -> bool:
    """
    Check if a mapping is preserved between the nodes.

    Parameters:
        source (Node): The source node.
        target (Node): The target node.
        mapping (dict): The mapping between node names.

    Returns:
        bool: True if the mapping preserves names, False otherwise.
    """
    return (
        source["label"] == Label.CONSTANT
        and target["label"] == Label.CONSTANT
        and mapping[source["name"]] == target["name"]
    )


def _same_typing(source: Node, target: Node) -> bool:
    """
    Check if the typing of two nodes is the same.

    Parameters:
        source (Node): The source node.
        target (Node): The target node.

    Returns:
        bool: True if typings are the same, False otherwise.
    """
    return (
        source["label"] == Label.CONSTANT
        and target["label"] == Label.CONSTANT
        and source["typing"] == target["typing"]
    )


def _matching(source: Node, target: Node, mapping: dict | None) -> bool:
    """
    Check if two nodes match based on their labels, positions, and typings.

    Parameters:
        source (Node): The source node.
        target (Node): The target node.
        mapping (dict | None): The mapping between node names.

    Returns:
        bool: True if nodes match, False otherwise.
    """
    match (source["label"], target["label"]):
        case (Label.CONSTANT, Label.CONSTANT):
            return _same_typing(source, target) and (
                _preserves_mapping(source, target, mapping) if mapping else True
            )
        case (Label.PREDICATE, Label.PREDICATE):
            # type of predicate should be the same as well
            return source["typing"] == target["typing"]
        case _:
            return False


def _map(
    source: SceneGraph,
    target: SceneGraph,
    mapping: dict | None = None,
) -> list[dict]:
    """
    Find all valid isomorphic mappings between nodes of two scene graphs.

    Parameters:
        source (SceneGraph): The source scene graph.
        target (SceneGraph): The target scene graph.
        mapping (dict | None): The initial mapping between node names.

    Returns:
        list: A list of dictionaries representing valid mappings.
    """
    if not nx.faster_could_be_isomorphic(source, target):
        return []

    matching = functools.partial(_matching, mapping=mapping)
    mapper = nx.isomorphism.MultiDiGraphMatcher(
        source,
        target,
        node_match=matching,
        edge_match=nx.isomorphism.categorical_edge_match(
            ["position", "predicate"],
            default=[-1, ""],
        ),
    )

    if mapper.is_isomorphic():
        mapper.initialize()
        return mapper.match()
    else:
        return []


def _distance(
    source: SceneGraph, target: SceneGraph, mapping: dict | None = None
) -> int:
    """
    Calculate the graph edit distance between two scene graphs.

    Parameters:
        source (SceneGraph): The source scene graph.
        target (SceneGraph): The target scene graph.
        mapping (dict | None): The initial mapping between node names.

    Returns:
        int: The graph edit distance.
    """
    matching = functools.partial(_matching, mapping=mapping)

    return nx.graph_edit_distance(
        source,
        target,
        node_match=matching,
        edge_match=nx.isomorphism.categorical_edge_match(
            ["position", "predicate"],
            default=[-1, ""],
        ),
    )


def _minimal_mappings(
    source: SceneGraph,
    target: SceneGraph,
    timeout: float | None = None,
) -> tuple[list, float, bool]:
    """
    Calculate the graph edit distance between two scene graphs.

    Parameters:
        source (SceneGraph): The source scene graph.
        target (SceneGraph): The target scene graph.
        max_attempts (int): The maximum number of edit path iterations to
            consider.

    Returns:
        tuple:
            - list: A list of list of tuples representing valid mappings.
            - float: The graph edit distance.
            - bool: True if the timeout was reached, False otherwise.
    """

    start_time = time.perf_counter()

    def timed_out() -> bool:
        return bool(timeout and time.perf_counter() - start_time > timeout)

    # try isomorphism first:
    iso_mappings = _map(source, target)
    if iso_mappings:
        return [[(k, v) for k, v in m.items()] for m in iso_mappings], 0.0, False

    # if it is not isomorphic, try edit distance:
    edit_path_gen = nx.similarity.optimize_edit_paths(
        source,
        target,
        node_match=nx.isomorphism.categorical_node_match(
            ["label", "typing"],
            default=["", ""],
        ),
        edge_match=nx.isomorphism.categorical_edge_match(
            ["position", "predicate"],
            default=[-1, ""],
        ),
        strictly_decreasing=False,
        timeout=timeout,
    )

    paths: list[Any] = []
    bestcost = float("inf")
    for vertex_path, _, cost in edit_path_gen:
        if bestcost != float("inf") and cost < bestcost:
            paths = []

        paths.append(vertex_path)
        bestcost = cost
        if timed_out():
            break

    return paths, bestcost, timed_out()


def map(
    source: ProblemGraph | SceneGraph,
    target: ProblemGraph | SceneGraph,
    mapping: dict | None = None,
    return_mappings: bool = False,
) -> list[dict] | bool:
    """
    Find all valid isomorphic mappings between nodes of two scene graphs.

    Parameters:
        source (ProblemGraph): The source problem graph.
        target (ProblemGraph): The target problem graph.
        mapping (dict | None): The initial mapping between node names.
        return_mappings (bool): If True, the function will return a list of
            dictionaries representing valid mappings. If False, the function
            will return a boolean indicating if there is a valid mapping.

    Returns:
        list | bool: A list of dictionaries representing valid mappings or a
            boolean indicating if there is a valid mapping.
    """
    if not nx.faster_could_be_isomorphic(source, target):
        return [] if return_mappings else False

    matching = functools.partial(_matching, mapping=mapping)
    mapper = nx.isomorphism.MultiDiGraphMatcher(
        source,
        target,
        node_match=matching,
        edge_match=nx.isomorphism.categorical_edge_match(
            ["scene", "position", "predicate"],
            default=["", -1, ""],
        ),
    )

    if return_mappings:
        return list(mapper.isomorphisms_iter()) if mapper.is_isomorphic() else []
    else:
        return mapper.is_isomorphic()


def equals(
    source: ProblemGraph,
    target: ProblemGraph,
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
    if not is_placeholder:
        return nx.utils.graphs_equal(source, target) or bool(map(source, target))
    else:
        source_init, source_goal = source.decompose()
        target_init, target_goal = target.decompose()

        if nx.utils.graphs_equal(
            source_init,
            target_init,
        ) and nx.utils.graphs_equal(
            source_goal,
            target_goal,
        ):
            return True

        valid_init = bool(map(source_init, target_init))
        valid_goal = bool(map(source_goal, target_goal))

        return valid_init and valid_goal


def distance(
    initial_source: SceneGraph,
    initial_target: SceneGraph,
    goal_source: SceneGraph,
    goal_target: SceneGraph,
    timeout: float | None = None,
) -> tuple[float, bool, float, bool]:
    """
    Calculate the graph edit distance between initial and goal scene graphs.

    Parameters:
        initial_source (SceneGraph): The initial source scene graph.
        initial_target (SceneGraph): The initial target scene graph.
        goal_source (SceneGraph): The goal source scene graph.
        goal_target (SceneGraph): The goal target scene graph.
        timeout (float | None): The maximum number of seconds to spend.

    Returns:
        tuple:
            - float: The graph edit distance between the two initial scenes.
            - bool: True if the timeout was reached, False otherwise while
                calculating initial scene distance.
            - float: The graph edit distance between the two goal scenes.
            - bool: True if the timeout was reached, False otherwise while
                calculating goal scene distance.
    """

    start_time = time.perf_counter()

    def timed_out() -> bool:
        return bool(timeout and time.perf_counter() - start_time > timeout)

    def mapping_to_fn(mapping: list) -> Callable[[dict, dict], bool]:
        """
        Convert a mapping to a matching function.

        Parameters:
            mapping (list): The mapping between node names.

        Returns:
            callable: A matching function.
        """
        map_dict = {k: v for k, v in mapping}

        def matching(source: Node, target: Node) -> bool:
            return (
                source["name"] not in map_dict
                or map_dict.get(source["name"]) == target["name"]
            )

        return matching

    if equals(
        ProblemGraph.join(initial_source, goal_source),
        ProblemGraph.join(initial_target, goal_target),
        is_placeholder=False,
    ):
        return 0.0, False, 0.0, False

    minimal_mappings, init_dist, approx_init_dist = _minimal_mappings(
        initial_source,
        initial_target,
        timeout=timeout,
    )

    goal_dist = float("inf")
    for mapping in minimal_mappings:
        # use the mapping from the initial graph
        edit_path_gen = nx.similarity.optimize_edit_paths(
            goal_source,
            goal_target,
            node_match=mapping_to_fn(mapping),
            edge_match=nx.isomorphism.categorical_edge_match(
                ["position", "predicate"],
                default=[-1, ""],
            ),
            timeout=timeout,
        )

        for _, _, cost in edit_path_gen:
            if cost < goal_dist:
                goal_dist = cost
            if timed_out():
                break

    return init_dist, approx_init_dist, goal_dist, timed_out()
