from typing import Any

from collections import defaultdict
import copy
import enum

import rustworkx as rx

from planetarium import graph


class ReducedNode(tuple, enum.Enum):
    TABLE = ("table", ("blocksworld",))
    CLEAR = ("clear", ("blocksworld", "gripper"))
    ARM = ("arm", ("blocksworld",))
    ROOMS = ("rooms", ("gripper",))
    BALLS = ("balls", ("gripper",))
    GRIPPERS = ("grippers", ("gripper",))
    ROBBY = ("robby", ("gripper",))


class ReducedGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        predicates: list[dict[str, Any]],
        domain: str,
    ):
        super().__init__(constants, predicates, domain=domain)

        for e in ReducedNode:
            predicate, r_node_domains = e.value
            if self.domain in r_node_domains:
                self.add_node(
                    graph.PlanGraphNode(
                        e,
                        name=predicate,
                        label=graph.Label.PREDICATE,
                        typing={predicate},
                    ),
                )


def _reduce_blocksworld(
    scene: graph.SceneGraph,
    validate: bool = True,
) -> ReducedGraph:
    """Reduces a blocksworld scene graph to a Directed Acyclic Graph.

    Args:
        problem (graph.SceneGraph): The scene graph to reduce.
        validate (bool, optional): Whether or not to validate if the reduced
            reprsentation is valid. Defaults to True.

    Raises:
        ValueError: If the reduced graph is not a Directed Acyclic Graph and
            validate is True.
        ValueError: If a node has multiple parents/children (not allowed in
            blocksworld) and if validate is True.

    Returns:
        ReducedGraph: The reduced problem graph.
    """

    nodes = defaultdict(list)
    for node in scene.nodes:
        nodes[node.label].append(node)

    reduced = ReducedGraph(
        constants=scene._constants,
        predicates=scene._predicates,
        domain="blocksworld",
    )

    if "arm-empty" in scene.predicates:
        reduced.add_edge(
            ReducedNode.CLEAR,
            ReducedNode.ARM,
            graph.PlanGraphEdge(predicate="arm-empty"),
        )

    pred_nodes = set()
    for node, obj, edge in scene.edges:
        pred = edge.predicate
        reduced_edge = graph.PlanGraphEdge(predicate=pred)
        if node in pred_nodes:
            continue
        elif pred == "on-table":
            reduced.add_edge(obj, ReducedNode.TABLE, reduced_edge)
        elif pred == "clear":
            reduced.add_edge(ReducedNode.CLEAR, obj, reduced_edge)
        elif pred == "on":
            pos = edge.position
            other_obj, *_ = [
                v.node for _, v, a in scene.out_edges(node) if a.position == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, reduced_edge)
        elif pred == "holding":
            reduced.add_edge(obj, ReducedNode.ARM, reduced_edge)
        pred_nodes.add(node)

    if validate:
        if not rx.is_directed_acyclic_graph(reduced.graph):
            raise ValueError("Scene graph is not a Directed Acyclic Graph.")
        for node in reduced.nodes:
            if (
                node.node != ReducedNode.TABLE and reduced.in_degree(node.node) > 1
            ) or (node.node != ReducedNode.CLEAR and reduced.out_degree(node.node) > 1):
                raise ValueError(
                    f"Node {node} has multiple parents/children. (not possible in blocksworld)."
                )
            if reduced.in_degree(ReducedNode.ARM) == 1:
                obj = reduced.predecessors(ReducedNode.ARM)[0]
                if (
                    obj.node != ReducedNode.CLEAR
                    and reduced.in_degree(obj) == 1
                    and reduced.predecessors(obj)[0].node != ReducedNode.CLEAR
                ):
                    raise ValueError("Object on arm is connected to another object.")

    return reduced


def _reduce_gripper(
    scene: graph.SceneGraph,
    validate: bool = True,
) -> ReducedGraph:
    """Reduces a gripper scene graph to a Directed Acyclic Graph.

    Args:
        scene (graph.SceneGraph): The scene graph to reduce.
        validate (bool, optional): Whether or not to validate if the reduced
            reprsentation is valid and a DAG. Defaults to True.

    Returns:
        ReducedGraph: The reduced problem graph.
    """
    nodes = defaultdict(list)
    for node in scene.nodes:
        nodes[node.label].append(node)

    reduced = ReducedGraph(
        constants=scene._constants,
        predicates=scene._predicates,
        domain="gripper",
    )

    pred_nodes = set()
    for node, obj, edge in scene.edges:
        pred = edge.predicate
        reduced_edge = graph.PlanGraphEdge(predicate=pred)
        if node in pred_nodes:
            continue
        elif pred == "at-robby":
            reduced.add_edge(ReducedNode.ROBBY, obj, reduced_edge)
        elif pred == "free":
            reduced.add_edge(ReducedNode.CLEAR, obj, reduced_edge)
        elif pred == "ball":
            reduced.add_edge(ReducedNode.BALLS, obj, reduced_edge)
        elif pred == "gripper":
            reduced.add_edge(ReducedNode.GRIPPERS, obj, reduced_edge)
        elif pred == "room":
            reduced.add_edge(ReducedNode.ROOMS, obj, reduced_edge)
        elif pred in {"carry", "at"}:
            pos = edge.position
            other_obj, *_ = [
                v for _, v, a in scene.out_edges(node) if a.position == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, reduced_edge)

        pred_nodes.add(node)

    if validate and not rx.is_directed_acyclic_graph(reduced.graph):
        raise ValueError("Scene graph is not a Directed Acyclic Graph.")

    return reduced


def _inflate_blocksworld(scene: ReducedGraph) -> graph.SceneGraph:
    """Respecify a blocksworld scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node in scene.nodes:
        if not isinstance(node.node, ReducedNode):
            constants.append({"name": node.node, "typing": node.typing})

    for u, v, _ in scene.edges:
        if u.node == ReducedNode.CLEAR and v.node == ReducedNode.ARM:
            predicates.append(
                {
                    "typing": "arm-empty",
                    "parameters": [],
                }
            )
        elif u.node == ReducedNode.CLEAR:
            predicates.append(
                {
                    "typing": "clear",
                    "parameters": [v.node],
                }
            )
        elif v.node == ReducedNode.TABLE:
            predicates.append(
                {
                    "typing": "on-table",
                    "parameters": [u.node],
                }
            )
        elif v.node == ReducedNode.ARM:
            predicates.append(
                {
                    "typing": "holding",
                    "parameters": [u.node],
                }
            )
        else:
            predicates.append(
                {
                    "typing": "on",
                    "parameters": [u.node, v.node],
                }
            )

    return graph.SceneGraph(constants, predicates, domain="blocksworld")


def _inflate_gripper(scene: ReducedGraph) -> graph.SceneGraph:
    """Respecify a gripper scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node in scene.nodes:
        if not isinstance(node.node, ReducedNode):
            constants.append({"name": node.node, "typing": node.typing})

    for u, v, edge in scene.edges:
        if u.node == ReducedNode.ROBBY:
            predicates.append(
                {
                    "typing": "at-robby",
                    "parameters": [v.node],
                }
            )
        elif u.node == ReducedNode.CLEAR:
            predicates.append(
                {
                    "typing": "free",
                    "parameters": [v.node],
                }
            )
        elif u.node == ReducedNode.BALLS:
            predicates.append(
                {
                    "typing": "ball",
                    "parameters": [v.node],
                }
            )
        elif u.node == ReducedNode.GRIPPERS:
            predicates.append(
                {
                    "typing": "gripper",
                    "parameters": [v.node],
                }
            )
        elif u.node == ReducedNode.ROOMS:
            predicates.append(
                {
                    "typing": "room",
                    "parameters": [v.node],
                }
            )
        else:
            predicates.append(
                {
                    "typing": edge.predicate,
                    "parameters": [u.node, v.node],
                }
            )

    return graph.SceneGraph(constants, predicates, domain="gripper")


def _blocksworld_underspecified_blocks(
    scene: ReducedGraph,
) -> tuple[set[str], set[str], bool]:
    """Finds blocks that are not fully specified.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        tuple[set[str], set[str], bool]: The set of blocks that are not fully
        specified.
         - blocks that do not specify what is on top.
         - blocks that do not specify what is on the bottom.
    """
    top_blocks = set()
    bottom_blocks = set()
    arm_behavior_defined = scene.in_degree(ReducedNode.ARM) > 0
    held_block = (
        scene.predecessors(ReducedNode.ARM)[0] if arm_behavior_defined else None
    )
    for node in scene.nodes:
        if node.label == graph.Label.CONSTANT:
            if not scene.in_edges(node) and node != held_block:
                top_blocks.add(node)
            if not scene.out_edges(node):
                bottom_blocks.add(node)
    return top_blocks, bottom_blocks, not arm_behavior_defined


def _gripper_get_typed_objects(scene: ReducedGraph):
    rooms = set()
    balls = set()
    grippers = set()

    for _, node, _ in scene.out_edges(ReducedNode.ROOMS):
        rooms.add(node)
    for _, node, _ in scene.out_edges(ReducedNode.BALLS):
        balls.add(node)
    for _, node, _ in scene.out_edges(ReducedNode.GRIPPERS):
        grippers.add(node)

    return {
        ReducedNode.ROOMS: rooms,
        ReducedNode.BALLS: balls,
        ReducedNode.GRIPPERS: grippers,
    }


def _gripper_underspecified_blocks(
    init: ReducedGraph,
    goal: ReducedGraph,
) -> tuple[set[str], set[str], bool]:
    """Finds blocks that are not fully specified.

    Args:
        init (ReducedGraph): The reduced SceneGraph of the initial scene.
        goal (ReducedGraph): The reduced SceneGraph of the goal scene.

    Returns:
        tuple[set[str], set[str]]: The set of blocks that are not fully
        specified.
         - balls that do not specify being carried or being in a room.
         - grippers that do not specify being free or carrying a ball.
         - whether robby is not in a room.
    """

    typed = _gripper_get_typed_objects(init)

    underspecified_balls = set()
    underspecified_grippers = set()

    for ball in typed[ReducedNode.BALLS]:
        ball_edges = [
            node
            for _, node, _ in goal.out_edges(ball)
            if not isinstance(node, ReducedNode)
        ]
        if not ball_edges:
            underspecified_balls.add(ball)
    for gripper in typed[ReducedNode.GRIPPERS]:
        gripper_edges = [
            node
            for node, _ in goal.in_edges(gripper)
            if node == ReducedNode.CLEAR or not isinstance(node, ReducedNode)
        ]
        if not gripper_edges:
            underspecified_grippers.add(gripper)

    return (
        underspecified_balls,
        underspecified_grippers,
        goal.out_degree(ReducedNode.ROBBY) == 0,
    )


def inflate(
    scene: ReducedGraph,
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
        case _:
            raise ValueError(f"Domain {domain} not supported.")


def _detached_blocks(
    nodesA: set[str],
    nodesB: set[str],
    scene: ReducedGraph,
) -> tuple[set[str], set[str]]:
    """Finds nodes that are not connected to the rest of the scene graph.

    Args:
        nodesA (set[str]): The set of nodes to check.
        nodesB (set[str]): The set of nodes to check against.
        scene (ReducedGraph): The scene graph to check against.

    Returns:
        tuple[set[str], set[str]]: The set of nodes that are not connected to
        the rest of the scene graph.
    """
    _nodesA = set(nodesA)
    _nodesB = set(nodesB)

    for a in nodesA:
        for b in nodesB:
            a_index = scene.nodes.index(a)
            b_index = scene.nodes.index(b)
            if not rx.has_path(scene.graph, a_index, b_index, as_undirected=True):
                _nodesA.discard(a)
                _nodesB.discard(b)

    return _nodesA, _nodesB


def _fully_specify_blocksworld(
    scene: ReducedGraph,
) -> graph.SceneGraph:
    """Fully specifies a blocksworld scene graph.

    Adds any missing edges to fully specify the scene graph, without adding
    edges that change the problem represented by the graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        SceneGraph: The fully specified scene graph.
    """
    scene = copy.deepcopy(scene)
    top_blocks, bottom_blocks, arm_empty = _blocksworld_underspecified_blocks(scene)
    top_blocks_, bottom_blocks_ = _detached_blocks(top_blocks, bottom_blocks, scene)

    for block in top_blocks_:
        scene.add_edge(
            ReducedNode.CLEAR,
            block,
            graph.PlanGraphEdge(predicate="clear"),
        )
    for block in bottom_blocks_:
        scene.add_edge(
            block,
            ReducedNode.TABLE,
            graph.PlanGraphEdge(predicate="on-table"),
        )

    # handle arm
    if arm_empty and not (top_blocks & bottom_blocks):
        scene.add_edge(
            ReducedNode.CLEAR,
            ReducedNode.ARM,
            graph.PlanGraphEdge(predicate="arm-empty"),
        )

    return scene


def _fully_specify_gripper(
    init: ReducedGraph,
    goal: ReducedGraph,
) -> ReducedGraph:
    """Fully specifies a gripper scene graph.

    Adds any missing edges to fully specify the scene graph, without adding
    edges that change the problem represented by the graph.

    Args:
        init (ReducedGraph): The reduced SceneGraph of the initial scene.
        goal (ReducedGraph): The reduced SceneGraph of the goal scene.

    Returns:
        SceneGraph: The fully specified scene graph.
    """
    scene = copy.deepcopy(goal)

    underspecified_balls, underspecified_grippers, _ = _gripper_underspecified_blocks(
        init,
        goal,
    )

    if underspecified_grippers and not underspecified_balls:
        for gripper in underspecified_grippers:
            scene.add_edge(
                ReducedNode.CLEAR, gripper, graph.PlanGraphEdge(predicate="free")
            )

    return scene


def fully_specify(
    problem: graph.ProblemGraph,
    domain: str | None = None,
) -> graph.ProblemGraph:
    """Fully specifies a goal state.

    Args:
        problem (graph.ProblemGraph): The problem graph with the goal state to
            fully specify.
        domain (str | None, optional): The domain of the scene graph. Defaults
            to None.

    Returns:
        graph.ProblemGraph: The fully specified problem graph.
    """
    domain = domain or problem.domain
    init, goal = problem.decompose()
    match domain:
        case "blocksworld":
            reduced_goal = _reduce_blocksworld(goal)
            fully_specified_goal = _fully_specify_blocksworld(reduced_goal)
        case "gripper":
            reduced_init = _reduce_gripper(init)
            reduced_goal = _reduce_gripper(goal)
            fully_specified_goal = _fully_specify_gripper(
                reduced_init,
                reduced_goal,
            )
        case _:
            raise ValueError(f"Domain {domain} not supported.")
    return graph.ProblemGraph.join(
        init,
        inflate(fully_specified_goal, domain=domain),
    )


def is_fully_specified(
    problem: graph.ProblemGraph,
    domain: str | None = None,
    is_placeholder: bool = False,
) -> bool:
    """Checks if a goal state is fully specified.

    Args:
        problem (graph.ProblemGraph): The problem graph with the goal state to
            evaluate.
        domain (str | None, optional): The domain of the scene graph. Defaults
            to None.
        is_placeholder (bool, optional): Whether or not every edge must be
            present. Defaults to False.

    Raises:
        ValueError: If a certain domain is provided but not supported.

    Returns:
        bool: True if the goal state is fully specified, False otherwise.
    """
    domain = domain or problem.domain
    init, goal = problem.decompose()
    match domain:
        case "blocksworld":
            reduced_goal = _reduce_blocksworld(goal)
            top, bottom, arm_empty = _blocksworld_underspecified_blocks(reduced_goal)

            if is_placeholder:
                if bottom.intersection(top) and arm_empty:
                    return False
                return (not top) or (not bottom)
            else:
                return (not (top | bottom)) and not arm_empty
        case "gripper":
            reduced_init = _reduce_gripper(init)
            reduced_goal = _reduce_gripper(goal)

            balls, grippers, no_at_robby = _gripper_underspecified_blocks(
                reduced_init,
                reduced_goal,
            )
            # check number of typed objects is the same as total
            goal_type_check = len(goal.constants) == len(
                set().union(*_gripper_get_typed_objects(reduced_goal).values())
            )
            type_check = (
                len(init.constants)
                == len(set().union(*_gripper_get_typed_objects(reduced_init).values()))
                and goal_type_check
            )
            if is_placeholder:
                return len(balls) == 0 and not no_at_robby
            else:
                return len(balls or grippers) == 0 and not no_at_robby and type_check
        case _:
            raise ValueError(f"Domain {domain} not supported.")


def reduce(
    graph: graph.SceneGraph,
    domain: str | None = None,
    validate: bool = True,
) -> ReducedGraph:
    """Reduces a scene graph to a Directed Acyclic Graph.

    Args:
        graph (graph.SceneGraph): The scene graph to reduce.
        domain (str, optional): The domain of the scene graph. Defaults to
            "blocksworld".
        validate (bool, optional): Whether or not to validate if the reduced
            reprsentation is valid and a DAG. Defaults to True.

    Raises:
        ValueError: If a certain domain is provided but not supported.

    Returns:
        ReducedGraph: The reduced problem graph.
    """
    domain = domain or graph.domain
    match domain:
        case "blocksworld":
            return _reduce_blocksworld(graph, validate=validate)
        case "gripper":
            return _reduce_gripper(graph, validate=validate)
        case _:
            raise ValueError(f"Domain {domain} not supported.")
