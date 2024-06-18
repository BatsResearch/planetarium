from collections import defaultdict
import copy
import enum

import networkx as nx

from planetarium import graph


class ReductionNode(tuple, enum.Enum):
    TABLE = ("table", ("blocksworld",))
    CLEAR = ("clear", ("blocksworld", "gripper"))
    ARM = ("arm", ("blocksworld",))
    ROOMS = ("rooms", ("gripper",))
    BALLS = ("balls", ("gripper",))
    GRIPPERS = ("grippers", ("gripper",))
    ROBBY = ("robby", ("gripper",))


REDUCTION_NODES = [e.value for e in ReductionNode]


def _reduce_blocksworld(
    scene: graph.SceneGraph,
    validate: bool = True,
) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
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
        nx.MultiDiGraph: The reduced scene graph.
    """

    nodes = defaultdict(list)
    for node, node_attr in scene.nodes(data=True):
        nodes[node_attr["label"]].append((node, node_attr))

    reduced = nx.MultiDiGraph()
    for e in ReductionNode:
        predicate, r_node_domains = e.value
        if "blocksworld" in r_node_domains:
            reduced.add_node(
                e,
                name=predicate,
                label=graph.Label.PREDICATE,
                typing={predicate},
            )

    for obj, attrs in nodes[graph.Label.CONSTANT]:
        reduced.add_node(obj, **attrs)
    if "arm-empty" in scene.predicates:
        reduced.add_edge(
            ReductionNode.CLEAR,
            ReductionNode.ARM,
            pred="arm-empty",
        )

    pred_nodes = set()
    for node, obj, edge_attr in scene.edges(data=True):
        pred = edge_attr["predicate"]
        if node in pred_nodes:
            continue
        elif pred == "on-table":
            reduced.add_edge(obj, ReductionNode.TABLE, pred="on-table")
        elif pred == "clear":
            reduced.add_edge(ReductionNode.CLEAR, obj, pred="clear")
        elif pred == "on":
            pos = edge_attr["position"]
            other_obj, *_ = [
                v
                for _, v, a in scene.out_edges(node, data=True)
                if a["position"] == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, pred="on")
        elif pred == "holding":
            reduced.add_edge(obj, ReductionNode.ARM, pred="holding")

        pred_nodes.add(node)

    if validate:
        if not nx.is_directed_acyclic_graph(reduced):
            raise ValueError("Scene graph is not a Directed Acyclic Graph.")
        for node in reduced.nodes:
            if (node != ReductionNode.TABLE and reduced.in_degree(node) > 1) or (
                node != ReductionNode.CLEAR and reduced.out_degree(node) > 1
            ):
                raise ValueError(
                    f"Node {node} has multiple parents/children. (not possible in blocksworld)."
                )
            if reduced.in_degree(ReductionNode.ARM) == 1:
                obj = next(reduced.predecessors(ReductionNode.ARM))
                if (
                    obj != ReductionNode.CLEAR
                    and reduced.in_degree(obj) == 1
                    and next(reduced.predecessors(obj)) != ReductionNode.CLEAR
                ):
                    raise ValueError("Object on arm is connected to another object.")

    reduced._domain = scene._domain
    reduced._constants = scene._constants
    reduced._predicates = scene._predicates

    return reduced


def _reduce_gripper(
    scene: graph.SceneGraph,
    validate: bool = True,
) -> nx.MultiDiGraph:
    """Reduces a gripper scene graph to a Directed Acyclic Graph.

    Args:
        scene (graph.SceneGraph): The scene graph to reduce.
        validate (bool, optional): Whether or not to validate if the reduced
            reprsentation is valid and a DAG. Defaults to True.

    Returns:
        nx.SceneGraph: The reduced problem graph.
    """
    nodes = defaultdict(list)
    for node, node_attr in scene.nodes(data=True):
        nodes[node_attr["label"]].append((node, node_attr))

    reduced = nx.MultiDiGraph()
    for e in ReductionNode:
        predicate, r_node_domains = e.value
        if "gripper" in r_node_domains:
            reduced.add_node(
                e,
                name=predicate,
                label=graph.Label.PREDICATE,
                typing={predicate},
            )

    for obj, attrs in nodes[graph.Label.CONSTANT]:
        reduced.add_node(obj, **copy.deepcopy(attrs))

    pred_nodes = set()
    for node, obj, edge_attr in scene.edges(data=True):
        pred = edge_attr["predicate"]
        if node in pred_nodes:
            continue
        elif pred == "at-robby":
            reduced.add_edge(ReductionNode.ROBBY, obj, pred=pred)
        elif pred == "free":
            reduced.add_edge(ReductionNode.CLEAR, obj, pred=pred)
        elif pred == "ball":
            reduced.add_edge(ReductionNode.BALLS, obj, pred=pred)
        elif pred == "gripper":
            reduced.add_edge(ReductionNode.GRIPPERS, obj, pred=pred)
        elif pred == "room":
            reduced.add_edge(ReductionNode.ROOMS, obj, pred=pred)
        elif pred in {"carry", "at"}:
            pos = edge_attr["position"]
            other_obj, *_ = [
                v
                for _, v, a in scene.out_edges(node, data=True)
                if a["position"] == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, pred=pred)

        pred_nodes.add(node)

    if validate and not nx.is_directed_acyclic_graph(reduced):
        raise ValueError("Scene graph is not a Directed Acyclic Graph.")

    reduced._domain = scene._domain
    reduced._constants = scene._constants
    reduced._predicates = scene._predicates

    return reduced


def _inflate_blocksworld(scene: nx.MultiDiGraph) -> graph.SceneGraph:
    """Respecify a blocksworld scene graph.

    Args:
        scene (nx.MultiDiGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node, attrs in scene.nodes(data=True):
        if not isinstance(node, ReductionNode):
            constants.append({"name": node, "typing": attrs["typing"]})

    for u, v, _ in scene.edges:
        if u == ReductionNode.CLEAR and v == ReductionNode.ARM:
            predicates.append(
                {
                    "typing": "arm-empty",
                    "parameters": [],
                }
            )
        elif u == ReductionNode.CLEAR:
            predicates.append(
                {
                    "typing": "clear",
                    "parameters": [v],
                }
            )
        elif v == ReductionNode.TABLE:
            predicates.append(
                {
                    "typing": "on-table",
                    "parameters": [u],
                }
            )
        elif v == ReductionNode.ARM:
            predicates.append(
                {
                    "typing": "holding",
                    "parameters": [u],
                }
            )
        else:
            predicates.append(
                {
                    "typing": "on",
                    "parameters": [u, v],
                }
            )

    return graph.SceneGraph(constants, predicates, domain="blocksworld")


def _inflate_gripper(scene: nx.MultiDiGraph) -> graph.SceneGraph:
    """Respecify a gripper scene graph.

    Args:
        scene (nx.MultiDiGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node, attrs in scene.nodes(data=True):
        if not isinstance(node, ReductionNode):
            constants.append({"name": node, "typing": attrs["typing"]})

    for u, v, attr in scene.edges(data=True):
        if u == ReductionNode.ROBBY:
            predicates.append(
                {
                    "typing": "at-robby",
                    "parameters": [v],
                }
            )
        elif u == ReductionNode.CLEAR:
            predicates.append(
                {
                    "typing": "free",
                    "parameters": [v],
                }
            )
        elif u == ReductionNode.BALLS:
            predicates.append(
                {
                    "typing": "ball",
                    "parameters": [v],
                }
            )
        elif u == ReductionNode.GRIPPERS:
            predicates.append(
                {
                    "typing": "gripper",
                    "parameters": [v],
                }
            )
        elif u == ReductionNode.ROOMS:
            predicates.append(
                {
                    "typing": "room",
                    "parameters": [v],
                }
            )
        else:
            predicates.append(
                {
                    "typing": attr["pred"],
                    "parameters": [u, v],
                }
            )

    return graph.SceneGraph(constants, predicates, domain="gripper")


def _blocksworld_underspecified_blocks(
    scene: nx.MultiDiGraph,
) -> tuple[set[str], set[str], bool]:
    """Finds blocks that are not fully specified.

    Args:
        scene (nx.MultiDiGraph): The reduced SceneGraph of a scene.

    Returns:
        tuple[set[str], set[str], bool]: The set of blocks that are not fully
        specified.
         - blocks that do not specify what is on top.
         - blocks that do not specify what is on the bottom.
    """
    top_blocks = set()
    bottom_blocks = set()
    held_block = next(scene.predecessors(ReductionNode.ARM), None)
    for node, attrs in scene.nodes(data=True, default=None):
        if attrs.get("label") == graph.Label.CONSTANT:
            if not scene.in_edges(node) and node != held_block:
                top_blocks.add(node)
            if not scene.out_edges(node):
                bottom_blocks.add(node)
    return top_blocks, bottom_blocks, held_block is None


def _gripper_get_typed_objects(scene: nx.MultiDiGraph):
    rooms = set()
    balls = set()
    grippers = set()

    for _, node in scene.out_edges(ReductionNode.ROOMS):
        rooms.add(node)
    for _, node in scene.out_edges(ReductionNode.BALLS):
        balls.add(node)
    for _, node in scene.out_edges(ReductionNode.GRIPPERS):
        grippers.add(node)

    return {
        ReductionNode.ROOMS: rooms,
        ReductionNode.BALLS: balls,
        ReductionNode.GRIPPERS: grippers,
    }


def _gripper_underspecified_blocks(
    init: nx.MultiDiGraph,
    goal: nx.MultiDiGraph,
) -> tuple[set[str], set[str], bool]:
    """Finds blocks that are not fully specified.

    Args:
        init (nx.MultiDiGraph): The reduced SceneGraph of the initial scene.
        goal (nx.MultiDiGraph): The reduced SceneGraph of the goal scene.

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

    for ball in typed[ReductionNode.BALLS]:
        ball_edges = [
            node
            for _, node in goal.out_edges(ball)
            if not isinstance(node, ReductionNode)
        ]
        if not ball_edges:
            underspecified_balls.add(ball)
    for gripper in typed[ReductionNode.GRIPPERS]:
        gripper_edges = [
            node
            for node, _ in goal.in_edges(gripper)
            if node == ReductionNode.CLEAR or not isinstance(node, ReductionNode)
        ]
        if not gripper_edges:
            underspecified_grippers.add(gripper)

    return (
        underspecified_balls,
        underspecified_grippers,
        goal.out_degree(ReductionNode.ROBBY) == 0,
    )


def inflate(
    scene: nx.MultiDiGraph,
    domain: str | None = None,
) -> graph.SceneGraph:
    """Inflate a reduced scene graph to a SceneGraph.

    Args:
        scene (nx.MultiDiGraph): The reduced scene graph to respecify.
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
    scene: nx.MultiDiGraph,
) -> tuple[set[str], set[str]]:
    """Finds nodes that are not connected to the rest of the scene graph.

    Args:
        nodesA (set[str]): The set of nodes to check.
        nodesB (set[str]): The set of nodes to check against.
        scene (nx.MultiDiGraph): The scene graph to check against.

    Returns:
        tuple[set[str], set[str]]: The set of nodes that are not connected to
        the rest of the scene graph.
    """
    _nodesA = set(nodesA)
    _nodesB = set(nodesB)

    for a in nodesA:
        for b in nodesB:
            if not nx.has_path(scene, a, b) and not nx.has_path(scene, b, a):
                _nodesA.discard(a)
                _nodesB.discard(b)

    return _nodesA, _nodesB


def _fully_specify_blocksworld(
    scene: nx.MultiDiGraph,
) -> graph.SceneGraph:
    """Fully specifies a blocksworld scene graph.

    Adds any missing edges to fully specify the scene graph, without adding
    edges that change the problem represented by the graph.

    Args:
        scene (nx.MultiDiGraph): The reduced SceneGraph of a scene.

    Returns:
        SceneGraph: The fully specified scene graph.
    """
    scene = copy.deepcopy(scene)
    top_blocks, bottom_blocks, arm_empty = _blocksworld_underspecified_blocks(scene)
    top_blocks_, bottom_blocks_ = _detached_blocks(top_blocks, bottom_blocks, scene)

    for block in top_blocks_:
        scene.add_edge(ReductionNode.CLEAR, block, pred="clear")
    for block in bottom_blocks_:
        scene.add_edge(block, ReductionNode.TABLE, pred="on-table")

    # handle arm
    if arm_empty and not (top_blocks & bottom_blocks):
        scene.add_edge(ReductionNode.CLEAR, ReductionNode.ARM, pred="arm-empty")

    return scene


def _fully_specify_gripper(
    init: nx.MultiDiGraph,
    goal: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Fully specifies a gripper scene graph.

    Adds any missing edges to fully specify the scene graph, without adding
    edges that change the problem represented by the graph.

    Args:
        init (nx.MultiDiGraph): The reduced SceneGraph of the initial scene.
        goal (nx.MultiDiGraph): The reduced SceneGraph of the goal scene.

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
            scene.add_edge(ReductionNode.CLEAR, gripper, pred="free")

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
) -> nx.MultiDiGraph:
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
        nx.MultiDiGraph: The reduced scene graph.
    """
    domain = domain or graph.domain
    match domain:
        case "blocksworld":
            return _reduce_blocksworld(graph, validate=validate)
        case "gripper":
            return _reduce_gripper(graph, validate=validate)
        case _:
            raise ValueError(f"Domain {domain} not supported.")
