from typing import Any

from collections import defaultdict
import copy
import enum

import rustworkx as rx

from planetarium import graph


class ReducedNode(str, enum.Enum):
    TABLE = "table"
    CLEAR = "clear"
    ARM = "arm"
    ROOMS = "room"
    BALLS = "ball"
    GRIPPERS = "gripper"
    ROBBY = "at-robby"
    FREE = "free"


BlocksworldReducedNodes = {
    ReducedNode.TABLE,
    ReducedNode.CLEAR,
    ReducedNode.ARM,
}

GripperReducedNodes = {
    ReducedNode.ROOMS,
    ReducedNode.BALLS,
    ReducedNode.GRIPPERS,
    ReducedNode.ROBBY,
    ReducedNode.FREE,
}

ReducedNodes = {
    "blocksworld": BlocksworldReducedNodes,
    "gripper": GripperReducedNodes,
}


class ReducedSceneGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str,
        scene: graph.Scene | None = None,
    ):
        super().__init__(constants, domain=domain)
        self.scene = scene

        for e in ReducedNodes[domain]:
            predicate = e.value
            self.add_node(
                graph.PlanGraphNode(
                    e,
                    name=predicate,
                    label=graph.Label.PREDICATE,
                    typing={predicate},
                ),
            )


class ReducedProblemGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str,
    ):
        super().__init__(constants, domain=domain)

        for e in ReducedNodes[domain]:
            predicate = e.value
            self.add_node(
                graph.PlanGraphNode(
                    e,
                    name=predicate,
                    label=graph.Label.PREDICATE,
                    typing={predicate},
                ),
            )

    def decompose(self) -> tuple[ReducedSceneGraph, ReducedSceneGraph]:
        init = ReducedSceneGraph(self.constants, self.domain, scene=graph.Scene.INIT)
        goal = ReducedSceneGraph(self.constants, self.domain, scene=graph.Scene.GOAL)

        for u, v, edge in self.edges:
            edge = copy.deepcopy(edge)
            if edge.scene == graph.Scene.INIT:
                init.add_edge(u, v, edge)
            elif edge.scene == graph.Scene.GOAL:
                goal.add_edge(u, v, edge)

        return init, goal

    @staticmethod
    def join(init: ReducedSceneGraph, goal: ReducedSceneGraph) -> "ReducedProblemGraph":
        problem = ReducedProblemGraph(init.constants, domain=init.domain)

        for u, v, edge in init.edges:
            edge = copy.deepcopy(edge)
            problem.add_edge(u, v, edge)
            edge.scene = graph.Scene.INIT
        for u, v, edge in goal.edges:
            edge = copy.deepcopy(edge)
            edge.scene = graph.Scene.GOAL
            problem.add_edge(u, v, edge)

        return problem


def _reduce_blocksworld(
    scene: graph.SceneGraph | graph.ProblemGraph,
    validate: bool = True,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a blocksworld scene graph to a Directed Acyclic Graph.

    Args:
        problem (graph.SceneGraph | graph.ProblemGraph): The scene graph to
            reduce.
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

    if isinstance(scene, graph.ProblemGraph):
        reduced = ReducedProblemGraph(constants=scene.constants, domain="blocksworld")
    elif isinstance(scene, graph.SceneGraph):
        reduced = ReducedSceneGraph(
            constants=scene.constants,
            domain="blocksworld",
            scene=scene.scene,
        )
    else:
        raise ValueError("Scene must be a SceneGraph or ProblemGraph.")

    for pred_node in scene.predicate_nodes:
        if pred_node.typing == "arm-empty":
            reduced.add_edge(
                ReducedNode.CLEAR,
                ReducedNode.ARM,
                graph.PlanGraphEdge(
                    predicate="arm-empty",
                    scene=pred_node.scene,
                ),
            )

    pred_nodes = set()
    for node, obj, edge in scene.edges:
        pred = edge.predicate
        reduced_edge = graph.PlanGraphEdge(predicate=pred, scene=edge.scene)
        if node in pred_nodes:
            continue
        if pred == "on-table":
            reduced.add_edge(obj, ReducedNode.TABLE, reduced_edge)
        elif pred == "clear":
            reduced.add_edge(ReducedNode.CLEAR, obj, reduced_edge)
        elif pred == "on":
            pos = edge.position
            other_obj, *_ = [
                v.node for v, a in scene.out_edges(node) if a.position == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, reduced_edge)
        elif pred == "holding":
            reduced.add_edge(obj, ReducedNode.ARM, reduced_edge)
        pred_nodes.add(node)

    if validate:
        if isinstance(reduced, ReducedProblemGraph):
            init, goal = reduced.decompose()
            _validate_blocksworld(init)
            _validate_blocksworld(goal)
        elif isinstance(reduced, ReducedSceneGraph):
            _validate_blocksworld(reduced)

    return reduced


def _validate_blocksworld(scene: graph.SceneGraph):
    """Validates a blocksworld scene graph.

    Args:
        scene (graph.SceneGraph): The scene graph to validate.

    Raises:
        ValueError: If the scene graph is not a Directed Acyclic Graph.
        ValueError: If a node has multiple parents/children (not allowed in
            blocksworld).
        ValueError: If an object on the arm is connected to another object.
    """
    if not rx.is_directed_acyclic_graph(scene.graph):
        raise ValueError("Scene graph is not a Directed Acyclic Graph.")
    for node in scene.nodes:
        if (node.node != ReducedNode.TABLE and scene.in_degree(node.node) > 1) or (
            node.node != ReducedNode.CLEAR and scene.out_degree(node.node) > 1
        ):
            raise ValueError(
                f"Node {node} has multiple parents/children. (not possible in blocksworld)."
            )
        if scene.in_degree(ReducedNode.ARM) == 1:
            obj = scene.predecessors(ReducedNode.ARM)[0]
            if (
                obj.node != ReducedNode.CLEAR
                and scene.in_degree(obj) == 1
                and scene.predecessors(obj)[0].node != ReducedNode.CLEAR
            ):
                raise ValueError("Object on arm is connected to another object.")


def _reduce_gripper(
    scene: graph.SceneGraph | graph.ProblemGraph,
    validate: bool = True,
) -> ReducedSceneGraph | ReducedProblemGraph:
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

    if isinstance(scene, graph.ProblemGraph):
        reduced = ReducedProblemGraph(constants=scene.constants, domain="gripper")
    elif isinstance(scene, graph.SceneGraph):
        reduced = ReducedSceneGraph(
            constants=scene.constants,
            domain="gripper",
            scene=scene.scene,
        )
    else:
        raise ValueError("Scene must be a SceneGraph or ProblemGraph.")

    pred_nodes = set()
    for node, obj, edge in scene.edges:
        pred = edge.predicate
        reduced_edge = graph.PlanGraphEdge(predicate=pred, scene=edge.scene)
        if node in pred_nodes:
            continue
        elif pred == "at-robby":
            reduced.add_edge(ReducedNode.ROBBY, obj, reduced_edge)
        elif pred == "free":
            reduced.add_edge(ReducedNode.FREE, obj, reduced_edge)
        elif pred == "ball":
            reduced.add_edge(ReducedNode.BALLS, obj, reduced_edge)
        elif pred == "gripper":
            reduced.add_edge(ReducedNode.GRIPPERS, obj, reduced_edge)
        elif pred == "room":
            reduced.add_edge(ReducedNode.ROOMS, obj, reduced_edge)
        elif pred in {"carry", "at"}:
            pos = edge.position
            other_obj, *_ = [
                v for v, a in scene.out_edges(node) if a.position == 1 - pos
            ]
            if pos == 0:
                reduced.add_edge(obj, other_obj, reduced_edge)

        pred_nodes.add(node)

    if validate:
        if isinstance(reduced, ReducedProblemGraph):
            init, goal = reduced.decompose()
            if not rx.is_directed_acyclic_graph(init.graph):
                raise ValueError("Initial scene graph is not a Directed Acyclic Graph.")
            if not rx.is_directed_acyclic_graph(goal.graph):
                raise ValueError("Goal scene graph is not a Directed Acyclic Graph.")
        elif isinstance(reduced, ReducedSceneGraph):
            if not rx.is_directed_acyclic_graph(reduced.graph):
                raise ValueError("Scene graph is not a Directed Acyclic Graph.")

    return reduced


def _inflate_blocksworld(
    scene: ReducedSceneGraph | ReducedProblemGraph,
) -> graph.SceneGraph:
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

    for u, v, edge in scene.edges:
        if u.node == ReducedNode.CLEAR and v.node == ReducedNode.ARM:
            predicates.append(
                {
                    "typing": "arm-empty",
                    "parameters": [],
                    "scene": edge.scene,
                }
            )
        elif u.node == ReducedNode.CLEAR:
            predicates.append(
                {
                    "typing": "clear",
                    "parameters": [v.node],
                    "scene": edge.scene,
                }
            )
        elif v.node == ReducedNode.TABLE:
            predicates.append(
                {
                    "typing": "on-table",
                    "parameters": [u.node],
                    "scene": edge.scene,
                }
            )
        elif v.node == ReducedNode.ARM:
            predicates.append(
                {
                    "typing": "holding",
                    "parameters": [u.node],
                    "scene": edge.scene,
                }
            )
        else:
            predicates.append(
                {
                    "typing": "on",
                    "parameters": [u.node, v.node],
                    "scene": edge.scene,
                }
            )

    if isinstance(scene, ReducedProblemGraph):
        return graph.ProblemGraph(
            constants,
            [pred for pred in predicates if pred["scene"] == graph.Scene.INIT],
            [pred for pred in predicates if pred["scene"] == graph.Scene.GOAL],
            domain="blocksworld",
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="blocksworld",
            scene=scene.scene,
        )


def _inflate_gripper(
    scene: ReducedSceneGraph | ReducedProblemGraph,
) -> graph.SceneGraph | graph.ProblemGraph:
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
                    "scene": edge.scene,
                }
            )
        elif u.node == ReducedNode.FREE:
            predicates.append(
                {
                    "typing": "free",
                    "parameters": [v.node],
                    "scene": edge.scene,
                }
            )
        elif u.node == ReducedNode.BALLS:
            predicates.append(
                {
                    "typing": "ball",
                    "parameters": [v.node],
                    "scene": edge.scene,
                }
            )
        elif u.node == ReducedNode.GRIPPERS:
            predicates.append(
                {
                    "typing": "gripper",
                    "parameters": [v.node],
                    "scene": edge.scene,
                }
            )
        elif u.node == ReducedNode.ROOMS:
            predicates.append(
                {
                    "typing": "room",
                    "parameters": [v.node],
                    "scene": edge.scene,
                }
            )
        else:
            predicates.append(
                {
                    "typing": edge.predicate,
                    "parameters": [u.node, v.node],
                    "scene": edge.scene,
                }
            )

    if isinstance(scene, ReducedProblemGraph):
        return graph.ProblemGraph(
            constants,
            [pred for pred in predicates if pred["scene"] == graph.Scene.INIT],
            [pred for pred in predicates if pred["scene"] == graph.Scene.GOAL],
            domain="gripper",
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="gripper",
            scene=scene.scene,
        )


def _blocksworld_underspecified_blocks(
    scene: ReducedSceneGraph,
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


def _gripper_get_typed_objects(
    scene: ReducedSceneGraph,
) -> dict[ReducedNode, set[graph.PlanGraphNode]]:
    """Get the typed objects in a gripper scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        dict[ReducedNode, set[graph.PlanGraphNode]]: The typed objects in the
            scene graph.
    """
    rooms = set()
    balls = set()
    grippers = set()

    for node, _ in scene.out_edges(ReducedNode.ROOMS):
        rooms.add(node)
    for node, _ in scene.out_edges(ReducedNode.BALLS):
        balls.add(node)
    for node, _ in scene.out_edges(ReducedNode.GRIPPERS):
        grippers.add(node)

    return {
        ReducedNode.ROOMS: rooms,
        ReducedNode.BALLS: balls,
        ReducedNode.GRIPPERS: grippers,
    }


def _gripper_underspecified_blocks(
    init: ReducedSceneGraph,
    goal: ReducedSceneGraph,
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
            for node, _ in goal.out_edges(ball)
            if not isinstance(node, ReducedNode)
        ]
        if not ball_edges:
            underspecified_balls.add(ball)
    for gripper in typed[ReducedNode.GRIPPERS]:
        gripper_edges = [
            node
            for node, _ in goal.in_edges(gripper)
            if node == ReducedNode.FREE or not isinstance(node, ReducedNode)
        ]
        if not gripper_edges:
            underspecified_grippers.add(gripper)

    return (
        underspecified_balls,
        underspecified_grippers,
        goal.out_degree(ReducedNode.ROBBY) == 0,
    )


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
        case _:
            raise ValueError(f"Domain {domain} not supported.")


def _detached_blocks(
    nodesA: set[str],
    nodesB: set[str],
    scene: ReducedSceneGraph,
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
            if (
                not rx.has_path(scene.graph, a_index, b_index)
                and not rx.has_path(scene.graph, b_index, a_index)
                and a != b
            ):
                _nodesA.discard(a)
                _nodesB.discard(b)

    return _nodesA, _nodesB


def _fully_specify_blocksworld(
    scene: ReducedSceneGraph,
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
            graph.PlanGraphEdge(predicate="clear", scene=scene.scene),
        )
    for block in bottom_blocks_:
        scene.add_edge(
            block,
            ReducedNode.TABLE,
            graph.PlanGraphEdge(predicate="on-table", scene=scene.scene),
        )

    # handle arm
    if arm_empty and not (top_blocks & bottom_blocks):
        scene.add_edge(
            ReducedNode.CLEAR,
            ReducedNode.ARM,
            graph.PlanGraphEdge(predicate="arm-empty", scene=scene.scene),
        )

    return scene


def _fully_specify_gripper(
    init: ReducedSceneGraph,
    goal: ReducedSceneGraph,
) -> ReducedSceneGraph:
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

    # bring "typing" predicates from init to goal
    typed_objects = _gripper_get_typed_objects(init)
    for typing, objects in typed_objects.items():
        for obj in objects:
            edge = graph.PlanGraphEdge(predicate=typing.value, scene=graph.Scene.GOAL)
            edge_ = graph.PlanGraphEdge(predicate=typing.value)
            if obj in scene.nodes and not (
                scene.has_edge(typing, obj, edge) or scene.has_edge(typing, obj, edge_)
            ):
                scene.add_edge(typing, obj, edge)

    underspecified_balls, underspecified_grippers, _ = _gripper_underspecified_blocks(
        init,
        goal,
    )

    if underspecified_grippers and not underspecified_balls:
        for gripper in underspecified_grippers:
            scene.add_edge(
                ReducedNode.FREE,
                gripper,
                graph.PlanGraphEdge(predicate="free", scene=scene.scene),
            )

    return scene


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
    reduced_init, reduced_goal = reduce(problem).decompose()

    match domain:
        case "blocksworld":
            fully_specified_goal = _fully_specify_blocksworld(reduced_goal)
        case "gripper":
            fully_specified_goal = _fully_specify_gripper(
                reduced_init,
                reduced_goal,
            )
        case _:
            raise ValueError(f"Domain {domain} not supported.")

    if return_reduced:
        return ReducedProblemGraph.join(reduced_init, fully_specified_goal)
    else:
        init, _ = problem.decompose()
        return graph.ProblemGraph.join(
            init,
            inflate(fully_specified_goal, domain=domain),
        )


def reduce(
    graph: graph.SceneGraph,
    domain: str | None = None,
    validate: bool = True,
) -> ReducedSceneGraph | ReducedProblemGraph:
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
