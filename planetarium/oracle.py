from typing import Any

from collections import defaultdict
import copy
import enum

import jinja2 as jinja
from pddl.core import Action
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

plan_template = jinja.Template(
    """
    {%- for action in actions -%}
    ({{ action.name }} {{ action.parameters | join(", ") }})
    {% endfor %}
    """
)


class ReducedSceneGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str,
        scene: graph.Scene | None = None,
        requirements: tuple[str] = (),
    ):
        super().__init__(constants, domain=domain, requirements=requirements)
        self.scene = scene

        for e in ReducedNodes[domain]:
            predicate = e.value
            self.add_node(
                graph.PlanGraphNode(
                    e,
                    name=predicate,
                    label=graph.Label.PREDICATE,
                    typing=predicate,
                ),
            )


class ReducedProblemGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str,
        requirements: tuple[str] = (),
    ):
        super().__init__(constants, domain=domain, requirements=requirements)

        for e in ReducedNodes[domain]:
            predicate = e.value
            self.add_node(
                graph.PlanGraphNode(
                    e,
                    name=predicate,
                    label=graph.Label.PREDICATE,
                    typing=predicate,
                ),
            )

    def decompose(self) -> tuple[ReducedSceneGraph, ReducedSceneGraph]:
        init = ReducedSceneGraph(
            self.constants,
            self.domain,
            scene=graph.Scene.INIT,
            requirements=self._requirements,
        )
        goal = ReducedSceneGraph(
            self.constants,
            self.domain,
            scene=graph.Scene.GOAL,
            requirements=self._requirements,
        )

        for u, v, edge in self.edges:
            edge = copy.deepcopy(edge)
            if edge.scene == graph.Scene.INIT:
                init.add_edge(u, v, edge)
            elif edge.scene == graph.Scene.GOAL:
                goal.add_edge(u, v, edge)

        return init, goal

    @staticmethod
    def join(init: ReducedSceneGraph, goal: ReducedSceneGraph) -> "ReducedProblemGraph":
        problem = ReducedProblemGraph(
            init.constants,
            domain=init.domain,
            requirements=init._requirements,
        )

        for u, v, edge in init.edges:
            edge = copy.deepcopy(edge)
            problem.add_edge(u, v, edge)
            edge.scene = graph.Scene.INIT
        for u, v, edge in goal.edges:
            edge = copy.deepcopy(edge)
            edge.scene = graph.Scene.GOAL
            problem.add_edge(u, v, edge)

        return problem


class DomainNotSupportedError(Exception):
    pass


def _reduce_blocksworld(
    scene: graph.SceneGraph | graph.ProblemGraph,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a blocksworld scene graph to a Directed Acyclic Graph.

    Args:
        problem (graph.SceneGraph | graph.ProblemGraph): The scene graph to
            reduce.

    Returns:
        ReducedGraph: The reduced problem graph.
    """

    nodes = defaultdict(list)
    for node in scene.nodes:
        nodes[node.label].append(node)

    match scene:
        case graph.ProblemGraph(
            _constants=constants,
            _predicates=predicates,
            _domain=domain,
            _requirements=requirements,
        ):
            reduced = ReducedProblemGraph(
                constants=constants,
                domain=domain,
                requirements=requirements,
            )
        case graph.SceneGraph(
            constants=constants,
            _predicates=predicates,
            scene=scene,
            _domain=domain,
            _requirements=requirements,
        ):
            reduced = ReducedSceneGraph(
                constants=constants,
                domain=domain,
                scene=scene,
                requirements=requirements,
            )
        case _:
            raise ValueError("Scene must be a SceneGraph or ProblemGraph.")

    for predicate in predicates:
        params = predicate["parameters"]
        reduced_edge = graph.PlanGraphEdge(
            predicate=predicate["typing"],
            scene=predicate.get("scene"),
        )
        match (predicate["typing"], len(params)):
            case ("arm-empty", 0):
                reduced.add_edge(ReducedNode.CLEAR, ReducedNode.ARM, reduced_edge)
            case ("on-table", 1):
                reduced.add_edge(params[0], ReducedNode.TABLE, reduced_edge)
            case ("clear", 1):
                reduced.add_edge(ReducedNode.CLEAR, params[0], reduced_edge)
            case ("on", 2):
                reduced.add_edge(params[0], params[1], reduced_edge)
            case ("holding", 1):
                reduced.add_edge(params[0], ReducedNode.ARM, reduced_edge)
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
    if scene.scene == graph.Scene.INIT:
        for node in scene.nodes:
            if not isinstance(node.node, ReducedNode):
                if scene.in_degree(node.node) != 1 or scene.out_degree(node.node) != 1:
                    # only case this is allowed is if the object is in the hand
                    if not scene.has_edge(node, ReducedNode.ARM):
                        raise ValueError(
                            f"Node {node} does not have top or bottom behavior defined."
                        )
    for node in scene.nodes:
        if (node.node != ReducedNode.TABLE and scene.in_degree(node.node) > 1) or (
            node.node != ReducedNode.CLEAR and scene.out_degree(node.node) > 1
        ):
            raise ValueError(
                f"Node {node} has multiple parents/children. (not possible in blocksworld)."
            )
        if scene.in_degree(ReducedNode.ARM) == 1:
            obj = scene.predecessors(ReducedNode.ARM)[0]
            if obj.node != ReducedNode.CLEAR and scene.in_degree(obj) > 0:
                raise ValueError("Object on arm is connected to another object.")


def _reduce_gripper(
    scene: graph.SceneGraph | graph.ProblemGraph,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a gripper scene graph to a Directed Acyclic Graph.

    Args:
        scene (graph.SceneGraph): The scene graph to reduce.

    Returns:
        ReducedGraph: The reduced problem graph.
    """
    nodes = defaultdict(list)
    for node in scene.nodes:
        nodes[node.label].append(node)

    match scene:
        case graph.ProblemGraph(
            _constants=constants,
            _predicates=predicates,
            _domain=domain,
            _requirements=requirements,
        ):
            reduced = ReducedProblemGraph(
                constants=constants,
                domain=domain,
                requirements=requirements,
            )
        case graph.SceneGraph(
            constants=constants,
            _predicates=predicates,
            scene=scene,
            _domain=domain,
            _requirements=requirements,
        ):
            reduced = ReducedSceneGraph(
                constants=constants,
                domain=domain,
                scene=scene,
                requirements=requirements,
            )
        case _:
            raise ValueError("Scene must be a SceneGraph or ProblemGraph.")

    for predicate in predicates:
        params = predicate["parameters"]
        reduced_edge = graph.PlanGraphEdge(
            predicate=predicate["typing"],
            scene=predicate.get("scene"),
        )
        match (predicate["typing"], len(params)):
            case ("at-robby", 1):
                reduced.add_edge(ReducedNode.ROBBY, params[0], reduced_edge)
            case ("free", 1):
                reduced.add_edge(ReducedNode.FREE, params[0], reduced_edge)
            case ("ball", 1):
                reduced.add_edge(ReducedNode.BALLS, params[0], reduced_edge)
            case ("gripper", 1):
                reduced.add_edge(ReducedNode.GRIPPERS, params[0], reduced_edge)
            case ("room", 1):
                reduced.add_edge(ReducedNode.ROOMS, params[0], reduced_edge)
            case ("at", 2):
                reduced.add_edge(params[0], params[1], reduced_edge)
            case ("carry", 2):
                reduced.add_edge(params[0], params[1], reduced_edge)

    return reduced


def reduce(
    graph: graph.SceneGraph,
    domain: str | None = None,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a scene graph to a Directed Acyclic Graph.

    Args:
        graph (graph.SceneGraph): The scene graph to reduce.
        domain (str, optional): The domain of the scene graph. Defaults to
            "blocksworld".

    Returns:
        ReducedGraph: The reduced problem graph.
    """
    domain = domain or graph.domain
    match domain:
        case "blocksworld":
            return _reduce_blocksworld(graph)
        case "gripper":
            return _reduce_gripper(graph)
        case _:
            raise DomainNotSupportedError(f"Domain {domain} not supported.")


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
        match (u.node, v.node):
            case (ReducedNode.CLEAR, ReducedNode.ARM):
                predicates.append(
                    {
                        "typing": "arm-empty",
                        "parameters": [],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.CLEAR, _):
                predicates.append(
                    {
                        "typing": "clear",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (_, ReducedNode.TABLE):
                predicates.append(
                    {
                        "typing": "on-table",
                        "parameters": [u.node],
                        "scene": edge.scene,
                    }
                )
            case (_, ReducedNode.ARM):
                predicates.append(
                    {
                        "typing": "holding",
                        "parameters": [u.node],
                        "scene": edge.scene,
                    }
                )
            case (_, _):
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
            requirements=scene._requirements,
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="blocksworld",
            scene=scene.scene,
            requirements=scene._requirements,
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
        match (u.node, v.node):
            case (ReducedNode.ROBBY, _):
                predicates.append(
                    {
                        "typing": "at-robby",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.FREE, _):
                predicates.append(
                    {
                        "typing": "free",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.BALLS, _):
                predicates.append(
                    {
                        "typing": "ball",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.GRIPPERS, _):
                predicates.append(
                    {
                        "typing": "gripper",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.ROOMS, _):
                predicates.append(
                    {
                        "typing": "room",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (_, _):
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
            requirements=scene._requirements,
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="gripper",
            scene=scene.scene,
            requirements=scene._requirements,
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
            raise DomainNotSupportedError(f"Domain {domain} not supported.")


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
            raise DomainNotSupportedError(f"Domain {domain} not supported.")

    if return_reduced:
        return ReducedProblemGraph.join(reduced_init, fully_specified_goal)
    else:
        init, _ = problem.decompose()
        return graph.ProblemGraph.join(
            init,
            inflate(fully_specified_goal, domain=domain),
        )


def _plan_blocksworld(problem: ReducedProblemGraph) -> list[Action]:
    init, goal = problem.decompose()
    actions = []

    # Process init scene
    # check if arm is empty
    if (
        not init.has_edge(ReducedNode.CLEAR, ReducedNode.ARM)
        and init.in_degree(ReducedNode.ARM) == 1
    ):
        obj = init.predecessors(ReducedNode.ARM)[0]
        actions.append(Action("putdown", [obj.name]))

    # unstack everything in init
    for idx in rx.topological_sort(init.graph):
        node = init.nodes[idx]
        if isinstance(node.node, ReducedNode):
            continue
        elif init.successors(node)[0].name in (ReducedNode.ARM, ReducedNode.TABLE):
            # if the block is on the table or in the arm, ignore it
            continue
        else:
            actions.append(
                Action("unstack", [node.name, init.successors(node)[0].name])
            )
            actions.append(Action("putdown", [node.name]))

    # Process goal scene
    # stack everything in goal
    for idx in reversed(rx.topological_sort(goal.graph)):
        node = goal.nodes[idx]
        if isinstance(node.node, ReducedNode):
            continue
        elif goal.out_degree(node.node) == 0:
            # isn't defined to be on anything (keep on table)
            continue
        elif goal.successors(node)[0].node in (ReducedNode.ARM, ReducedNode.TABLE):
            # if the block is on the table or in the arm, ignore it
            continue
        else:
            actions.append(Action("pickup", [node.name]))
            actions.append(Action("stack", [node.name, goal.successors(node)[0].name]))

    # Check if arm should be holding it
    if (
        not goal.has_edge(ReducedNode.CLEAR, ReducedNode.ARM)
        and goal.in_degree(ReducedNode.ARM) == 1
    ):
        obj = goal.predecessors(ReducedNode.ARM)[0]
        actions.append(Action("pickup", [obj.name]))

    return actions


def _plan_gripper(problem: ReducedProblemGraph) -> list[Action]:
    # TODO: this function is not "complete": it does not handle all cases
    # - multiple "types" per object
    # - robby not at a room (can be valid in a few cases)
    # - balls not in rooms
    # - objects without typing

    init, goal = problem.decompose()
    actions = []

    # Process init scene
    typed = _gripper_get_typed_objects(init)
    rooms = list(typed[ReducedNode.ROOMS])
    grippers = list(typed[ReducedNode.GRIPPERS])

    # get current room
    if init.out_degree(ReducedNode.ROBBY) < 1:
        return actions

    current_room = init.successors(ReducedNode.ROBBY)[0]
    # move to first room
    if current_room != rooms[0]:
        actions.append(Action("move", [current_room.name, rooms[0].name]))

    # ensure all grippers are free
    for gripper in grippers:
        if not init.has_edge(ReducedNode.FREE, gripper):
            # get in_edge
            ball = [
                b for b in init.predecessors(gripper) if b in typed[ReducedNode.BALLS]
            ]
            if ball:
                actions.append(
                    Action("drop", [ball[0].name, rooms[0].name, gripper.name])
                )

    # move all balls to first room
    for room in rooms:
        for obj in init.predecessors(room):
            if obj in typed[ReducedNode.BALLS]:
                actions.append(Action("move", [rooms[0].name, room.name]))
                actions.append(Action("pick", [obj.name, room.name, grippers[0].name]))
                actions.append(Action("move", [room.name, rooms[0].name]))
                actions.append(
                    Action("drop", [obj.name, rooms[0].name, grippers[0].name])
                )

    # Process goal scene
    for room in rooms:
        for obj in goal.predecessors(room):
            if obj in typed[ReducedNode.BALLS]:
                actions.append(
                    Action("pick", [obj.name, rooms[0].name, grippers[0].name])
                )
                actions.append(Action("move", [rooms[0].name, room.name]))
                actions.append(Action("drop", [obj.name, room.name, grippers[0].name]))
                actions.append(Action("move", [room.name, rooms[0].name]))

    # pick up balls in first room tied to grippers
    for gripper in grippers:
        for ball in typed[ReducedNode.BALLS]:
            if goal.has_edge(ball, gripper):
                actions.append(Action("pick", [ball.name, rooms[0].name, gripper.name]))

    # move to room with robby
    goal_room = next(iter(goal.successors(ReducedNode.ROBBY)), None)
    if goal_room:
        actions.append(Action("move", [rooms[0].name, goal_room.name]))

    return actions


def plan(problem: graph.ProblemGraph, domain: str | None = None) -> list[Action]:
    """Plans a sequence of actions to solve a problem.

    Args:
        problem (graph.ProblemGraph): The problem to plan for.

    Returns:
        str: The sequence of actions to solve the problem.
    """
    domain = domain or problem.domain
    try:
        problem = fully_specify(problem, domain=domain, return_reduced=True)
        match domain:
            case "blocksworld":
                return _plan_blocksworld(problem)
            case "gripper":
                return _plan_gripper(problem)
            case _:
                raise DomainNotSupportedError(f"Domain {domain} not supported.")
    except Exception:
        return []


def plan_to_string(actions: list[Action]) -> str:
    """Converts a list of actions to a string.

    Args:
        actions (list[Action]): The list of actions to convert.

    Returns:
        str: The string representation of the actions.
    """
    return plan_template.render(actions=actions)
