from collections import defaultdict
import copy

from pddl.core import Action

from planetarium import graph
from ..reduced_graph import ReducedSceneGraph, ReducedProblemGraph, ReducedNode


# Add the ReducedNode enum for the gripper domain
ReducedNode.register(
    {
        "ROOMS": "room",
        "BALLS": "ball",
        "GRIPPERS": "gripper",
        "ROBBY": "at-robby",
        "FREE": "free",
    },
    "gripper",
)


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
