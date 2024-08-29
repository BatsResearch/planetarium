from typing import Any

from collections import defaultdict
import copy

import rustworkx as rx

from planetarium import graph
from ..reduced_graph import (
    ReducedProblemGraph,
    ReducedSceneGraph,
    ReducedNode,
    ReducedNodes,
)


def _reduced_rover_add_predicate(
    reduced: graph.SceneGraph | graph.ProblemGraph,
    predicate: dict[str, Any],
):
    """Add a predicate to a reduced rover scene graph.

    Args:
        reduced (graph.SceneGraph | graph.ProblemGraph): The scene graph to add
            the predicate to.
        predicate (dict[str, Any]): The predicate to add.
    """
    params = predicate["parameters"]
    reduced_edge = graph.PlanGraphEdge(
        predicate=predicate["typing"],
        scene=predicate.get("scene"),
    )

    match (predicate["typing"], len(params)):
        case ("at", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("at_lander", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("can_traverse", 3):
            predicate_node = graph.PlanGraphNode(
                f"{params[0]}-{params[1]}-{params[2]}",
                name="can_traverse",
                label=graph.Label.PREDICATE,
                typing="can_traverse",
                scene=predicate.get("scene"),
            )
            reduced.add_node(predicate_node)
            for i, param in enumerate(params):
                pos_edge = copy.deepcopy(reduced_edge)
                pos_edge.position = i
                reduced.add_edge(predicate_node, param, pos_edge)

        case ("equipped_for_soil_analysis", 1):
            reduced.add_edge(ReducedNode.SOIL_ANALYSIS, params[0], reduced_edge)
        case ("equipped_for_rock_analysis", 1):
            reduced.add_edge(ReducedNode.ROCK_ANALYSIS, params[0], reduced_edge)
        case ("equipped_for_imaging", 1):
            reduced.add_edge(ReducedNode.IMAGING, params[0], reduced_edge)
        case ("empty", 1):
            reduced.add_edge(ReducedNode.EMPTY, params[0], reduced_edge)
        case ("full", 1):
            reduced.add_edge(ReducedNode.FULL, params[0], reduced_edge)
        case ("have_rock_analysis", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("have_soil_analysis", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("calibrated", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("supports", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("available", 1):
            reduced.add_edge(ReducedNode.AVAILABLE, params[0], reduced_edge)
        case ("visible", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("have_image", 3):
            predicate_node = graph.PlanGraphNode(
                f"{params[0]}-{params[1]}-{params[2]}",
                name="have_image",
                label=graph.Label.PREDICATE,
                typing="have_image",
                scene=predicate.get("scene"),
            )
            reduced.add_node(predicate_node)
            for i, param in enumerate(params):
                pos_edge = copy.deepcopy(reduced_edge)
                pos_edge.position = i
                reduced.add_edge(predicate_node, param, pos_edge)
        case ("communicated_soil_data", 1):
            reduced.add_edge(
                ReducedNode.COMMUNICATED_SOIL_DATA, params[0], reduced_edge
            )
        case ("communicated_rock_data", 1):
            reduced.add_edge(
                ReducedNode.COMMUNICATED_ROCK_DATA, params[0], reduced_edge
            )
        case ("communicated_image_data", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("at_soil_sample", 1):
            reduced.add_edge(ReducedNode.AT_SOIL_SAMPLE, params[0], reduced_edge)
        case ("at_rock_sample", 1):
            reduced.add_edge(ReducedNode.AT_ROCK_SAMPLE, params[0], reduced_edge)
        case ("visible_from", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("store_of", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("calibration_target", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("on_board", 2):
            reduced.add_edge(params[0], params[1], reduced_edge)
        case ("channel_free", 1):
            reduced.add_edge(ReducedNode.CHANNEL_FREE, params[0], reduced_edge)


def _reduce_rover(
    scene: graph.SceneGraph | graph.ProblemGraph,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a rover scene graph to a Directed Acyclic Graph.

    Args:
        scene (graph.SceneGraph | graph.ProblemGraph): The scene graph to reduce.

    Raises:
        ValueError: If the scene graph is not a Directed Acyclic Graph.

    Returns:
        ReducedSceneGraph | ReducedProblemGraph: The reduced problem graph.
    """
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
        _reduced_rover_add_predicate(reduced, predicate)

    return reduced


def _inflate_rover(
    scene: ReducedSceneGraph | ReducedProblemGraph,
) -> graph.SceneGraph | graph.ProblemGraph:
    """Respecify a rover scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node in scene.nodes:
        if (
            not isinstance(node.node, ReducedNode)
            and node.label == graph.Label.CONSTANT
        ):
            # add constants
            constants.append({"name": node.node, "typing": node.typing})
        elif node.label == graph.Label.PREDICATE and node.typing in (
            "can_traverse",
            "have_image",
        ):
            # add multi-arity predicates
            params = scene.out_edges(node)
            params.sort(key=lambda x: x[1].position)

            predicates.append(
                {
                    "typing": node.typing,
                    "parameters": [param.node for param, _ in params],
                    "scene": node.scene,
                }
            )

    for u, v, edge in scene.edges:
        match (u, v):
            case (predicate, _) if predicate.node in ReducedNodes["rover"]:
                predicates.append(
                    {
                        "typing": predicate.node.value,
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (
                predicate,
                _,
            ) if predicate.label == graph.Label.PREDICATE and predicate.typing in (
                "can_traverse",
                "have_image",
            ):
                # add separately to handle multi-arity predicates
                pass
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
            domain="rover",
            requirements=scene._requirements,
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="rover",
            scene=scene.scene,
            requirements=scene._requirements,
        )


def _rover_get_unchangeable_predicates(
    scene: ReducedSceneGraph,
) -> list[dict[str, Any]]:
    """Get the unchangeable predicates in a rover scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        list[dict[str, Any]]: The unchangeable predicates in the scene graph.
    """
    predicates = []

    for node in scene.nodes:
        # ternary unchangeable predicates
        if node.label == graph.Label.PREDICATE and node.typing in (
            "can_traverse",
            "have_image",
        ):
            params = scene.out_edges(node)
            params.sort(key=lambda x: x[1].position)
            predicates.append(
                {
                    "typing": node.node,
                    "parameters": [param.node for param, _ in params],
                }
            )

    for u, v, edge in scene.edges:
        match edge:
            # unary unchangeable predicates
            case pred if pred.predicate in (
                "equipped_for_soil_analysis",
                "equipped_for_rock_analysis",
                "equipped_for_imaging",
                "available",  # changes but always changes back
                "channel_free",  # changes but always changes back
                "communicated_soil_data",
                "communicated_rock_data",
                "at_soil_sample",
            ):
                predicates.append(
                    {
                        "typing": pred.predicate,
                        "parameters": [v.node],
                    }
                )
            # binary unchangeable predicates
            case pred if pred.predicate in (
                "supports",
                "communicated_image_data",
                "visible_from",
                "store_of",
                "calibration_target",
                "on_board",
                "at_lander",
            ):
                predicates.append(
                    {
                        "typing": pred.predicate,
                        "parameters": [u.node, v.node],
                    }
                )

    return predicates


def _rover_get_typed_objects(
    scene: ReducedSceneGraph,
) -> dict[
    ReducedNode,
    set[graph.PlanGraphNode]
    | dict[
        graph.PlanGraphNode,
        set[graph.PlanGraphNode],
    ],
]:
    """Get the typed objects in a rover scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        dict[ReducedNode, set[graph.PlanGraphNode] | dict[graph.PlanGraphNode,
            set[graph.PlanGraphNode]]]: The typed objects in the scene graph.
    """
    rover: set[graph.PlanGraphNode] = set()
    equipped_for_soil: set[graph.PlanGraphNode] = set()
    equipped_for_rock: set[graph.PlanGraphNode] = set()
    equipped_for_imaging: set[graph.PlanGraphNode] = set()
    has_cameras = defaultdict(set)
    has_stores = defaultdict(set)

    waypoints: set[graph.PlanGraphNode] = set()
    stores: set[graph.PlanGraphNode] = set()
    cameras: set[graph.PlanGraphNode] = set()
    modes: set[graph.PlanGraphNode] = set()
    landers: set[graph.PlanGraphNode] = set()
    objectives: set[graph.PlanGraphNode] = set()

    objects: dict[
        str,
        set[graph.PlanGraphNode] | dict[graph.PlanGraphNode, set[graph.PlanGraphNode]],
    ] = {
        "rover": rover,
        "equipped_for_soil_analysis": equipped_for_soil,
        "equipped_for_rock_analysis": equipped_for_rock,
        "equipped_for_imaging": equipped_for_imaging,
        "has_cameras": has_cameras,
        "has_stores": has_stores,
        "waypoint": waypoints,
        "store": stores,
        "camera": cameras,
        "mode": modes,
        "lander": landers,
        "objective": objectives,
    }

    for node in scene.nodes:
        if node.label == graph.Label.CONSTANT:
            for typing in node.typing:
                objects[typing].add(node)
            if node.typing == "rover":
                if scene.has_edge(ReducedNode.SOIL_ANALYSIS, node):
                    equipped_for_soil.add(node)
                if scene.has_edge(ReducedNode.ROCK_ANALYSIS, node):
                    equipped_for_rock.add(node)
                if scene.has_edge(ReducedNode.IMAGING, node):
                    equipped_for_imaging.add(node)
            elif node.typing == "camera":
                rover = [
                    r
                    for r, edge in scene.in_edges(node)
                    if edge.predicate == "on_board"
                ]
                for rover in rover:
                    has_cameras[rover].add(node)
            elif node.typing == "store":
                rover = [
                    r
                    for r, edge in scene.in_edges(node)
                    if edge.predicate == "store_of"
                ]
                for rover in rover:
                    has_stores[rover].add(node)

    return objects


def _rover_subgraph(
    scene: ReducedSceneGraph,
    rover: graph.PlanGraphNode,
) -> ReducedSceneGraph:
    """Get the subgraph of a specific rover in a rover scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.
        rover (graph.PlanGraphNode): The rover to get the subgraph of.

    Returns:
        ReducedSceneGraph: The subgraph for the rover, where each waypoint is
            traversable by the rover.
    """
    objects = _rover_get_typed_objects(scene)
    subgraph = ReducedSceneGraph(
        constants=[],
        domain="ROVER_SUBGRAPH",
        scene=scene.scene,
        requirements=(*scene._requirements, "ROVER_SUBGRAPH"),
    )

    for node in [
        rover,
        *objects["waypoint"],
        *objects["objective"],
        *objects["lander"],
    ]:
        subgraph.add_node(copy.deepcopy(node))

    # take all edges related to the above and copy it over
    for u, v, edge in scene.edges:
        if u in subgraph.nodes and v in subgraph.nodes:
            if edge.predicate == "at_lander":
                for way, in_edge in scene.in_edges(u):
                    if in_edge.predicate == "visible":
                        subgraph.add_edge(way, u)
            elif edge.predicate == "visible_from":
                for way, in_edge in scene.in_edges(u):
                    if in_edge.predicate == "visible_from":
                        subgraph.add_edge(way, u)

    # can_traverse is encoded in predicate nodes
    for node in scene.nodes:
        if node.label == graph.Label.PREDICATE and node.typing == "can_traverse":
            params = sorted(scene.out_edges(node), key=lambda x: x[1].position)
            params = [p for p, _ in params]
            if params[0] == rover:
                reduced_edge = graph.PlanGraphEdge(
                    predicate=node.typing,
                    scene=scene.scene,
                )
                # if visible edge exists, add can_traverse edge
                edges = [edge for u, v, edge in scene.edges if edge.predicate == "visible"]
                if edges:
                    subgraph.add_edge(params[1], params[2], reduced_edge)

    return subgraph


def _fully_specify_rover(
    inflated_init: graph.SceneGraph,
    inflated_goal: graph.SceneGraph,
) -> tuple[ReducedSceneGraph, ReducedSceneGraph]:
    """Fully specifies a rover scene graph.

    Args:
        init (graph.SceneGraph): The reduced SceneGraph of the initial scene.
        goal (graph.SceneGraph): The reduced SceneGraph of the goal scene.

    Returns:
        ReducedSceneGraph: The fully specified scene graph.
    """
    # NOTE: we assume bidirectional traversal and visibility
    # Removing this assumption would require bfs planning for this domain.
    init = _reduce_rover(inflated_init)

    # bring unchangeable predicates from init to goal
    unchangeable_predicates = _rover_get_unchangeable_predicates(init)
    for pred in unchangeable_predicates:
        if pred not in inflated_goal.predicates:
            inflated_goal._add_predicate(pred, scene=graph.Scene.GOAL)

    objects = _rover_get_typed_objects(init)
    # if there is only one waypoint, everyone must be at it in the goal (if they are at it in the init)
    if len(objects["waypoint"]) == 1:
        (waypoint,) = objects["waypoint"]
        for rover in objects["rover"]:
            pred = {
                "predicate": "at",
                "parameters": [rover.name, waypoint.name],
            }
            init_pred = dict(**pred, scene=graph.Scene.INIT)
            goal_pred = dict(**pred, scene=graph.Scene.GOAL)
            if (
                goal_pred not in inflated_goal.predicates
                and init_pred in inflated_init.predicates
            ):
                inflated_goal._add_predicate(goal_pred, scene=graph.Scene.GOAL)
        for lander in objects["lander"]:
            pred = {
                "predicate": "at_lander",
                "parameters": [lander.name, waypoint.name],
            }
            init_pred = dict(**pred, scene=graph.Scene.INIT)
            goal_pred = dict(**pred, scene=graph.Scene.GOAL)
            if (
                goal_pred not in inflated_goal.predicates
                and init_pred in inflated_init.predicates
            ):
                inflated_goal._add_predicate(goal_pred, scene=graph.Scene.GOAL)

    # Handle rock, soil, and image data:
    # - if data has been communicated, and there is only rover that
    # could have communicated it, then we need to add this to the fully specified goal
    # - if data has been communicated, and there is more than one rover that
    # could have communicated it, then we can't decide who communicated it, and
    # we don't have anything to add to the fully specified goal

    rover_maps = {r: _rover_subgraph(init, r) for r in objects["rover"]}

    # for every soil sample that needs collecting/communicating:
    # check if we can attach to a specific rover
    def rover_can_reach_and_communicate(rover, waypoint):
        # can this rover reach the sample
        reachable = False
        communicable = False
        if reachable := rx.has_path(rover_maps[rover].graph, rover, waypoint):
            # can this rover, from the waypoint, communicate to a lander?
            for lander in objects["lander"]:
                if communicable := rx.has_path(
                    rover_maps[rover].graph, waypoint, lander
                ):
                    break

        return reachable and communicable

    goal = _reduce_rover(inflated_goal)
    # soil and rock
    for analysis_type, have, reduced_node in (
        ("soil_analysis", "soil_sample", ReducedNode.COMMUNICATED_SOIL_DATA),
        ("rock_analysis", "rock_sample", ReducedNode.COMMUNICATED_ROCK_DATA),
    ):
        for waypoint in goal.successors(reduced_node):
            if waypoint in init.successors(reduced_node):
                continue
            equipped_rover = objects[f"equipped_for_{analysis_type}"]
            # only equipped rover with storage can communicate
            equipped_rover = [
                rover for rover in equipped_rover if objects["has_stores"][rover]
            ]
            # If a store is not specified as full or empty in the init, we can't
            # use it to communicate data (it might be full or empty).
            equipped_rover = [
                rover
                for rover in equipped_rover
                if any(
                    store
                    for store in objects["has_stores"][rover]
                    if init.has_edge(ReducedNode.EMPTY, store)
                    or init.has_edge(ReducedNode.FULL, store)
                )
            ]
            valid_rover: list[graph.PlanGraphNode] = []
            for rover in equipped_rover:
                if rover_can_reach_and_communicate(rover, waypoint):
                    valid_rover.append(rover)
                    if len(valid_rover) > 1:
                        break

            if len(valid_rover) == 1:
                inflated_goal._add_predicate(
                    {
                        "predicate": f"have_{have}",
                        "parameters": [valid_rover[0].node],
                        "scene": graph.Scene.GOAL,
                    }
                )
    # images
    for u, v, edge in goal.edges:
        if edge.predicate == "have_image":
            if init.has_edge(u, v, edge):
                continue
            valid_rover: list[graph.PlanGraphNode] = []
            for rover in objects[f"equipped_for_imaging"]:
                if rover_can_reach_and_communicate(rover, v):
                    valid_rover.append(rover)
                    if len(valid_rover) > 1:
                        break

            if len(valid_rover) == 1:
                inflated_goal._add_predicate(
                    {
                        "predicate": "have_image",
                        "parameters": [valid_rover[0].node, u.node, v.node],
                        "scene": graph.Scene.GOAL,
                    }
                )
    print('at the end', [u.predicate for *_, u in goal.edges])
    return init, goal
