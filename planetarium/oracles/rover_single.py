from typing import Any

import copy
from itertools import permutations

import rustworkx as rx

from planetarium import graph
from ..reduced_graph import ReducedProblemGraph, ReducedSceneGraph, ReducedNode

ReducedNode.register(
    {
        "ROVER": "rover",
        "LANDER": "lander",
        "EMPTY": "empty",
        "FULL": "full",
        "AVAILABLE": "available",
        "AT_ROCK_SAMPLE": "at_rock_sample",
        "AT_SOIL_SAMPLE": "at_soil_sample",
        "HAVE_ROCK_ANALYSIS": "have_rock_analysis",
        "HAVE_SOIL_ANALYSIS": "have_soil_analysis",
        "COMMUNICATED_SOIL_DATA": "communicated_soil_data",
        "COMMUNICATED_ROCK_DATA": "communicated_rock_data",
        "CHANNEL_FREE": "channel_free",
    },
    "rover-single",
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

    match (predicate["typing"], params):
        case ("at_rover", [waypoint]):
            reduced.add_edge(ReducedNode.ROVER, waypoint, reduced_edge)
        case ("at_lander", [waypoint]):
            reduced.add_edge(ReducedNode.LANDER, waypoint, reduced_edge)
        case ("available", []):
            reduced.add_edge(
                ReducedNode.AVAILABLE,
                ReducedNode.ROVER,
                reduced_edge,
            )
        case ("channel_free", []):
            reduced.add_edge(
                ReducedNode.CHANNEL_FREE,
                ReducedNode.LANDER,
                reduced_edge,
            )
        case ("visible_from", [x, y]):
            reduced.add_edge(y, x, reduced_edge)
        case (predicate_type, [x]):
            Reduced = getattr(ReducedNode, predicate_type.upper())
            reduced.add_edge(Reduced, x, reduced_edge)
        case (predicate_type, [x, y]):
            reduced.add_edge(x, y, reduced_edge)


def _reduce_rover_single(
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


def _inflate_rover_single(
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

    for u, v, edge in scene.edges:
        match (u.node, edge.predicate):
            case (_, pred) if pred in ("available", "channel_free"):
                predicates.append(
                    {
                        "typing": edge.predicate,
                        "parameters": [],
                        "scene": edge.scene,
                    }
                )
            case (_, pred) if pred in ("at_rover", "at_lander"):
                predicates.append(
                    {
                        "typing": edge.predicate,
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (node, _) if isinstance(node, ReducedNode):
                predicates.append(
                    {
                        "typing": node.value,
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case _:
                if edge.predicate == "visible_from":
                    v, u = u, v
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
            domain="rover-single",
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

    for u, v, edge in scene.edges:
        match edge.predicate:
            case pred if pred in ("available", "channel_free"):
                # all actions that affect these predicates always set them back
                predicates.append(
                    {
                        "typing": pred,
                        "parameters": [],
                    }
                )
            # unary unchangeable predicates
            case pred if pred in (
                "communicated_soil_data",
                "communicated_rock_data",
                "at_lander",  # landers don't move
                "have_soil_analysis",
                "have_rock_analysis",
            ):
                predicates.append(
                    {
                        "typing": pred,
                        "parameters": [v.node],
                    }
                )
            # binary unchangeable predicates
            case pred if pred in (
                "supports",
                "communicated_image_data",
                "visible_from",
                "store_of",
                "on_board",
                "have_image",
                "visible",
                "can_traverse",
            ):
                if pred == "visible_from":
                    v, u = u, v
                predicates.append(
                    {
                        "typing": pred,
                        "parameters": [u.node, v.node],
                    }
                )

    return predicates


def rover_subgraph(scene: ReducedSceneGraph) -> ReducedSceneGraph:
    """Get the subgraph of places the rover can reach.

    Args:
        scene (ReducedSceneGraph): The scene graph to get the subgraph of.

    Returns:
        ReducedSceneGraph: The subgraph of places the rover can reach.
    """
    subgraph = ReducedSceneGraph(
        constants=[],
        domain="rover-single",
        scene=scene.scene,
        requirements=(*scene._requirements, "ROVER_SUBGRAPH"),
    )

    for node in scene.nodes:
        if node.label == graph.Label.CONSTANT and not isinstance(
            node.node, ReducedNode
        ):
            if node.typing.issubset(
                {"waypoint", "objective", "store", "camera", "mode"}
            ):
                subgraph.add_node(copy.deepcopy(node))

    for u, v, edge in scene.edges:
        match edge.predicate:
            case "can_traverse":
                # add if visible edge also exists
                if scene.has_edge(
                    u,
                    v,
                    graph.PlanGraphEdge(
                        predicate="visible",
                        scene=edge.scene,
                    ),
                ):
                    subgraph.add_edge(u, v, copy.deepcopy(edge))
            case pred if pred in ("visible_from", "at_rover"):
                subgraph.add_edge(u, v, copy.deepcopy(edge))
            case "at_lander":
                subgraph.add_edge(v, u, copy.deepcopy(edge))
                # add visible edges from waypoint to lander
                for waypoint, in_edge in scene.in_edges(v):
                    # print(in_edge, waypoint.node, u.node, "IN EDGE")
                    if in_edge.predicate == "visible":
                        subgraph.add_edge(waypoint, u, copy.deepcopy(in_edge))

    return subgraph

def all_endpoints(
    scene: ReducedSceneGraph,
    path_map: rx.AllPairsMultiplePathMapping,
    sample_waypoints: list[tuple[graph.PlanGraphNode, bool]],
    final_position: graph.PlanGraphNode | None = None,
    needs_lander: bool = False,
) -> set[int]:

    def source_to_dest(source_idx: int, dest_idx: int) -> list[list[int]]:
        source = scene.graph.nodes()[source_idx]
        dest = scene.graph.nodes()[dest_idx]

        if source == dest or dest_idx not in path_map[source_idx]:
            return []

        paths = []
        remove_last = dest.typing in ({"objective"}, "lander")
        for path in path_map[source_idx][dest_idx]:
            path = list(path)
            if remove_last:
                path = path[:-1]
            paths.append(path[1:])

        return paths

    def check_lander(waypoints: list[tuple[graph.PlanGraphNode, bool]]) -> bool:
        if not needs_lander:
            return True
        lander = False
        for waypoint, communicate in waypoints:
            lander = lander or (waypoint == ReducedNode.LANDER)
            if communicate and lander:
                return False
        return True

    def validate_order(
        waypoints: list[graph.PlanGraphNode], current_path: list | None = None
    ) -> list[list[int]]:
        if len(waypoints) == 0:
            return [current_path]
        if current_path is None:
            current_path = [scene._node_lookup[ReducedNode.ROVER][0]]
        start = current_path[-1]

        paths = []
        for path in source_to_dest(start, waypoints[0]):
            paths.extend(validate_order(waypoints[1:], current_path + path))

        return paths

    if needs_lander:
        # lander must be visited
        sample_waypoints.append((ReducedNode.LANDER, False))
    waypoint_order = filter(check_lander, permutations([*sample_waypoints]))
    to_idx = ([scene._node_lookup[w][0] for w, _ in wo] for wo in waypoint_order)
    if final_position:
        final_pos = scene._node_lookup[final_position][0]
        to_idx = ([*idx, final_pos] for idx in to_idx)

    valid_paths = set([tuple(s) for vo in to_idx for s in validate_order(vo)])

    final_states = set(path[-1] for path in valid_paths)

    for state in list(final_states):
        final_states.update(
            [
                dest
                for dest in path_map[state]
                if scene.graph.nodes()[dest].typing == {"waypoint"}
            ]
        )

    return {scene.graph.nodes()[e] for e in final_states}


def _fully_specify_rover_single(
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
    init = _reduce_rover_single(inflated_init)

    # bring all unchangeable predicates to the goal
    unchangeable_predicates = _rover_get_unchangeable_predicates(init)
    for pred in unchangeable_predicates:
        if pred not in inflated_goal.predicates:
            inflated_goal._add_predicate(pred, scene=graph.Scene.GOAL)

    # communicated predicates need have_* predicates
    for pred in inflated_goal.predicates:
        params = pred["parameters"]
        match pred["typing"].split("_"):
            case [_, sample, _] if sample in ("rock", "soil", "image"):
                have_pred = {
                    "typing": f"have_{sample}{'_analysis' if sample != 'image' else ''}",
                    "parameters": params,
                    "scene": graph.Scene.GOAL,
                }
                if have_pred not in inflated_goal.predicates:
                    inflated_goal._add_predicate(
                        have_pred,
                        scene=graph.Scene.GOAL,
                    )

    # all waypoints that must be visited
    sample_waypoints = []
    needs_lander = False
    for pred in inflated_goal.predicates:
        match pred["typing"].split("_"):
            case ["have", sample, *_] if sample in ("rock", "soil", "image"):
                print(pred)
                waypoint = pred["parameters"][0]
                # check if the rover needs to communicate the data
                com_pred = {
                    "typing": f"communicated_{sample}_data",
                    "parameters": pred["parameters"],
                    "scene": graph.Scene.GOAL,
                }
                # needs to communicate if it wasn't already communicated
                communicate = (
                    com_pred in inflated_goal.predicates
                    and {**com_pred, "scene": graph.Scene.INIT}
                    not in inflated_init.predicates
                )
                needs_lander = needs_lander or communicate
                init_pred = {
                    "typing": pred["typing"],
                    "parameters": pred["parameters"],
                    "scene": graph.Scene.INIT,
                }
                if init_pred not in inflated_init.predicates:
                    sample_waypoints.append((waypoint, communicate))
            case ["communicated", *_]:
                com_pred = {
                    "typing": pred["typing"],
                    "parameters": pred["parameters"],
                    "scene": graph.Scene.INIT,
                }
                if com_pred not in inflated_init.predicates:
                    needs_lander = True

    rover_map = rover_subgraph(init)
    path_map = rx.all_pairs_all_simple_paths(rover_map.graph)

    # find paths:
    # rover needs to sample -> communicate -> goal position

    def final_destination():
        for pred in inflated_goal.predicates:
            if pred["typing"] == "at_rover":
                return pred["parameters"][0]

    def _init_waypoint() -> list[int]:
        return init.successors(ReducedNode.ROVER)

    final_pos = final_destination()
    if sample_waypoints:
        endpoints = all_endpoints(
            rover_map,
            path_map,
            sample_waypoints,
            final_pos,
            needs_lander=needs_lander,
        )
    elif final_pos is None:
        # TODO: endpoints is the original rover position
        endpoints = {rover_map.graph.nodes().index(i) for i in _init_waypoint()}
        for state in list(endpoints):
            endpoints.update(
                [
                    dest
                    for dest in path_map[state]
                    if rover_map.graph.nodes()[dest].typing == {"waypoint"}
                ]
            )

        endpoints = {rover_map.graph.nodes()[e] for e in endpoints}

    if not final_destination():
        # rover doesn't need to be at a particular location,
        # check how many locations he can be at
        if len(endpoints) == 1:
            (endpoint,) = endpoints
            inflated_goal._add_predicate(
                {
                    "typing": "at_rover",
                    "parameters": [endpoint.node],
                    "scene": graph.Scene.GOAL,
                }
            )

    return init, _reduce_rover_single(inflated_goal)
