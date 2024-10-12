from functools import partial
from typing import Any

from collections import Counter, defaultdict
import copy
from itertools import permutations

import rustworkx as rx

from . import oracle
from .. import graph
from ..reduced_graph import ReducedProblemGraph, ReducedSceneGraph, ReducedNode

ReducedNode.register(
    {
        "ROVER": "rover",
        "LANDER": "lander",
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


class RoverSingleOracle(oracle.Oracle):

    @staticmethod
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

    def reduce(
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
            RoverSingleOracle._reduced_rover_add_predicate(reduced, predicate)

        return reduced

    def inflate(
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
                domain="rover-single",
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

    @staticmethod
    def _apply_unchangeable_predicates(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
    ):
        """Apply unchangeable predicates from the init to the goal of a rover
            problem.

        Args:
            scene (ReducedGraph): The reduced SceneGraph of a scene.
        """

        for u, v, edge in init.edges:
            copy_edge = copy.deepcopy(edge)
            copy_edge.scene = graph.Scene.GOAL
            match edge.predicate:
                case pred if pred in ("available", "channel_free"):
                    # all actions that affect these predicates always set them back
                    if not goal.has_edge(u, v, copy_edge):
                        goal.add_edge(u, v, copy_edge)
                # unary unchangeable predicates
                case pred if pred in (
                    "communicated_soil_data",
                    "communicated_rock_data",
                    "at_lander",  # landers don't move
                    "have_soil_analysis",
                    "have_rock_analysis",
                    "at_rock_sample",
                    "at_soil_sample",
                ):
                    if not goal.has_edge(u, v, copy_edge):
                        goal.add_edge(u, v, copy_edge)
                # binary unchangeable predicates
                case pred if pred in (
                    "supports",
                    "communicated_image_data",
                    "visible_from",
                    "have_image",
                    "visible",
                    "can_traverse",
                ):

                    if not goal.has_edge(u, v, copy_edge):
                        goal.add_edge(u, v, copy_edge)

    def _apply_have_predicates(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
    ):
        """Apply have_* predicates when needed.

        If a goal is to communicate a certain sample, then the rover must have
        that sample as well, so the have_* prediate must also exist.

        Args:
            init (ReducedSceneGraph): The reduced SceneGraph of the init.
            goal (ReducedSceneGraph): The reduced SceneGraph of the goal.
        """

        # if soil needs to be communicated but has not already,
        # then the rover must have the soil sample
        for comm, have, pred in (
            (
                ReducedNode.COMMUNICATED_SOIL_DATA,
                ReducedNode.HAVE_SOIL_ANALYSIS,
                "have_soil_analysis",
            ),
            (
                ReducedNode.COMMUNICATED_ROCK_DATA,
                ReducedNode.HAVE_ROCK_ANALYSIS,
                "have_rock_analysis",
            ),
        ):
            for waypoint in goal.successors(comm):
                # if the rover hasn't already communicated the sample
                # the rover has to have the sample
                if not init.has_edge(comm, waypoint) and not goal.has_edge(
                    have, waypoint
                ):
                    goal.add_edge(
                        have,
                        waypoint,
                        graph.PlanGraphEdge(predicate=pred, scene=graph.Scene.GOAL),
                    )

        for u, v, edge in list(goal.edges):
            if edge.predicate == "communicated_image_data":
                copy_edge = copy.deepcopy(edge)
                copy_edge.scene = graph.Scene.INIT

                have_edge = graph.PlanGraphEdge(
                    predicate="have_image",
                    scene=graph.Scene.GOAL,
                )
                if not init.has_edge(u, v, copy_edge) and not goal.has_edge(u, v, have_edge):
                    goal.add_edge(u, v, have_edge)


    @staticmethod
    def _get_sample_waypoints(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
    ) -> tuple[
        dict[str, set[graph.PlanGraphNode]],
        dict[str, set[graph.PlanGraphNode]],
        dict[str, set[graph.PlanGraphNode]],
    ]:
        """Get the sample waypoints as defined in init.

        Args:
            init (ReducedSceneGraph): The reduced SceneGraph of the init.
            goal (ReducedSceneGraph): The reduced SceneGraph of the goal.

        Returns:
            tuple[dict[str, set[graph.PlanGraphNode]], dict[str, set[graph.PlanGraphNode]]]:
                A tuple containing the sample waypoints in the init:
                    - "available": on ground & can be picked up.
                    - waypoint samples that started on the rover.
                    - waypoints that are required to be visited.
        """
        # AVAILABLE SAMPLES
        available_samples = defaultdict(set)

        rock_sample_node = ReducedNode.AT_ROCK_SAMPLE
        soil_sample_node = ReducedNode.AT_SOIL_SAMPLE
        goal_rock_samples = set(n for n, _ in goal.out_edges(rock_sample_node))
        for waypoint in init.successors(ReducedNode.AT_ROCK_SAMPLE):
            if waypoint not in goal_rock_samples:
                available_samples["rock"].add(waypoint)

        goal_soil_samples = set(n for n, _ in goal.out_edges(soil_sample_node))
        for waypoint in init.successors(ReducedNode.AT_SOIL_SAMPLE):
            if waypoint not in goal_soil_samples:
                available_samples["soil"].add(waypoint)

        # HELD SAMPLES
        held_samples = defaultdict(set)

        have_rock_node = ReducedNode.HAVE_ROCK_ANALYSIS
        have_soil_node = ReducedNode.HAVE_SOIL_ANALYSIS
        for waypoint, _ in init.out_edges(have_rock_node):
            held_samples["rock"].add(waypoint)

        for waypoint, _ in init.out_edges(have_soil_node):
            held_samples["soil"].add(waypoint)

        required_have = set()
        for _type, have in (
            # ("image", ReducedNode.HAVE_IMAGE),
            ("soil", ReducedNode.HAVE_SOIL_ANALYSIS),
            ("rock", ReducedNode.HAVE_ROCK_ANALYSIS),
        ):
            goal_have = (n for n in goal.successors(have))
            init_have = set(n for n in init.successors(have))

            for waypoint in goal_have:
                if waypoint not in init_have:
                    required_have.add((waypoint, _type))

        required_comm = defaultdict(set)
        for _type, comm in (
            # ("image", ReducedNode.COMMUNICATED_IMAGE_DATA),
            ("soil", ReducedNode.COMMUNICATED_SOIL_DATA),
            ("rock", ReducedNode.COMMUNICATED_ROCK_DATA),
        ):
            goal_comm = (n for n in goal.successors(comm))
            init_comm = set(n for n in init.successors(comm))

            for waypoint in goal_comm:
                if waypoint not in init_comm:
                    required_comm[_type].add((waypoint, _type))


        for u, v, edge in goal.edges:
            if edge == graph.PlanGraphEdge(predicate="have_image"):
                copy_edge = copy.deepcopy(edge)
                copy_edge.scene = graph.Scene.INIT

                if not init.has_edge(u, v, copy_edge):
                    required_have.add((u, v))
            elif edge == graph.PlanGraphEdge(predicate="communicated_image_data"):
                copy_edge = copy.deepcopy(edge)
                copy_edge.scene = graph.Scene.INIT

                if not init.has_edge(u, v, copy_edge):
                    required_comm["image"].add((u, v))

        required_stops = {
            "have": required_have,
            "comm": required_comm,
        }

        return available_samples, held_samples, required_stops

    @staticmethod
    def _rover_subgraph(scene: ReducedSceneGraph) -> ReducedSceneGraph:
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
                if node.typing.issubset({"waypoint", "objective", "camera", "mode"}):
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
                        if in_edge.predicate == "visible":
                            subgraph.add_edge(waypoint, u, copy.deepcopy(in_edge))

        return subgraph

    @staticmethod
    def _all_endpoints(
        scene: ReducedSceneGraph,
        path_map: rx.AllPairsMultiplePathMapping,
        sample_waypoints: set[tuple[graph.PlanGraphNode, bool]],
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
            sample_waypoints.add(
                (
                    graph.PlanGraphNode(
                        ReducedNode.LANDER,
                        name=ReducedNode.LANDER.value,
                        label=graph.Label.PREDICATE,
                        typing=ReducedNode.LANDER.value,
                    ),
                    False,
                )
            )
        waypoint_order = list(filter(check_lander, permutations(sample_waypoints)))
        to_idx = (
            [scene._node_lookup[w.node][0] for w, _ in wo] for wo in waypoint_order
        )

        if final_position:
            final_pos = scene._node_lookup[final_position.node][0]
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

    def fully_specify(
        problem: graph.ProblemGraph,
        return_reduced: bool = False,
    ) -> graph.ProblemGraph | ReducedProblemGraph:
        """Fully specifies a rover scene graph.

        Args:
            problem (graph.ProblemGraph): The problem graph to fully specify.
            return_reduced (bool, optional): Whether to return the reduced scene
                graph. Defaults to False.

        Returns:
            graph.ProblemGraph | ReducedProblemGraph: The fully specified
                problem graph.
        """
        inflated_init, inflated_goal = copy.deepcopy(problem).decompose()
        init: ReducedSceneGraph = RoverSingleOracle.reduce(inflated_init)
        goal: ReducedSceneGraph = RoverSingleOracle.reduce(inflated_goal)

        # bring all unchangeable predicates to the goal
        RoverSingleOracle._apply_unchangeable_predicates(init, goal)
        RoverSingleOracle._apply_have_predicates(init, goal)

        # all waypoints that must be visited
        sampleable_waypoints, held_waypoints, required_stops = (
            RoverSingleOracle._get_sample_waypoints(init, goal)
        )
        needs_lander = len(required_stops["comm"]) > 0

        combined_required_stops = dict()
        for waypoint, _ in required_stops["have"]:
            combined_required_stops[waypoint] = False
        for waypoint, _ in required_stops["comm"].items():
            if waypoint in combined_required_stops:
                combined_required_stops[waypoint] = True

        sample_waypoints = set((w, c) for w, c in combined_required_stops.items())

        rover_map = RoverSingleOracle._rover_subgraph(init)
        path_map = rx.all_pairs_all_simple_paths(rover_map.graph)

        # find paths:
        # rover needs to sample -> communicate -> goal position

        initial_pos = list(
            v for v, e in init.out_edges(ReducedNode.ROVER) if e.predicate == "at_rover"
        )[0]
        final_pos = (
            list(
                v
                for v, e in goal.out_edges(ReducedNode.ROVER)
                if e.predicate == "at_rover"
            )
            or None
        )

        if final_pos:
            final_pos = final_pos[0]

        if sample_waypoints:
            endpoints = RoverSingleOracle._all_endpoints(
                rover_map,
                path_map,
                sample_waypoints,
                final_pos,
                needs_lander=needs_lander,
            )
        elif final_pos is None:
            # TODO: endpoints is the original rover position
            endpoints = set([rover_map.graph.nodes().index(initial_pos)])
            for state in list(endpoints):
                endpoints.update(
                    [
                        dest
                        for dest in path_map[state]
                        if rover_map.graph.nodes()[dest].typing == {"waypoint"}
                    ]
                )

            endpoints = {rover_map.graph.nodes()[e] for e in endpoints}

        if not final_pos:
            # rover doesn't need to be at a particular location,
            # check how many locations he can be at
            if len(endpoints) == 1:
                (endpoint,) = endpoints
                goal.add_edge(
                    ReducedNode.ROVER,
                    endpoint,
                    graph.PlanGraphEdge(predicate="at_rover", scene=graph.Scene.GOAL),
                )

        if return_reduced:
            return ReducedProblemGraph.join(init, goal)
        else:
            return graph.ProblemGraph.join(
                inflated_init, RoverSingleOracle.inflate(goal)
            )
