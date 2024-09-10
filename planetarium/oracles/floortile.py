from collections import defaultdict
import copy

from pddl.core import Action
import rustworkx as rx

from . import oracle
from .. import graph
from ..reduced_graph import ReducedSceneGraph, ReducedProblemGraph, ReducedNode

# Add the ReducedNode enum for the gripper domain
ReducedNode.register(
    {
        "AVAILABLE": "available-color",
    },
    "floor-tile",
)


class FloorTileOracle(oracle.Oracle):

    def reduce(
        scene: graph.SceneGraph | graph.ProblemGraph,
    ) -> ReducedSceneGraph | ReducedProblemGraph:
        """Reduces a floortile scene graph to a reduced scene graph.

        Args:
            scene (graph.SceneGraph | graph.ProblemGraph): The scene graph to reduce.

        Returns:
            ReducedSceneGraph | ReducedProblemGraph: The reduced scene graph.
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
            params = predicate["parameters"]
            reduced_edge = graph.PlanGraphEdge(
                predicate=predicate["typing"],
                scene=predicate.get("scene"),
            )
            match (predicate["typing"], params):
                case (_, [x, y]):
                    reduced.add_edge(x, y, reduced_edge)
                case ("clear", [x]):
                    reduced.add_edge(ReducedNode.CLEAR, x, reduced_edge)
                case ("available-color", [x]):
                    reduced.add_edge(ReducedNode.AVAILABLE, x, reduced_edge)

        return reduced

    def inflate(
        scene: ReducedSceneGraph | ReducedProblemGraph,
    ) -> graph.SceneGraph | graph.ProblemGraph:
        """Respecify a reduced floortile scene graph to a full scene graph.

        Args:
            scene (ReducedSceneGraph | ReducedProblemGraph): The reduced scene graph
                to inflate.

        Returns:
            graph.SceneGraph | graph.ProblemGraph: The inflated scene graph.
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
            match u.node:
                case pred_node if isinstance(pred_node, ReducedNode):
                    predicates.append(
                        {
                            "typing": edge.predicate,
                            "parameters": [v.node],
                            "scene": edge.scene,
                        }
                    )
                case _:
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
                domain=scene.domain,
                requirements=scene._requirements,
            )
        else:
            return graph.SceneGraph(
                constants,
                predicates,
                domain=scene.domain,
                scene=scene.scene,
                requirements=scene._requirements,
            )

    @staticmethod
    def _apply_unchangeable_predicates(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
    ):
        unchangeable = {
            "up",
            "right",
            "painted",
            "available-color",
        }
        for u, v, edge in init.edges:
            if edge.predicate in unchangeable:
                edge = copy.deepcopy(edge)
                edge.scene = graph.Scene.GOAL
                if not goal.has_edge(u, v, edge):
                    goal.add_edge(u, v, edge)

    @staticmethod
    def _fixed_color_predicates(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
        robots: list[ReducedNode],
    ):
        # if two or more colors are available, then the robot color cannot be fixed
        # (they can finish painting and switch to any of the available colors)
        available_colors = init.successors(ReducedNode.AVAILABLE)
        if len(available_colors) > 1:
            return

        # find the color each robot has
        has_color = defaultdict(list)
        for robot in robots:
            has_color[robot] = [
                v for v, edge in init.out_edges(robot) if edge.predicate == "robot-has"
            ]

        # if no colors are available, then all robots must have their original color
        if len(available_colors) == 0:
            for robot, colors in has_color.items():
                edge = graph.PlanGraphEdge(
                    predicate="robot-has",
                    scene=graph.Scene.GOAL,
                )
                for color in colors:
                    if not goal.has_edge(robot, color, edge):
                        goal.add_edge(robot, color, edge)
        # if only one color is available:
        elif len(available_colors) == 1:
            # if there is only one robot and it either
            # a) has the available color
            # b) the available color needs to be painted
            #
            # then the robot color is fixed to the available color
            (available_color,) = available_colors
            if len(robots) == 1:
                (robot,) = robots
                robot_color = has_color[robot][0]
                if robot_color == available_color:
                    edge = graph.PlanGraphEdge(
                        predicate="robot-has",
                        scene=graph.Scene.GOAL,
                    )
                    if not goal.has_edge(robot, robot_color, edge):
                        goal.add_edge(robot, robot_color, edge)
                else:
                    painted_colors = set()
                    for u, v, edge in goal.edges:
                        if edge.predicate == "painted":
                            init_edge = copy.deepcopy(edge)
                            if not init.has_edge(u, v, init_edge):
                                painted_colors.add(v)

                    if available_color in painted_colors:
                        edge = graph.PlanGraphEdge(
                            predicate="robot-has",
                            scene=graph.Scene.GOAL,
                        )
                        if not goal.has_edge(robot, available_color, edge):
                            goal.add_edge(robot, available_color, edge)
            else:
                # if there are more than one robot, every robot that already has that
                # color must keep it
                for robot, colors in has_color.items():
                    if available_color in colors:
                        edge = graph.PlanGraphEdge(
                            predicate="robot-has",
                            scene=graph.Scene.GOAL,
                        )
                        if not goal.has_edge(robot, available_color, edge):
                            goal.add_edge(robot, available_color, edge)

                # for each node that needs to be painted the available_color,
                # if there is only one robot that can paint it, then that robot
                # must paint end with that color
                painted_nodes = []
                node_reachable_by: list[list] = []

                subgraph_nodes = [
                    i
                    for i, n in enumerate(goal.nodes)
                    if n.typing in ({"tile"}, {"robot"})
                ]
                subgraph = init.graph.subgraph(subgraph_nodes).to_undirected()
                print('subgraph', subgraph.nodes())

                for u, v, edge in goal.edges:
                    if edge.predicate == "painted":
                        painted_nodes.append(u)
                        reachable = []
                        # find all robots that can reach this node
                        for robot in robots:
                            robot_idx = subgraph.nodes().index(robot)
                            u_idx = subgraph.nodes().index(u)

                            if rx.has_path(subgraph, robot_idx, u_idx):
                                reachable.append(robot)
                        node_reachable_by.append(reachable)

                for r in node_reachable_by:
                    # if there's only one robot that can paint it, then it must end up
                    # painting it
                    if len(r) == 1:
                        robot = r[0]
                        # assign the robot the only available color
                        # (notice branch above ensures there's only one available color)
                        edge = graph.PlanGraphEdge(
                            predicate="robot-has",
                            scene=graph.Scene.GOAL,
                        )
                        if not goal.has_edge(robot, available_color, edge):
                            goal.add_edge(robot, available_color, edge)

    def _fix_possible_positions(
        init: ReducedSceneGraph,
        goal: ReducedSceneGraph,
        robots: list[graph.PlanGraphNode],
    ):
        for robot in robots:
            init_pos = [
                v for v, edge in init.out_edges(robot) if edge.predicate == "robot-at"
            ]
            goal_pos = [
                v for v, edge in goal.out_edges(robot) if edge.predicate == "robot-at"
            ]

            if not init_pos or goal_pos:
                # if no initial position is specified or the goal position is already
                # specified, we can't determine the final robot position
                continue

            init_pos = init_pos[0]
            pos_neighbors = [
                v
                for v, edge in (*init.out_edges(init_pos), *init.in_edges(init_pos))
                if edge.predicate in {"up", "right"}
            ]

            if not pos_neighbors:
                # if the initial position has no neighbors, the robot must end there
                edge = graph.PlanGraphEdge(
                    predicate="robot-at",
                    scene=graph.Scene.GOAL,
                )
                if not goal.has_edge(robot, init_pos, edge):
                    goal.add_edge(robot, init_pos, edge)

    # TODO: if a robot is the only one that can paint a tile, it must end up painting it
    # if a robot ends up on a tile that is disconnected, then it must end up on that tile:

    def fully_specify(
        problem: graph.ProblemGraph,
        return_reduced: bool = False,
    ) -> graph.ProblemGraph | ReducedProblemGraph:
        """Fully specifies a floortile scene graph.

        Args:
            problem (graph.ProblemGraph): The problem graph to fully specify.
            return_reduced (bool, optional): Whether to return a reduced problem graph.
                Defaults to False.

        Returns:
            graph.ProblemGraph | ReducedProblemGraph: The fully specified problem graph.
        """
        inflated_init, inflated_goal = problem.decompose()

        init: ReducedSceneGraph = FloorTileOracle.reduce(inflated_init)
        goal: ReducedSceneGraph = FloorTileOracle.reduce(inflated_goal)

        robots = [r for r in init.nodes if r.typing == {"robot"}]

        FloorTileOracle._apply_unchangeable_predicates(init, goal)
        FloorTileOracle._fixed_color_predicates(init, goal, robots)

        # if a robot that starts on a tile with no neighbors, it must also end
        # on that tile
        FloorTileOracle._fix_possible_positions(init, goal, robots)

        if return_reduced:
            return ReducedProblemGraph.join(init, goal)
        else:
            return graph.ProblemGraph.join(inflated_init, FloorTileOracle.inflate(goal))
