import abc

from pddl.core import Action

from .. import graph, reduced_graph


class Oracle(abc.ABC):

    @staticmethod
    def reduce(
        scene: graph.SceneGraph | graph.ProblemGraph,
    ) -> reduced_graph.ReducedSceneGraph | reduced_graph.ReducedProblemGraph:
        """Reduces a scene graph to a reduced scene graph.

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
                reduced = reduced_graph.ReducedProblemGraph(
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
                reduced = reduced_graph.ReducedSceneGraph(
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
            match params:
                case [x]:
                    reduced.add_node(x, reduced_edge)
                case [x, y]:
                    reduced.add_edge(x, y, reduced_edge)
                case _:
                    raise ValueError("Predicate parameters must be 1 or 2.")

    @staticmethod
    def inflate(
        scene: reduced_graph.ReducedSceneGraph | reduced_graph.ReducedProblemGraph,
    ) -> graph.SceneGraph | graph.ProblemGraph:
        """Inflates a reduced scene graph to a scene graph.

        Args:
            scene (ReducedSceneGraph | ReducedProblemGraph): The reduced scene graph to inflate.

        Returns:
            SceneGraph | ProblemGraph: The inflated scene graph.
        """
        constants = []
        predicates = []

        for node in scene.nodes:
            if (
                not isinstance(node.node, reduced_graph.ReducedNode)
                and node.label == graph.Label.CONSTANT
            ):
                # add constants
                constants.append({"name": node.node, "typing": node.typing})

        for u, v, edge in scene.edges:
            match u.node:
                case pred_node if isinstance(pred_node, reduced_graph.ReducedNode):
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

        if isinstance(scene, reduced_graph.ReducedProblemGraph):
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
    @abc.abstractmethod
    def fully_specify(
        problem: graph.ProblemGraph,
        return_reduced: bool = False,
    ) -> graph.ProblemGraph | reduced_graph.ReducedProblemGraph:
        """Fully specifies a goal state.

        Args:
            problem (graph.ProblemGraph): The problem graph to fully specify.
            return_reduced (bool, optional): Whether to return a reduced problem graph.
                Defaults to False.

        Returns:
            ProblemGraph | ReducedProblemGraph: The fully specified problem graph.
        """

    @staticmethod
    def plan(problem: graph.ProblemGraph) -> list[Action]:
        """Generates a plan for a problem graph.

        Args:
            problem (graph.ProblemGraph): The problem graph to plan.

        Returns:
            list[Action]: The plan for the problem graph.
        """
        raise NotImplementedError("Planning not supported for this oracle.")
