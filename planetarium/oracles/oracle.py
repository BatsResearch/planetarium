import abc

from pddl.core import Action

from .. import graph, reduced_graph


class Oracle(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def reduce(
        scene: graph.SceneGraph | graph.ProblemGraph,
    ) -> reduced_graph.ReducedSceneGraph | reduced_graph.ReducedProblemGraph:
        """Reduces a scene graph to a reduced scene graph.

        Args:
            scene (graph.SceneGraph | graph.ProblemGraph): The scene graph to reduce.

        Returns:
            ReducedSceneGraph | ReducedProblemGraph: The reduced scene graph.
        """

    @staticmethod
    @abc.abstractmethod
    def inflate(
        scene: reduced_graph.ReducedSceneGraph | reduced_graph.ReducedProblemGraph,
    ) -> graph.SceneGraph | graph.ProblemGraph:
        """Inflates a reduced scene graph to a scene graph.

        Args:
            scene (ReducedSceneGraph | ReducedProblemGraph): The reduced scene graph to inflate.

        Returns:
            SceneGraph | ProblemGraph: The inflated scene graph.
        """

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
