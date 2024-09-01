from typing import Any

import copy
import aenum

from planetarium import graph


class ReducedNode(str, aenum.Enum):

    @classmethod
    def register(
        cls,
        attrs: dict[str, str],
        domain: str | None = None,
    ) -> set["ReducedNode"]:
        """Extend the ReducedNode enum with new nodes.

        Args:
            attrs (dict[str, str]): The new nodes to add.
            domain (str, optional): The domain to extend. Defaults to None.

        Raises:
            AttributeError: If the domain is already extended.

        Returns:
            set[ReducedNode]: The set of new nodes.
        """
        nodes = set()
        for name, value in attrs.items():
            if name not in cls.__members__:
                aenum.extend_enum(cls, name, value)
            nodes.add(cls[name])

        if isinstance(domain, str):
            if domain in ReducedNodes:
                raise AttributeError(
                    f"ReducedNode already extended for domain {domain}."
                )
            ReducedNodes[domain] = nodes

        return nodes


ReducedNodes: dict[str, set[ReducedNode]] = {}


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

        for e in ReducedNodes.get(domain, set()):
            predicate = e.value
            self.add_node(
                graph.PlanGraphNode(
                    e,
                    name=predicate,
                    label=graph.Label.PREDICATE,
                    typing=predicate,
                ),
            )

    def _add_predicate(
        self,
        predicate: dict[str, Any],
        scene: graph.Scene | None = None,
    ):
        raise AttributeError(
            "ReducedSceneGraph does not support adding predicates directly."
        )


class ReducedProblemGraph(graph.PlanGraph):
    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str,
        requirements: tuple[str] = (),
    ):
        super().__init__(constants, domain=domain, requirements=requirements)

        for e in ReducedNodes.get(domain, []):
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

        for node in (*init.nodes, *goal.nodes):
            if (
                node not in problem.nodes
                and node.label == graph.Label.PREDICATE
                and not isinstance(node.node, ReducedNode)
            ):
                node = copy.deepcopy(node)
                problem.add_node(node)

        for u, v, edge in init.edges:
            edge = copy.deepcopy(edge)
            problem.add_edge(u, v, edge)
            edge.scene = graph.Scene.INIT
        for u, v, edge in goal.edges:
            edge = copy.deepcopy(edge)
            edge.scene = graph.Scene.GOAL
            problem.add_edge(u, v, edge)

        return problem
