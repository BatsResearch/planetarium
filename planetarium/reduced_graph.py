from typing import Any

import copy
import enum

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
    SOIL_ANALYSIS = "equipped_for_soil_analysis"
    ROCK_ANALYSIS = "equipped_for_rock_analysis"
    IMAGING = "equipped_for_imaging"
    EMPTY = "empty"
    FULL = "full"
    AVAILABLE = "available"
    COMMUNICATED_SOIL_DATA = "communicated_soil_data"
    COMMUNICATED_ROCK_DATA = "communicated_rock_data"
    AT_SOIL_SAMPLE = "at_soil_sample"
    AT_ROCK_SAMPLE = "at_rock_sample"
    CHANNEL_FREE = "channel_free"


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

RoverReducedNodes = {
    ReducedNode.SOIL_ANALYSIS,
    ReducedNode.ROCK_ANALYSIS,
    ReducedNode.IMAGING,
    ReducedNode.EMPTY,
    ReducedNode.FULL,
    ReducedNode.AVAILABLE,
    ReducedNode.COMMUNICATED_SOIL_DATA,
    ReducedNode.COMMUNICATED_ROCK_DATA,
    ReducedNode.AT_SOIL_SAMPLE,
    ReducedNode.AT_ROCK_SAMPLE,
    ReducedNode.CHANNEL_FREE,
}

ReducedNodes = {
    "blocksworld": BlocksworldReducedNodes,
    "gripper": GripperReducedNodes,
    "rover": RoverReducedNodes,
}


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
