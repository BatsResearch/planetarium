from typing import Any, Iterable

import abc
import enum
from functools import cached_property

import matplotlib.pyplot as plt
import networkx as nx
import rustworkx as rx


class Label(str, enum.Enum):
    CONSTANT = "constant"
    PREDICATE = "predicate"


class Scene(str, enum.Enum):
    INIT = "init"
    GOAL = "goal"


class PlanGraphNode:
    def __init__(
        self,
        node: str,
        name: str,
        label: Label,
        typing: str | None = None,
        scene: Scene | None = None,
    ):
        self.node = node
        self.name = name
        self.label = label
        self.typing = typing
        self.scene = scene

    def __eq__(self, other: "PlanGraphNode") -> bool:
        return (
            isinstance(other, PlanGraphNode)
            and self.node == other.node
            and self.name == other.name
            and self.label == other.label
            and self.typing == other.typing
            and self.scene == other.scene
        )

    def __hash__(self) -> int:
        return hash((self.name, self.label, (*sorted(self.typing),), self.scene))

    def __repr__(self) -> str:
        return f"PlanGraphNode(node={self.node}, name={self.name}, label={self.label}, typing={self.typing}, scene={self.scene})"

    def __str__(self) -> str:
        return f"PlanGraphNode(node={self.node}, name={self.name}, label={self.label}, typing={self.typing}, scene={self.scene})"


class PlanGraphEdge:
    def __init__(
        self,
        predicate: str,
        position: int | None = None,
        scene: Scene | None = None,
    ):
        self.predicate = predicate
        self.position = position
        self.scene = scene

    def __eq__(self, other: "PlanGraphEdge") -> bool:
        return (
            isinstance(other, PlanGraphEdge)
            and self.predicate == other.predicate
            and self.position == other.position
            and self.scene == other.scene
        )

    def __hash__(self) -> int:
        return hash((self.predicate, self.position, self.scene))

    def __repr__(self) -> str:
        return f"PlanGraphEdge(predicate={self.predicate}, position={self.position}, scene={self.scene})"

    def __str__(self) -> str:
        return f"PlanGraphEdge(predicate={self.predicate}, position={self.position}, scene={self.scene})"


class PlanGraph(metaclass=abc.ABCMeta):
    """
    Subclass of rx.PyDiGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        predicates (property): A dictionary of predicate nodes in the scene graph.
        domain (property): The domain of the scene graph.
    """

    def __init__(
        self,
        constants: list[dict[str, Any]],
        domain: str | None = None,
    ):
        """
        Initialize the SceneGraph instance.

        Parameters:
            constants (list): List of dictionaries representing constants.
            domain (str, optional): The domain of the scene graph.
                Defaults to None.
        """
        super().__init__()

        self._domain = domain
        self.graph = rx.PyDiGraph()

        for constant in constants:
            self.add_node(
                PlanGraphNode(
                    constant["name"],
                    name=constant["name"],
                    label=Label.CONSTANT,
                    typing=constant["typing"],
                )
            )

    @property
    def _node_lookup(self) -> dict[str, tuple[int, PlanGraphNode]]:
        return {node.node: (index, node) for index, node in enumerate(self.nodes)}

    @cached_property
    def nodes(self) -> list[PlanGraphNode]:
        return self.graph.nodes()

    @cached_property
    def edges(self) -> set[tuple[PlanGraphNode, PlanGraphNode, PlanGraphEdge]]:
        return [
            (self.nodes[u], self.nodes[v], data)
            for u, v, data in self.graph.edge_index_map().values()
        ]

    def add_node(self, node: PlanGraphNode):
        if node in self.nodes:
            raise ValueError(f"Node {node} already exists in the graph.")
        self.graph.add_node(node)

        if node.label == Label.CONSTANT:
            self.__dict__.pop("constant_nodes", None)
            self.__dict__.pop("constants", None)
        elif node.label == Label.PREDICATE:
            self.__dict__.pop("predicate_nodes", None)
            self.__dict__.pop("predicates", None)

        self.__dict__.pop("nodes", None)
        self.__dict__.pop("_node_lookup", None)

    def has_edge(
        self,
        u: str | PlanGraphNode,
        v: str | PlanGraphNode,
        edge: PlanGraphEdge | None = None,
    ) -> bool:
        if isinstance(u, PlanGraphNode):
            u_index = self.nodes.index(u)
        else:
            u_index, _ = self._node_lookup[u]

        if isinstance(v, PlanGraphNode):
            v_index = self.nodes.index(v)
        else:
            v_index, _ = self._node_lookup[v]

        if edge:
            return (u_index, v_index, edge) in self.graph.edge_index_map().values()
        else:
            return self.graph.has_edge(u_index, v_index)

    def add_edge(
        self, u: str | PlanGraphNode, v: str | PlanGraphNode, edge: PlanGraphEdge
    ):
        if isinstance(u, PlanGraphNode):
            u_index = self.nodes.index(u)
        else:
            u_index, _ = self._node_lookup[u]

        if isinstance(v, PlanGraphNode):
            v_index = self.nodes.index(v)
        else:
            v_index, _ = self._node_lookup[v]

        self.graph.add_edge(u_index, v_index, edge)

        self.__dict__.pop("edges", None)
        self.__dict__.pop("predicates", None)

    def _add_predicate(
        self,
        predicate: dict[str, Any],
        scene: Scene | None = None,
    ):
        """
        Add a predicate to the plan graph.

        Parameters:
            predicate (dict): A dictionary representing the predicate.
            scene (Scene, optional): The scene in which the predicate occurs.
        """
        predicate_name = self._build_unique_predicate_name(
            predicate_name=predicate["typing"],
            argument_names=predicate["parameters"],
        )
        self.add_node(
            PlanGraphNode(
                predicate_name,
                name=predicate_name,
                label=Label.PREDICATE,
                typing=predicate["typing"],
                scene=scene,
            )
        )

        for position, parameter_name in enumerate(predicate["parameters"]):
            if parameter_name not in [node.name for node in self.constant_nodes]:
                raise ValueError(f"Parameter {parameter_name} not found in constants.")
            self.add_edge(
                predicate_name,
                parameter_name,
                PlanGraphEdge(
                    predicate=predicate["typing"],
                    position=position,
                    scene=scene,
                ),
            )

    def in_degree(self, node: str | PlanGraphNode) -> int:
        if isinstance(node, PlanGraphNode):
            return self.graph.in_degree(self.nodes.index(node))
        else:
            return self.graph.in_degree(self._node_lookup[node][0])

    def out_degree(self, node: str | PlanGraphNode) -> int:
        if isinstance(node, PlanGraphNode):
            return self.graph.out_degree(self.nodes.index(node))
        else:
            return self.graph.out_degree(self._node_lookup[node][0])

    def predecessors(self, node: str | PlanGraphNode) -> list[PlanGraphNode]:
        if isinstance(node, PlanGraphNode):
            preds = self.graph.predecessors(self.nodes.index(node))
        else:
            preds = self.graph.predecessors(self._node_lookup[node][0])

        return preds

    def successors(self, node: str | PlanGraphNode) -> list[PlanGraphNode]:
        if isinstance(node, PlanGraphNode):
            succs = self.graph.successors(self.nodes.index(node))
        else:
            succs = self.graph.successors(self._node_lookup[node][0])

        return [self.nodes[succ] for succ in succs]

    def in_edges(
        self, node: str | PlanGraphNode
    ) -> list[tuple[PlanGraphNode, PlanGraphEdge]]:
        if isinstance(node, PlanGraphNode):
            edges = self.graph.in_edges(self.nodes.index(node))
        else:
            edges = self.graph.in_edges(self._node_lookup[node][0])

        return [(self.nodes[u], edge) for u, _, edge in edges]

    def out_edges(
        self, node: str | PlanGraphNode
    ) -> list[tuple[PlanGraphNode, PlanGraphEdge]]:
        if isinstance(node, PlanGraphNode):
            edges = self.graph.out_edges(self.nodes.index(node))
        else:
            edges = self.graph.out_edges(self._node_lookup[node][0])

        return [(self.nodes[v], edge) for _, v, edge in edges]

    @staticmethod
    def _build_unique_predicate_name(
        predicate_name: str, argument_names: Iterable[str]
    ) -> str:
        """
        Build a unique name for a predicate based on its name and argument names.

        Parameters:
            predicate_name (str): The name of the predicate.
            argument_names (Iterable[str]): Sequence of argument names
            for the predicate.

        Returns:
            str: The unique name for the predicate.
        """
        return "-".join([predicate_name, *argument_names])

    @cached_property
    def domain(self) -> str | None:
        """
        Get the domain of the scene graph.

        Returns:
            str: The domain of the scene graph.
        """
        return self._domain

    @cached_property
    def constant_nodes(self) -> list[PlanGraphNode]:
        """Get a list of constant nodes in the scene graph.

        Returns:
            list[PlanGraphNode]: A list of constant nodes.
        """
        return [node for node in self.nodes if node.label == Label.CONSTANT]

    @cached_property
    def constants(self) -> list[dict[str, Any]]:
        return [
            {"name": constant.name, "typing": constant.typing}
            for constant in self.constant_nodes
        ]

    @cached_property
    def predicate_nodes(self) -> list[PlanGraphNode]:
        """Get a list of predicate nodes in the scene graph.

        Returns:
            list[PlanGraphNode]: A list of predicate nodes.
        """
        return [node for node in self.nodes if node.label == Label.PREDICATE]

    @property
    def predicates(self) -> list[dict[str, Any]]:
        predicates = []
        for node in self.predicate_nodes:
            edges = self.out_edges(node)
            edges.sort(key=lambda x: x[1].position)
            predicates.append(
                {
                    "typing": node.typing,
                    "parameters": [obj_node.name for obj_node, _ in edges],
                    "scene": node.scene,
                }
            )

        return predicates

    def __eq__(self, other: "PlanGraph") -> bool:
        """
        Check if two plan graphs are equal.

        Parameters:
            other (PlanGraph): The other plan graph to compare.

        Returns:
            bool: True if the plan graphs are equal, False otherwise.
        """
        return (
            isinstance(other, PlanGraph)
            and set(self.nodes) == set(other.nodes)
            and set(self.edges) == set(other.edges)
            and self.domain == other.domain
        )

    def plot(self, fig: plt.Figure | None = None) -> plt.Figure:
        """Generate a plot of the graph, sorted by topological generation.

        Args:
            fig (plt.Figure | None, optional): The figure to plot on. Defaults
                to None.

        Returns:
            plt.Figure: The figure containing the plot.
        """
        # rx has no plotting functionality
        nx_graph = nx.MultiDiGraph()
        nx_graph.add_edges_from(
            [(u.node, v.node, {"data": edge}) for u, v, edge in self.edges]
        )

        for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
            for node in nodes:
                nx_graph.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(
            nx_graph,
            align="horizontal",
            subset_key="layer",
            scale=-1,
        )

        if fig is None:
            fig = plt.figure()

        nx.draw(nx_graph, pos=pos, ax=fig.gca(), with_labels=True)

        return fig


class SceneGraph(PlanGraph):
    """
    Subclass of PlanGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        predicates (property): A dictionary of predicate nodes in the scene graph.
        domain (property): The domain of the scene graph.
    """

    def __init__(
        self,
        constants: list[dict[str, Any]],
        predicates: list[dict[str, Any]],
        domain: str | None = None,
        scene: Scene | None = None,
    ):
        """
        Initialize the SceneGraph instance.

        Parameters:
            constants (list): List of dictionaries representing constants.
            predicates (list): List of dictionaries representing predicates.
            domain (str, optional): The domain of the scene graph.
                Defaults to None.
            scene (str, optional): The scene of the scene graph.
        """

        super().__init__(constants, domain=domain)

        self.scene = scene

        for predicate in predicates:
            self._add_predicate(predicate, scene=scene)


class ProblemGraph(PlanGraph):
    """
    Subclass of PlanGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        init_predicates (property): A dictionary of predicate nodes in the initial scene graph.
        goal_predicates (property): A dictionary of predicate nodes in the goal scene graph.

    """

    def __init__(
        self,
        constants: list[dict[str, Any]],
        init_predicates: list[dict[str, Any]],
        goal_predicates: list[dict[str, Any]],
        domain: str | None = None,
    ):
        """
        Initialize the ProblemGraph instance.

        Parameters:
            constants (list): List of dictionaries representing constants.
            init_predicates (list): List of dictionaries representing predicates
                in the initial scene.
            goal_predicates (list): List of dictionaries representing predicates
                in the goal scene.
            domain (str, optional): The domain of the scene graph.
                Defaults to None.
        """
        super().__init__(constants, domain=domain)

        for scene, predicates in (
            (Scene.INIT, init_predicates),
            (Scene.GOAL, goal_predicates),
        ):
            for predicate in predicates:
                self._add_predicate(predicate, scene=scene)

    def __eq__(self, other: "ProblemGraph") -> bool:
        return (
            super().__eq__(other)
            and set(self.init_predicate_nodes) == set(other.init_predicate_nodes)
            and set(self.goal_predicate_nodes) == set(other.goal_predicate_nodes)
        )

    def add_node(self, node: PlanGraphNode):
        super().add_node(node)
        if node.label == Label.PREDICATE:
            self.__dict__.pop("init_predicate_nodes", None)
            self.__dict__.pop("goal_predicate_nodes", None)
            self.__dict__.pop("init_predicates", None)
            self.__dict__.pop("goal_predicates", None)

        self.__dict__.pop("_decompose", None)

    def add_edge(
        self, u: str | PlanGraphNode, v: str | PlanGraphNode, edge: PlanGraphEdge
    ):
        super().add_edge(u, v, edge)

        self.__dict__.pop("init_predicate_nodes", None)
        self.__dict__.pop("goal_predicate_nodes", None)
        self.__dict__.pop("init_predicates", None)
        self.__dict__.pop("goal_predicates", None)

        self.__dict__.pop("_decompose", None)

    @cached_property
    def init_predicate_nodes(self) -> list[PlanGraphNode]:
        """Get a list of predicate nodes in the initial scene.

        Returns:
            list[PlanGraphNode]: A list of predicate nodes in the initial scene.
        """
        return [
            node
            for node in self.nodes
            if node.label == Label.PREDICATE and node.scene == Scene.INIT
        ]

    @cached_property
    def goal_predicate_nodes(self) -> list[PlanGraphNode]:
        """Get a list of predicate nodes in the goal scene.

        Returns:
            list[PlanGraphNode]: A list of predicate nodes in the goal scene.
        """
        return [
            node
            for node in self.nodes
            if node.label == Label.PREDICATE and node.scene == Scene.GOAL
        ]

    @cached_property
    def init_predicates(self) -> list[dict[str, Any]]:
        predicates = []
        for node in self.init_predicate_nodes:
            edges = self.out_edges(node)
            edges.sort(key=lambda x: x[1].position)
            predicates.append(
                {
                    "typing": node.typing,
                    "parameters": [obj_node.name for obj_node, _ in edges],
                    "scene": node.scene,
                }
            )

        return predicates

    @cached_property
    def goal_predicates(self) -> list[dict[str, Any]]:
        predicates = []
        for node in self.goal_predicate_nodes:
            edges = self.out_edges(node)
            edges.sort(key=lambda x: x[1].position)
            predicates.append(
                {
                    "typing": node.typing,
                    "parameters": [obj_node.name for obj_node, _ in edges],
                    "scene": node.scene,
                }
            )

        return predicates

    @cached_property
    def _decompose(self) -> tuple[SceneGraph, SceneGraph]:
        """
        Decompose the problem graph into initial and goal scene graphs.

        Returns:
            tuple[SceneGraph, SceneGraph]: A tuple containing the initial and goal scene graphs.
        """

        init_scene = SceneGraph(
            constants=self.constants,
            predicates=self.init_predicates,
            domain=self.domain,
            scene=Scene.INIT,
        )

        goal_scene = SceneGraph(
            constants=self.constants,
            predicates=self.goal_predicates,
            domain=self.domain,
            scene=Scene.GOAL,
        )

        return init_scene, goal_scene

    def decompose(self) -> tuple[SceneGraph, SceneGraph]:
        """
        Decompose the problem graph into initial and goal scene graphs.

        Returns:
            tuple[SceneGraph, SceneGraph]: A tuple containing the initial and goal scene graphs.
        """

        return self._decompose

    @staticmethod
    def join(init: SceneGraph, goal: SceneGraph) -> "ProblemGraph":
        """
        Combine initial and goal scene graphs into a problem graph.

        Parameters:
            init (SceneGraph): The initial scene graph.
            goal (SceneGraph): The goal scene graph.

        Returns:
            ProblemGraph: The combined problem graph.
        """
        return ProblemGraph(
            constants=init.constants,
            init_predicates=init.predicates,
            goal_predicates=goal.predicates,
            domain=init.domain,
        )
