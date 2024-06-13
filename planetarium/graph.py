import abc
import enum
import networkx as nx
import typing


class Label(str, enum.Enum):
    CONSTANT = "constant"
    PREDICATE = "predicate"


class PlanGraph(nx.MultiDiGraph, metaclass=abc.ABCMeta):
    """
    Subclass of nx.MultiDiGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        predicates (property): A dictionary of predicate nodes in the scene graph.
        domain (property): The domain of the scene graph.
    """

    def __init__(
        self,
        constants: list[dict[str, typing.Any]],
        predicates: list[dict[str, typing.Any]],
        domain: str | None = None,
    ):
        """
        Initialize the SceneGraph instance.

        Parameters:
            constants (list): List of dictionaries representing constants.
            predicates (list): List of dictionaries representing predicates.
            domain (str, optional): The domain of the scene graph.
                Defaults to None.
        """

        super().__init__()

        self._constants = constants
        self._predicates = predicates
        self._domain = domain

        for constant in constants:
            self.add_node(
                constant["name"],
                name=constant["name"],
                typing=constant["typing"],
                label=Label.CONSTANT,
            )

    def _add_predicate(self, predicate: dict[str, typing.Any], **kwargs):
        """
        Add a predicate to the plan graph.

        Parameters:
            predicate (dict): A dictionary representing the predicate.
        """
        predicate_name = self._build_unique_predicate_name(
            predicate_name=predicate["typing"],
            argument_names=predicate["parameters"],
        )
        self.add_node(
            predicate_name,
            name=predicate_name,
            typing=predicate["typing"],
            label=Label.PREDICATE,
        )

        for position, parameter_name in enumerate(predicate["parameters"]):
            if parameter_name not in self.constants:
                raise ValueError(f"Parameter {parameter_name} not found in constants.")
            self.add_edge(
                predicate_name,
                parameter_name,
                position=position,
                predicate=predicate["typing"],
                **kwargs,
            )

    @staticmethod
    def _build_unique_predicate_name(
        predicate_name: str, argument_names: typing.Iterable[str]
    ) -> str:
        """
        Build a unique name for a predicate based on its name and argument names.

        Parameters:
            predicate_name (str): The name of the predicate.
            argument_names (typing.Iterable[str]): Sequence of argument names
            for the predicate.

        Returns:
            str: The unique name for the predicate.
        """
        return "-".join([predicate_name, *argument_names])

    @property
    def domain(self) -> str | None:
        """
        Get the domain of the scene graph.

        Returns:
            str: The domain of the scene graph.
        """
        return self._domain

    @property
    def constants(self) -> dict:
        """
        Get a dictionary of constant nodes in the scene graph.

        Returns:
            dict: A dictionary containing constant nodes.
        """
        return dict(
            filter(lambda node: node[1]["label"] == Label.CONSTANT, self.nodes.items())
        )

    @property
    def predicates(self) -> dict:
        """
        Get a dictionary of predicate nodes in the scene graph.

        Returns:
            dict: A dictionary containing predicate nodes.
        """
        return dict(
            filter(lambda node: node[1]["label"] == Label.PREDICATE, self.nodes.items())
        )


class SceneGraph(PlanGraph):
    """
    Subclass of nx.MultiDiGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        predicates (property): A dictionary of predicate nodes in the scene graph.
        domain (property): The domain of the scene graph.
    """

    def __init__(
        self,
        constants: list[dict[str, typing.Any]],
        predicates: list[dict[str, typing.Any]],
        domain: str | None = None,
    ):
        """
        Initialize the SceneGraph instance.

        Parameters:
            constants (list): List of dictionaries representing constants.
            predicates (list): List of dictionaries representing predicates.
            domain (str, optional): The domain of the scene graph.
                Defaults to None.
        """

        super().__init__(constants, predicates, domain=domain)

        for predicate in predicates:
            self._add_predicate(predicate)


class ProblemGraph(PlanGraph):
    """
    Subclass of nx.MultiDiGraph representing a scene graph.

    Attributes:
        constants (property): A dictionary of constant nodes in the scene graph.
        init_predicates (property): A dictionary of predicate nodes in the initial scene graph.
        goal_predicates (property): A dictionary of predicate nodes in the goal scene graph.

    """

    def __init__(
        self,
        constants: list[dict[str, typing.Any]],
        init_predicates: list[dict[str, typing.Any]],
        goal_predicates: list[dict[str, typing.Any]],
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

        super().__init__(constants, init_predicates + goal_predicates, domain=domain)

        self._init_predicates = init_predicates
        self._goal_predicates = goal_predicates

        for scene, predicates in (
            ("init", init_predicates),
            ("goal", goal_predicates),
        ):
            for predicate in predicates:
                self._add_predicate(predicate, scene=scene)

    @property
    def init_predicates(self) -> dict:
        """
        Get a dictionary of predicate nodes in the initial scene graph.

        Returns:
            dict: A dictionary containing predicate nodes.
        """
        return dict(
            filter(
                lambda node: node[1]["label"] == Label.PREDICATE
                and node[1]["scene"] == "init",
                self.nodes.items(),
            )
        )

    @property
    def goal_predicates(self) -> dict:
        """
        Get a dictionary of predicate nodes in the initial scene graph.

        Returns:
            dict: A dictionary containing predicate nodes.
        """
        return dict(
            filter(
                lambda node: node[1]["label"] == Label.PREDICATE
                and node[1]["scene"] == "goal",
                self.nodes.items(),
            )
        )

    def decompose(self) -> tuple[SceneGraph, SceneGraph]:
        """
        Decompose the problem graph into initial and goal scene graphs.

        Returns:
            tuple[SceneGraph, SceneGraph]: A tuple containing the initial and goal scene graphs.
        """
        init_scene = SceneGraph(
            constants=self._constants,
            predicates=self._init_predicates,
            domain=self.domain,
        )

        goal_scene = SceneGraph(
            constants=self._constants,
            predicates=self._goal_predicates,
            domain=self.domain,
        )

        return init_scene, goal_scene

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
            constants=init._constants,
            init_predicates=init._predicates,
            goal_predicates=goal._predicates,
            domain=init.domain,
        )
