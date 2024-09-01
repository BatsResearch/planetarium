from collections import defaultdict
import copy

from pddl.core import Action
import rustworkx as rx

from planetarium import graph
from ..reduced_graph import ReducedSceneGraph, ReducedProblemGraph, ReducedNode


# Add enum for blocksworld domain
ReducedNode.register(
    {
        "TABLE": "table",
        "CLEAR": "clear",
        "ARM": "arm",
    },
    "blocksworld",
)


def _reduce_blocksworld(
    scene: graph.SceneGraph | graph.ProblemGraph,
) -> ReducedSceneGraph | ReducedProblemGraph:
    """Reduces a blocksworld scene graph to a Directed Acyclic Graph.

    Args:
        problem (graph.SceneGraph | graph.ProblemGraph): The scene graph to
            reduce.

    Returns:
        ReducedGraph: The reduced problem graph.
    """

    nodes = defaultdict(list)
    for node in scene.nodes:
        nodes[node.label].append(node)

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
        match (predicate["typing"], len(params)):
            case ("arm-empty", 0):
                reduced.add_edge(ReducedNode.CLEAR, ReducedNode.ARM, reduced_edge)
            case ("on-table", 1):
                reduced.add_edge(params[0], ReducedNode.TABLE, reduced_edge)
            case ("clear", 1):
                reduced.add_edge(ReducedNode.CLEAR, params[0], reduced_edge)
            case ("on", 2):
                reduced.add_edge(params[0], params[1], reduced_edge)
            case ("holding", 1):
                reduced.add_edge(params[0], ReducedNode.ARM, reduced_edge)
    return reduced


def _inflate_blocksworld(
    scene: ReducedSceneGraph | ReducedProblemGraph,
) -> graph.SceneGraph:
    """Respecify a blocksworld scene graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        graph.SceneGraph: The respecified scene graph.
    """
    constants = []
    predicates = []

    for node in scene.nodes:
        if not isinstance(node.node, ReducedNode):
            constants.append({"name": node.node, "typing": node.typing})

    for u, v, edge in scene.edges:
        match (u.node, v.node):
            case (ReducedNode.CLEAR, ReducedNode.ARM):
                predicates.append(
                    {
                        "typing": "arm-empty",
                        "parameters": [],
                        "scene": edge.scene,
                    }
                )
            case (ReducedNode.CLEAR, _):
                predicates.append(
                    {
                        "typing": "clear",
                        "parameters": [v.node],
                        "scene": edge.scene,
                    }
                )
            case (_, ReducedNode.TABLE):
                predicates.append(
                    {
                        "typing": "on-table",
                        "parameters": [u.node],
                        "scene": edge.scene,
                    }
                )
            case (_, ReducedNode.ARM):
                predicates.append(
                    {
                        "typing": "holding",
                        "parameters": [u.node],
                        "scene": edge.scene,
                    }
                )
            case (_, _):
                predicates.append(
                    {
                        "typing": "on",
                        "parameters": [u.node, v.node],
                        "scene": edge.scene,
                    }
                )

    if isinstance(scene, ReducedProblemGraph):
        return graph.ProblemGraph(
            constants,
            [pred for pred in predicates if pred["scene"] == graph.Scene.INIT],
            [pred for pred in predicates if pred["scene"] == graph.Scene.GOAL],
            domain="blocksworld",
            requirements=scene._requirements,
        )
    else:
        return graph.SceneGraph(
            constants,
            predicates,
            domain="blocksworld",
            scene=scene.scene,
            requirements=scene._requirements,
        )


def _blocksworld_underspecified_blocks(
    scene: ReducedSceneGraph,
) -> tuple[set[str], set[str], bool]:
    """Finds blocks that are not fully specified.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        tuple[set[str], set[str], bool]: The set of blocks that are not fully
        specified.
         - blocks that do not specify what is on top.
         - blocks that do not specify what is on the bottom.
    """
    top_blocks = set()
    bottom_blocks = set()
    arm_behavior_defined = scene.in_degree(ReducedNode.ARM) > 0
    held_block = (
        scene.predecessors(ReducedNode.ARM)[0] if arm_behavior_defined else None
    )
    for node in scene.nodes:
        if node.label == graph.Label.CONSTANT:
            if not scene.in_edges(node) and node != held_block:
                top_blocks.add(node)
            if not scene.out_edges(node):
                bottom_blocks.add(node)
    return top_blocks, bottom_blocks, not arm_behavior_defined


def _detached_blocks(
    nodesA: set[str],
    nodesB: set[str],
    scene: ReducedSceneGraph,
) -> tuple[set[str], set[str]]:
    """Finds nodes that are not connected to the rest of the scene graph.

    Args:
        nodesA (set[str]): The set of nodes to check.
        nodesB (set[str]): The set of nodes to check against.
        scene (ReducedGraph): The scene graph to check against.

    Returns:
        tuple[set[str], set[str]]: The set of nodes that are not connected to
        the rest of the scene graph.
    """
    _nodesA = set(nodesA)
    _nodesB = set(nodesB)

    for a in nodesA:
        for b in nodesB:
            a_index = scene.nodes.index(a)
            b_index = scene.nodes.index(b)
            if (
                not rx.has_path(scene.graph, a_index, b_index)
                and not rx.has_path(scene.graph, b_index, a_index)
                and a != b
            ):
                _nodesA.discard(a)
                _nodesB.discard(b)

    return _nodesA, _nodesB


def _fully_specify_blocksworld(
    scene: ReducedSceneGraph,
) -> graph.SceneGraph:
    """Fully specifies a blocksworld scene graph.

    Adds any missing edges to fully specify the scene graph, without adding
    edges that change the problem represented by the graph.

    Args:
        scene (ReducedGraph): The reduced SceneGraph of a scene.

    Returns:
        SceneGraph: The fully specified scene graph.
    """
    scene = copy.deepcopy(scene)
    top_blocks, bottom_blocks, arm_empty = _blocksworld_underspecified_blocks(scene)
    top_blocks_, bottom_blocks_ = _detached_blocks(top_blocks, bottom_blocks, scene)

    for block in top_blocks_:
        scene.add_edge(
            ReducedNode.CLEAR,
            block,
            graph.PlanGraphEdge(predicate="clear", scene=scene.scene),
        )
    for block in bottom_blocks_:
        scene.add_edge(
            block,
            ReducedNode.TABLE,
            graph.PlanGraphEdge(predicate="on-table", scene=scene.scene),
        )

    # handle arm
    if arm_empty and not (top_blocks & bottom_blocks):
        scene.add_edge(
            ReducedNode.CLEAR,
            ReducedNode.ARM,
            graph.PlanGraphEdge(predicate="arm-empty", scene=scene.scene),
        )

    return scene


def _plan_blocksworld(problem: ReducedProblemGraph) -> list[Action]:
    init, goal = problem.decompose()
    actions = []

    # Process init scene
    # check if arm is empty
    if (
        not init.has_edge(ReducedNode.CLEAR, ReducedNode.ARM)
        and init.in_degree(ReducedNode.ARM) == 1
    ):
        obj = init.predecessors(ReducedNode.ARM)[0]
        actions.append(Action("putdown", [obj.name]))

    # unstack everything in init
    for idx in rx.topological_sort(init.graph):
        node = init.nodes[idx]
        if isinstance(node.node, ReducedNode):
            continue
        elif init.successors(node)[0].name in (ReducedNode.ARM, ReducedNode.TABLE):
            # if the block is on the table or in the arm, ignore it
            continue
        else:
            actions.append(
                Action("unstack", [node.name, init.successors(node)[0].name])
            )
            actions.append(Action("putdown", [node.name]))

    # Process goal scene
    # stack everything in goal
    for idx in reversed(rx.topological_sort(goal.graph)):
        node = goal.nodes[idx]
        if isinstance(node.node, ReducedNode):
            continue
        elif goal.out_degree(node.node) == 0:
            # isn't defined to be on anything (keep on table)
            continue
        elif goal.successors(node)[0].node in (ReducedNode.ARM, ReducedNode.TABLE):
            # if the block is on the table or in the arm, ignore it
            continue
        else:
            actions.append(Action("pickup", [node.name]))
            actions.append(Action("stack", [node.name, goal.successors(node)[0].name]))

    # Check if arm should be holding it
    if (
        not goal.has_edge(ReducedNode.CLEAR, ReducedNode.ARM)
        and goal.in_degree(ReducedNode.ARM) == 1
    ):
        obj = goal.predecessors(ReducedNode.ARM)[0]
        actions.append(Action("pickup", [obj.name]))

    return actions
