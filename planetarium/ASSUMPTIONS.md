# Assumptions

For each domain, the following assumptions are made regarding the types of problems evaluated.
We also assume that problems are solvable via the domain's actions.

Our equivalence check is based on the assumption that the problems are solvable already (i.e. our evaluator checks solvability before equivalence).

## Blocksworld

Generally, since Blocks World has reversible actions, essentially all problems are evaluable.

`:init` conditions:
- No two blocks are on top of a single block
- No block is initialized on top of two blocks
- No loops of blocks are made
- Arm can only hold one block

## Grippers

Generally, since Grippers has reversible actions, essentially all problems are evaluable.

`:init` conditions:
- No double "typing" of objects (we don't using `:typing`, we use certain immutable predicates)
- All balls have only one location
- All grippers have only one location
- All grippers hold up to 1 ball

## Rover Single
Rover has the capability of being a much more complex domain, but for the purposes of this benchmark, we work only with a single rover and a single lander.

`:init` conditions:
- No double `at_*` predicates (rover and lander can only be in one location at a time)

## Floortile

Generally, all valid problems with reachable goal states are evaluable.

`:init` conditions:
- No robot has two colors (`robot-has`)
