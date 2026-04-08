"""
BLOCKSWORD (Blocks World - Extended)
Domain: Blocks world planning with blocks that can be moved
Source: https://github.com/primaryobjects/strips/blob/master/examples/blocksworld4

Problem: Arrange blocks on tables by moving them
- Blocks can be on other blocks or on tables
- Only the topmost block can be moved
- Complex multi-goal arrangement problems
"""

from aipython_strips import Strips, STRIPS_domain, Planning_problem

# ==================================================================================
# BLOCKSWORD DOMAIN DEFINITION
# ==================================================================================

def create_blocks_world_domain(blocks, tables):
    """
    Create a blocks world domain.
    Features:
    - on_<block>: location (block or table) - where the block is
    - clear_<obj>: {True, False} - whether obj has nothing on top
    """

    boolean = {True, False}
    blocks_set = set(blocks)
    tables_set = set(tables)
    blocks_and_tables = blocks_set | tables_set

    feature_domain_dict = {}

    # For each block, track what it's on
    for block in blocks_set:
        possible_locations = blocks_and_tables - {block}
        feature_domain_dict[f'on_{block}'] = possible_locations

    # For each block and table, whether it's clear (nothing on top)
    for obj in blocks_and_tables:
        feature_domain_dict[f'clear_{obj}'] = boolean

    # Create move actions
    actions = set()

    # Move block X from Y to Z
    # Preconditions:
    #   - X is on Y
    #   - X is clear
    #   - Z is clear
    # Effects:
    #   - X is on Z
    #   - Y is clear
    #   - Z is not clear

    for x in blocks_set:
        for y in blocks_and_tables:
            if x == y:
                continue
            for z in blocks_and_tables:
                if z == x or z == y:
                    continue

                action_name = f'move_{x}_from_{y}_to_{z}'
                preconds = {
                    f'on_{x}': y,
                    f'clear_{x}': True,
                    f'clear_{z}': True,
                }
                effects = {
                    f'on_{x}': z,
                    f'clear_{y}': True,
                    f'clear_{z}': False,
                }
                actions.add(Strips(action_name, preconds, effects))

    return STRIPS_domain(feature_domain_dict, actions)


def blocksword_problem_1():
    """
    PROBLEM 1 (Basic - 3 blocks, 2 tables)
    Initial: c on b on a on table1
    Goal: a on table1; b on table2; c on table1
    - Simple rearrangement
    - ~3-4 actions minimum
    """
    blocks = ['a', 'b', 'c']
    tables = ['t1', 't2']
    domain = create_blocks_world_domain(blocks, tables)

    initial_state = {
        'on_a': 't1',
        'on_b': 'a',
        'on_c': 'b',
        'clear_a': False,
        'clear_b': False,
        'clear_c': True,
        'clear_t1': False,
        'clear_t2': True,
    }

    goal = {
        'on_a': 't1',
        'on_b': 't2',
        'on_c': 't1',
    }

    return Planning_problem(domain, initial_state, goal)


def blocksword_problem_2():
    """
    PROBLEM 2 (Medium - 4 blocks, 2 tables)
    Initial: b on a on t1; d on c on t2
    Goal: a on t1, b on t2, c on b, d on t1
    - Requires several moves
    - ~6-8 actions minimum
    """
    blocks = ['a', 'b', 'c', 'd']
    tables = ['t1', 't2']
    domain = create_blocks_world_domain(blocks, tables)

    initial_state = {
        'on_a': 't1',
        'on_b': 'a',
        'on_c': 't2',
        'on_d': 'c',
        'clear_a': False,
        'clear_b': True,
        'clear_c': False,
        'clear_d': True,
        'clear_t1': False,
        'clear_t2': False,
    }

    goal = {
        'on_a': 't1',
        'on_b': 't2',
        'on_c': 'b',
        'on_d': 't1',
    }

    return Planning_problem(domain, initial_state, goal)


def blocksword_problem_3():
    """
    PROBLEM 3 (Complex - 8 blocks, 4 tables)
    Initial: Blocks stacked on t1 and t2
    Goal: Different arrangement across 4 tables
    - Complex multi-goal arrangement
    - ~20+ actions minimum
    """
    blocks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    tables = ['t1', 't2', 't3', 't4']
    domain = create_blocks_world_domain(blocks, tables)

    initial_state = {
        'on_a': 't1',
        'on_b': 'a',
        'on_c': 'b',
        'on_d': 'c',
        'on_e': 't2',
        'on_f': 'e',
        'on_g': 'f',
        'on_h': 'g',
        'clear_a': False,
        'clear_b': False,
        'clear_c': False,
        'clear_d': True,
        'clear_e': False,
        'clear_f': False,
        'clear_g': False,
        'clear_h': True,
        'clear_t1': False,
        'clear_t2': False,
        'clear_t3': True,
        'clear_t4': True,
    }

    goal = {
        'on_a': 't1',
        'on_b': 'a',
        'on_c': 't2',
        'on_d': 'c',
        'on_e': 't3',
        'on_f': 'e',
        'on_g': 't4',
        'on_h': 'g',
    }

    return Planning_problem(domain, initial_state, goal)


def get_blocksword_subgoals():
    """
    Subgoals for blocksword problems:
    1. Problem 1: (a) Clear top blocks, (b) Move to tables
    2. Problem 2: (a) Separate blocks, (b) Move to destinations
    3. Problem 3: (a) Clear top blocks, (b) Distribute to tables
    """
    return {
        'blocksword_1': [
            {'clear_c': True},
            {'on_b': 't2'},
        ],
        'blocksword_2': [
            {'clear_d': True, 'clear_b': True},
            {'on_b': 't2', 'on_d': 't1'},
        ],
        'blocksword_3': [
            {'clear_d': True, 'clear_h': True},
            {'on_d': 't1', 'on_h': 't2'},
        ],
    }


def heuristic_blocksword(state_dict, goal_dict):
    """
    Heuristic for blocks world:
    Count how many blocks are not in their goal position.
    """
    blocked_count = 0

    for block_key, goal_loc in goal_dict.items():
        if block_key in state_dict:
            if state_dict[block_key] != goal_loc:
                blocked_count += 1

            # Check if block is clear
            clear_key = block_key.replace('on_', 'clear_')
            if clear_key in state_dict and not state_dict[clear_key]:
                blocked_count += 1

    return blocked_count
