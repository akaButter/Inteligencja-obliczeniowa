"""
BLOCKSWORD (Blocks World - Corrected)
Achievable goals - bloki mogą być na sobie lub na stołach
"""

from aipython_strips import Strips, STRIPS_domain, Planning_problem

def create_blocks_world_domain(blocks, tables):
    """Create blocks world domain"""
    boolean = {True, False}
    blocks_set = set(blocks)
    tables_set = set(tables)
    blocks_and_tables = blocks_set | tables_set

    feature_domain_dict = {}

    for block in blocks_set:
        possible_locations = blocks_and_tables - {block}
        feature_domain_dict[f'on_{block}'] = possible_locations

    for obj in blocks_and_tables:
        feature_domain_dict[f'clear_{obj}'] = boolean

    actions = set()

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
    PROBLEM 1
    Initial: a on t1, b on t2, c on t3
    Goal: a on t1, b on a, c on t2
    Solution: move_b_from_t2_to_t1, then move_c_from_t3_to_t2
    Requires: ~4 actions
    """
    blocks = ['a', 'b', 'c']
    tables = ['t1', 't2', 't3']
    domain = create_blocks_world_domain(blocks, tables)

    initial_state = {
        'on_a': 't1',
        'on_b': 't2',
        'on_c': 't3',
        'clear_a': True,
        'clear_b': True,
        'clear_c': True,
        'clear_t1': False,
        'clear_t2': False,
        'clear_t3': False,
    }

    # Achievable: b on a, c on t2, a on t1
    goal = {
        'on_a': 't1',
        'on_b': 'a',
        'on_c': 't2',
    }

    return Planning_problem(domain, initial_state, goal)


def blocksword_problem_2():
    """
    PROBLEM 2
    Initial: a on t1, b on t2, c on t3, d on t4
    Goal: a on b, c on d, b on t1, d on t2
    Solution: needs to stack blocks and rearrange
    Requires: ~6-8 actions
    """
    blocks = ['a', 'b', 'c', 'd']
    tables = ['t1', 't2', 't3']
    domain = create_blocks_world_domain(blocks, tables)

    initial_state = {
        'on_a': 't1',
        'on_b': 't2',
        'on_c': 't3',
        'on_d': 't1',
        'clear_a': True,
        'clear_b': True,
        'clear_c': True,
        'clear_d': True,
        'clear_t1': False,
        'clear_t2': False,
        'clear_t3': False,
    }

    goal = {
        'on_a': 'b',
        'on_b': 't1',
        'on_c': 'd',
        'on_d': 't2',
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
    return {
        'blocksword_1': [
            {'clear_b': True},
            {'on_b': 't3'},
        ],
        'blocksword_2': [
            {'clear_a': True, 'clear_c': True},
            {'on_a': 'b', 'on_c': 'd'},
        ],
        'blocksword_3': [
            {'clear_d': True, 'clear_h': True},
            {'on_d': 't1', 'on_h': 't2', 'on_b': 't3'},
        ],
    }


def heuristic_blocksword(state_dict, goal_dict):
    """Count blocks not in goal + blocked blocks"""
    count = 0
    for key, goal_val in goal_dict.items():
        if state_dict.get(key) != goal_val:
            count += 1
            block = key.replace('on_', '')
            if not state_dict.get(f'clear_{block}', True):
                count += 1
    return count
