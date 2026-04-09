import time

# Import standalone STRIPS framework
from aipython_strips import (forward_planner_bfs, forward_planner_astar,
                             forward_planner_astar_subgoals,
                             heuristic_unsatisfied_goals, forward_planner_subgoals)

# Import problem domains
from dinner import (dinner_problem_1, dinner_problem_2, dinner_problem_3,
                   get_dinner_subgoals, heuristic_dinner)
from magicworld import (magicworld_problem_1, magicworld_problem_2, magicworld_problem_3,
                       get_magicworld_subgoals, heuristic_magicworld)
from blocksword4 import (blocksword_problem_1, blocksword_problem_2, blocksword_problem_3,
                        get_blocksword_subgoals, heuristic_blocksword)

# ==================================================================================
# UTILITIES
# ==================================================================================

def solve_problem_bfs(problem, problem_name, timeout=300):
    """Solve problem using BFS (no heuristic)"""
    print(f"\n[BFS] Solving: {problem_name}")
    start_time = time.time()

    try:
        actions, elapsed, expanded = forward_planner_bfs(problem, timeout)

        if actions is not None:
            print(f"    Found solution: {len(actions)} steps in {elapsed:.4f}s")
            print(f"    Actions: {', '.join([str(a) for a in actions])}")

            return {
                'status': 'success',
                'solution': [str(a) for a in actions],
                'steps': len(actions),
                'time': elapsed,
                'expanded': expanded,
            }
        else:
            print(f"    No solution found (timeout after {elapsed:.2f}s)")
            return {
                'status': 'timeout' if elapsed >= timeout else 'no_solution',
                'time': elapsed,
                'expanded': expanded,
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    Error: {str(e)[:100]}")
        return {
            'status': 'error',
            'error': str(e)[:200],
            'time': elapsed,
        }


def solve_problem_astar(problem, heuristic, problem_name, timeout=300):
    """Solve problem using A* with heuristic"""
    print(f"[A*]  Solving: {problem_name} (with heuristic)")
    start_time = time.time()

    try:
        actions, elapsed, expanded = forward_planner_astar(problem, heuristic, timeout)

        if actions is not None:
            print(f"    Found solution: {len(actions)} steps in {elapsed:.4f}s")
            print(f"    Actions: {', '.join([str(a) for a in actions])}")

            return {
                'status': 'success',
                'solution': [str(a) for a in actions],
                'steps': len(actions),
                'time': elapsed,
                'expanded': expanded,
            }
        else:
            print(f"    No solution found (timeout after {elapsed:.2f}s)")
            return {
                'status': 'timeout' if elapsed >= timeout else 'no_solution',
                'time': elapsed,
                'expanded': expanded,
            }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    Error: {str(e)[:100]}")
        return {
            'status': 'error',
            'error': str(e)[:200],
            'time': elapsed,
        }


def solve_problem_with_subgoals_astar(problem, heuristic, subgoals, problem_name, timeout=300):
    """Solve problem using A* with subgoals"""
    print(f"[A*+SG] Solving: {problem_name} (with {len(subgoals)} subgoals)")

    try:
        actions, elapsed, steps = forward_planner_astar_subgoals(
            problem, heuristic, subgoals, timeout
        )

        if actions is not None:
            print(f"    All subgoals achieved: {len(actions)} total steps in {elapsed:.4f}s")

            return {
                'status': 'success',
                'solution': [str(a) for a in actions],
                'steps': len(actions),
                'time': elapsed,
            }
        else:
            print(f"    Failed to achieve all subgoals (time: {elapsed:.2f}s)")
            return {
                'status': 'failed' if elapsed < timeout else 'timeout',
                'time': elapsed,
            }

    except Exception as e:
        print(f"    Error: {str(e)[:100]}")
        return {
            'status': 'error',
            'error': str(e)[:200],
        }

def solve_problem_with_subgoals(problem, subgoals, problem_name, timeout=300):
    """Solve problem with subgoals"""
    print(f"[BFS+SG] Solving: {problem_name} (with {len(subgoals)} subgoals)")

    try:
        actions, elapsed, steps = forward_planner_subgoals(
            problem, subgoals, timeout
        )

        if actions is not None:
            print(f"    All subgoals achieved: {len(actions)} total steps in {elapsed:.4f}s")

            return {
                'status': 'success',
                'solution': [str(a) for a in actions],
                'steps': len(actions),
                'time': elapsed,
            }
        else:
            print(f"    Failed to achieve all subgoals (time: {elapsed:.2f}s)")
            return {
                'status': 'failed' if elapsed < timeout else 'timeout',
                'time': elapsed,
            }

    except Exception as e:
        print(f"    Error: {str(e)[:100]}")
        return {
            'status': 'error',
            'error': str(e)[:200],
        }
# ==================================================================================
# MAIN RUNNER
# ==================================================================================

def main():
    """Run all problems for all domains"""

    print("\n" + "="*90)
    print("STRIPS PLANNING - KOMPLETNE ROZWIĄZANIE".center(90))
    print("Domeny: Dinner | Magicworld | Blocksword".center(90))
    print("="*90)

    results = {
        'dinner': [],
        'magicworld': [],
        'blocksword': [],
        'subgoals': [],
    }

    # ========== DOMAIN 1: DINNER ==========
    print("\n" + "#"*90)
    print("# DOMENA 1: DINNER (Przygotowanie Kolacji)".ljust(90))
    print("#"*90)

    dinner_problems = [
        ('dinner_1', dinner_problem_1(), heuristic_dinner),
        ('dinner_2', dinner_problem_2(), heuristic_dinner),
        ('dinner_3', dinner_problem_3(), heuristic_dinner),
    ]

    for name, problem, heuristic in dinner_problems:
        print(f"\n--- {name:20} (goal: {len(problem.goal)} facts) ---")

        result = {
            'name': name,
            'domain': 'dinner',
            'initial_facts': len(problem.initial_state),
            'goal_facts': len(problem.goal),
        }

        # BFS
        bfs_result = solve_problem_bfs(problem, name, timeout=300)
        result['bfs'] = bfs_result

        # A* with heuristic
        astar_result = solve_problem_astar(problem, heuristic, name, timeout=300)
        result['astar'] = astar_result

        results['dinner'].append(result)

    # ========== DOMAIN 2: MAGICWORLD ==========
    print("\n" + "#"*90)
    print("# DOMENA 2: MAGICWORLD (Zaklęcia i Mikstury)".ljust(90))
    print("#"*90)

    magic_problems = [
        ('magicworld_1', magicworld_problem_1(), heuristic_magicworld),
        ('magicworld_2', magicworld_problem_2(), heuristic_magicworld),
        ('magicworld_3', magicworld_problem_3(), heuristic_magicworld),
    ]

    for name, problem, heuristic in magic_problems:
        print(f"\n--- {name:20} (goal: {len(problem.goal)} facts) ---")

        result = {
            'name': name,
            'domain': 'magicworld',
            'initial_facts': len(problem.initial_state),
            'goal_facts': len(problem.goal),
        }

        bfs_result = solve_problem_bfs(problem, name, timeout=300)
        result['bfs'] = bfs_result

        astar_result = solve_problem_astar(problem, heuristic, name, timeout=300)
        result['astar'] = astar_result

        results['magicworld'].append(result)

    # ========== DOMAIN 3: BLOCKSWORD ==========
    print("\n" + "#"*90)
    print("# DOMENA 3: BLOCKSWORD (Blocks World)".ljust(90))
    print("#"*90)

    blocks_problems = [
        ('blocksword_1', blocksword_problem_1(), heuristic_blocksword),
        ('blocksword_2', blocksword_problem_2(), heuristic_blocksword),
        ('blocksword_3', blocksword_problem_3(), heuristic_blocksword),
    ]

    for name, problem, heuristic in blocks_problems:
        print(f"\n--- {name:20} (goal: {len(problem.goal)} facts) ---")

        result = {
            'name': name,
            'domain': 'blocksword',
            'initial_facts': len(problem.initial_state),
            'goal_facts': len(problem.goal),
        }

        bfs_result = solve_problem_bfs(problem, name, timeout=300)
        result['bfs'] = bfs_result

        astar_result = solve_problem_astar(problem, heuristic, name, timeout=300)
        result['astar'] = astar_result

        results['blocksword'].append(result)

    # ========== PROBLEMS WITH SUBGOALS (6-POINT TASKS) ==========
    print("\n" + "#"*90)
    print("# ZADANIA NA 6 PUNKTÓW: Problemy z podecelami".ljust(90))
    print("#"*90)

    dinner_subgoals_dict = get_dinner_subgoals()
    magic_subgoals_dict = get_magicworld_subgoals()
    blocks_subgoals_dict = get_blocksword_subgoals()

    # Dinner with subgoals
    for idx, (name, problem, heuristic) in enumerate(dinner_problems):
        key = f'dinner_{idx+1}'
        if key in dinner_subgoals_dict:
            print(f"\n--- {name}_subgoals ---")
            result2 = solve_problem_with_subgoals_astar(
                problem, heuristic, dinner_subgoals_dict[key], name, timeout=300
            )
            result1 = solve_problem_with_subgoals(
                problem, dinner_subgoals_dict[key], name, timeout=300
            )
            result1['name'] = name + '_subgoals'
            result1['domain'] = 'dinner'
            results['subgoals'].append([result1, result2])

    # Magicworld with subgoals
    for idx, (name, problem, heuristic) in enumerate(magic_problems):
        key = f'magicworld_{idx+1}'
        if key in magic_subgoals_dict:
            print(f"\n--- {name}_subgoals ---")
            result2 = solve_problem_with_subgoals_astar(
                problem, heuristic, magic_subgoals_dict[key], name, timeout=300
            )
            result1 = solve_problem_with_subgoals(
                problem, magic_subgoals_dict[key], name, timeout=300
            )
            result1['name'] = name + '_subgoals'
            result1['domain'] = 'magicworld'
            results['subgoals'].append([result1, result2])

    # Blocksword with subgoals
    for idx, (name, problem, heuristic) in enumerate(blocks_problems):
        key = f'blocksword_{idx+1}'
        if key in blocks_subgoals_dict:
            print(f"\n--- {name}_subgoals ---")
            result2 = solve_problem_with_subgoals_astar(
                problem, heuristic, blocks_subgoals_dict[key], name, timeout=300
            )
            result1 = solve_problem_with_subgoals(
                problem, blocks_subgoals_dict[key], name, timeout=300
            )
            result1['name'] = name + '_subgoals'
            result1['domain'] = 'blocksword'
            results['subgoals'].append([result1, result2])

    # ========== SUMMARY ==========
    print("\n" + "="*90)
    print("PODSUMOWANIE WYNIKÓW".ljust(90))
    print("="*90)

    print_summary(results)

    return results


def print_summary(results):
    """Print summary table of results"""

    print("\n[4-POINT TASKS] Basic Problems - BFS vs A*")
    print("-" * 110)
    print(f"{'Problem':20} | {'Facts':6} | {'BFS Steps':12} | {'A* Steps':12} | {'BFS time':10} | {'A* time':10}")
    print("-" * 110)

    for domain_results in [results['dinner'], results['magicworld'], results['blocksword']]:
        for r in domain_results:
            bfs_steps = f"{r['bfs']['steps']}" if r['bfs']['status'] == 'success' else "TIMEOUT"
            astar_steps = f"{r['astar']['steps']}" if r['astar']['status'] == 'success' else "TIMEOUT"
            bfs_time = f"{r['bfs']['time']:.3f}s" if r['bfs']['status'] == 'success' else "TIMEOUT"
            astar_time = f"{r['astar']['time']:.3f}s" if r['astar']['status'] == 'success' else "TIMEOUT"

            print(f"{r['name']:20} | {r['initial_facts']:6} | {bfs_steps:>12} | {astar_steps:>12} | "
                  f"{bfs_time:>10} | {astar_time:>10}")

    print("\n[6-POINT TASKS] Problems with Subgoals")
    print("-" * 100)
    print(f"{'Problem':30} | {'Steps BFS':8} | {'Time BFS':10} | {'Steps A*':8} | {'Time A*':10}")
    print("-" * 100)

    for r in results['subgoals']:
        bfs_steps = f"{r[0]['steps']}" if r[0]['status'] == 'success' else r[0]['status']
        bfs_time = f"{r[0]['time']:.4f}s" if r[0]['status'] == 'success' else "-"
        astar_steps = f"{r[1]['steps']}" if r[1]['status'] == 'success' else r[1]['status']
        astar_time = f"{r[1]['time']:.4f}s" if r[1]['status'] == 'success' else "-"

        print(f"{r[0]['name']:30} | {bfs_steps:>8} | {bfs_time:>10} | {astar_steps:>8} | {astar_time:>10}")


# Performance metrics
def print_performance_analysis(results):
    """Analyze and print performance metrics"""
    print("\n" + "="*90)
    print("ANALIZA WYDAJNOŚCI HEURYSTYK".ljust(90))
    print("="*90)

    for domain in ['dinner', 'magicworld', 'blocksword']:
        domain_results = results[domain]
        print(f"\nDomena: {domain.upper()}")
        print("-" * 50)

        total_bfs_time = sum(r['bfs'].get('time', 0) for r in domain_results if r['bfs']['status'] == 'success')
        total_astar_time = sum(r['astar'].get('time', 0) for r in domain_results if r['astar']['status'] == 'success')
        total_bfs_steps = sum(r['bfs'].get('steps', 0) for r in domain_results if r['bfs']['status'] == 'success')
        total_astar_steps = sum(r['astar'].get('steps', 0) for r in domain_results if r['astar']['status'] == 'success')

        print(f"BFS:  Razem {total_bfs_steps} kroków w {total_bfs_time:.4f}s")
        print(f"A*:   Razem {total_astar_steps} kroków w {total_astar_time:.4f}s")

        if total_bfs_time > 0:
            speedup = total_bfs_time / total_astar_time
            print(f"Przyspieszenie A* względem BFS: {speedup:.2f}x")


if __name__ == '__main__':
    results = main()
    print_performance_analysis(results)

