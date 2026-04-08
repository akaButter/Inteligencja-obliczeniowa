"""
STRIPS Planning Framework - Self-contained Implementation
Standalone (no external GUI dependencies)
"""

import sys
import time
import heapq
from collections import deque

# ==================================================================================
# BASIC STRIPS CLASSES
# ==================================================================================

class Strips:
    """Represent a STRIPS action"""
    def __init__(self, name, preconds, effects, cost=1):
        self.name = name
        self.preconds = preconds  # dict
        self.effects = effects     # dict
        self.cost = cost

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class STRIPS_domain:
    """Represent a STRIPS planning domain"""
    def __init__(self, feature_domain_dict, actions):
        self.feature_domain_dict = feature_domain_dict
        self.actions = actions


class Planning_problem:
    """Represent a STRIPS planning problem"""
    def __init__(self, prob_domain, initial_state, goal):
        self.prob_domain = prob_domain
        self.initial_state = initial_state
        self.goal = goal


class State:
    """Represent a planning state"""
    def __init__(self, assignment):
        self.assignment = dict(assignment)
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self.assignment.items()))
        return self._hash

    def __eq__(self, other):
        return isinstance(other, State) and self.assignment == other.assignment

    def __repr__(self):
        return f"S({len(self.assignment)} facts)"


# ==================================================================================
# FORWARD PLANNERS
# ==================================================================================

def forward_planner_bfs(problem, timeout=120):
    """
    Forward planning using BFS (no heuristic)
    Returns: (actions, elapsed_time, expanded_nodes) or (None, elapsed_time, expanded_nodes)
    """
    start_time = time.time()
    initial = State(problem.initial_state)

    # Check if initial state is goal
    if all(initial.assignment.get(k) == v for k, v in problem.goal.items()):
        return [], 0.0, 0

    queue = deque([(initial, [])])
    visited = {initial}
    expanded = 0

    while queue:
        if time.time() - start_time > timeout:
            return None, timeout, expanded

        state, actions = queue.popleft()
        expanded += 1

        # Generate successors
        for action in problem.prob_domain.actions:
            # Check if action is applicable
            if not all(state.assignment.get(k) == v for k, v in action.preconds.items()):
                continue

            # Apply action effects
            new_assign = state.assignment.copy()
            new_assign.update(action.effects)
            new_state = State(new_assign)

            if new_state not in visited:
                new_actions = actions + [action]

                # Check if goal is reached
                if all(new_state.assignment.get(k) == v for k, v in problem.goal.items()):
                    elapsed = time.time() - start_time
                    return new_actions, elapsed, expanded

                visited.add(new_state)
                queue.append((new_state, new_actions))

    elapsed = time.time() - start_time
    return None, elapsed, expanded


def forward_planner_astar(problem, heuristic, timeout=120):
    """
    Forward planning using A* with heuristic
    Returns: (actions, elapsed_time, expanded_nodes) or (None, elapsed_time, expanded_nodes)
    """
    start_time = time.time()
    initial = State(problem.initial_state)

    # Check if initial state is goal
    if all(initial.assignment.get(k) == v for k, v in problem.goal.items()):
        return [], 0.0, 0

    # Heap: (f_score, counter, state, actions)
    counter = 0
    h_initial = heuristic(initial.assignment, problem.goal)
    heap = [(h_initial, counter, initial, [])]
    visited = {initial: 0}
    expanded = 0

    while heap:
        if time.time() - start_time > timeout:
            return None, timeout, expanded

        _, _, state, actions = heapq.heappop(heap)
        expanded += 1

        # Check if goal
        if all(state.assignment.get(k) == v for k, v in problem.goal.items()):
            elapsed = time.time() - start_time
            return actions, elapsed, expanded

        # Generate successors
        for action in problem.prob_domain.actions:
            if not all(state.assignment.get(k) == v for k, v in action.preconds.items()):
                continue

            new_assign = state.assignment.copy()
            new_assign.update(action.effects)
            new_state = State(new_assign)

            new_actions = actions + [action]
            new_g = len(new_actions)

            if new_state not in visited or visited[new_state] > new_g:
                visited[new_state] = new_g
                h = heuristic(new_state.assignment, problem.goal)
                f = new_g + h
                counter += 1
                heapq.heappush(heap, (f, counter, new_state, new_actions))

    elapsed = time.time() - start_time
    return None, elapsed, expanded


def forward_planner_astar_subgoals(problem, heuristic, subgoals, timeout=120):
    """
    Forward planning with A* through series of subgoals
    subgoals: list of dict (intermediate goal states)
    """
    start_time = time.time()
    current_state_dict = dict(problem.initial_state)
    all_actions = []

    # Add final goal to subgoals
    goals_sequence = list(subgoals) + [problem.goal]

    for subgoal_idx, subgoal in enumerate(goals_sequence):
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time <= 0:
            return None, time.time() - start_time, -1

        # Create temporary problem
        temp_problem = Planning_problem(
            problem.prob_domain,
            current_state_dict,
            subgoal
        )

        # Solve
        actions, _, _ = forward_planner_astar(temp_problem, heuristic, remaining_time)

        if actions is None:
            return None, time.time() - start_time, -1

        all_actions.extend(actions)

        # Update current state for next subgoal
        for action in actions:
            current_state_dict.update(action.effects)

    elapsed = time.time() - start_time
    return all_actions, elapsed, len(all_actions)


# ==================================================================================
# HEURISTIC FUNCTIONS
# ==================================================================================

def heuristic_unsatisfied_goals(state_dict, goal_dict):
    """Count unsatisfied goal facts"""
    count = 0
    for k, v in goal_dict.items():
        if state_dict.get(k) != v:
            count += 1
    return count


# Export for use in problem definitions
__all__ = ['Strips', 'STRIPS_domain', 'Planning_problem', 'State',
           'forward_planner_bfs', 'forward_planner_astar', 'forward_planner_astar_subgoals',
           'heuristic_unsatisfied_goals']
