"""
=============================================================================
PLANOWANIE STRIPS - Rozwiązanie zadań na 4, 6 i 8 punktów
=============================================================================
Implementacja oparta na AIPython (Poole & Mackworth) 
Zawiera:
  - Implementację STRIPS (stripsProblem + stripsForwardPlanner z A*)
  - 3 domeny: Air Cargo, Dock Worker Robot, 1D Rubik's Cube
  - Dla każdej domeny: 3 problemy (problem bazowy + 2 rozszerzone)
  - Forward planning bez i z heurystyką
  - Podecele (subgoals) - zadanie na 6 pkt
  - Problemy wymagające min. 20 akcji - zadanie na 8 pkt
=============================================================================
"""

import time
import heapq
from collections import deque

# =============================================================================
# IMPLEMENTACJA STRIPS (na wzór AIPython stripsProblem.py)
# =============================================================================

class STRIPS_domain:
    def __init__(self, name, feature_domain_dict, actions):
        self.name = name
        self.feature_domain_dict = feature_domain_dict  # {feature: domain}
        self.actions = actions

class Action:
    def __init__(self, name, preconds, effects):
        self.name = name
        self.preconds = preconds  # frozenset of (feature, value) that must be True
        self.effects = effects    # dict {feature: value} changes

    def __repr__(self):
        return self.name

class Planning_problem:
    def __init__(self, domain, initial_state, goal):
        self.domain = domain
        self.initial_state = frozenset(initial_state)
        self.goal = frozenset(goal)

    def is_goal(self, state):
        return self.goal.issubset(state)

    def get_successors(self, state):
        """Zwraca listę (action, new_state)"""
        successors = []
        for action in self.domain.actions:
            if action.preconds.issubset(state):
                # Zastosuj efekty
                new_state = set(state)
                for feat, val in action.effects.items():
                    # Usuń stare wartości tej cechy
                    to_remove = {(f, v) for (f, v) in new_state if f == feat}
                    new_state -= to_remove
                    if val is not None:
                        new_state.add((feat, val))
                successors.append((action, frozenset(new_state)))
        return successors

# =============================================================================
# PLANERY
# =============================================================================

def forward_planner_bfs(problem, timeout=300):
    """Forward planner BFS (bez heurystyki)"""
    start_time = time.time()
    start = problem.initial_state
    if problem.is_goal(start):
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    nodes_expanded = 0
    
    while queue:
        if time.time() - start_time > timeout:
            return None  # timeout
        
        state, path = queue.popleft()
        nodes_expanded += 1
        
        for action, next_state in problem.get_successors(state):
            if next_state not in visited:
                new_path = path + [action]
                if problem.is_goal(next_state):
                    elapsed = time.time() - start_time
                    return new_path, elapsed, nodes_expanded
                visited.add(next_state)
                queue.append((next_state, new_path))
    
    return None, time.time() - start_time, nodes_expanded

def forward_planner_astar(problem, heuristic, timeout=300):
    """Forward planner A* z heurystyką"""
    start_time = time.time()
    start = problem.initial_state
    if problem.is_goal(start):
        return [], 0, 0
    
    # (f, g, state, path)
    counter = 0
    heap = [(heuristic(start, problem.goal), 0, counter, start, [])]
    visited = {}
    nodes_expanded = 0
    
    while heap:
        if time.time() - start_time > timeout:
            return None, timeout, nodes_expanded
        
        f, g, _, state, path = heapq.heappop(heap)
        
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        nodes_expanded += 1
        
        for action, next_state in problem.get_successors(state):
            new_g = g + 1
            if next_state not in visited or visited[next_state] > new_g:
                new_path = path + [action]
                if problem.is_goal(next_state):
                    elapsed = time.time() - start_time
                    return new_path, elapsed, nodes_expanded
                h = heuristic(next_state, problem.goal)
                counter += 1
                heapq.heappush(heap, (new_g + h, new_g, counter, next_state, new_path))
    
    return None, time.time() - start_time, nodes_expanded

def forward_planner_astar_subgoals(problem, heuristic, subgoals, timeout=300):
    """Forward planner A* z podecelami - rozwiązuje sekwencyjnie przez subgoale"""
    start_time = time.time()
    current_state = problem.initial_state
    full_plan = []
    nodes_total = 0
    goals_sequence = list(subgoals) + [problem.goal]
    for subgoal in goals_sequence:
        sub_problem = ProblemSTRIPS(current_state, subgoal, problem.actions)
        result = forward_planner_astar(sub_problem, heuristic,
                                        timeout=(timeout - (time.time()-start_time)))
        if result[0] is None:
            return None, time.time()-start_time, nodes_total
        plan, _, nodes = result
        full_plan += plan
        nodes_total += nodes
        for action in plan:
            current_state = (current_state - action.del_effects) | action.add_effects
    elapsed = time.time() - start_time
    return full_plan, elapsed, nodes_total



# =============================================================================
# HEURYSTYKI OGÓLNA
# =============================================================================

def heuristic_unsatisfied_goals(state, goal):
    """Liczy ile warunków celu nie jest spełnionych"""
    return len(goal - state)

# =============================================================================
# DOMENA 1: AIR CARGO
# =============================================================================
# Stan reprezentowany jako zbiór krotek: 
#   ('At', obj, place), ('In', obj, plane), ('Cargo', obj), ('Plane', obj), ('Airport', obj)
# W naszej implementacji używamy Feature=Value par
# Kodowanie: ('At_C1', 'SFO'), ('In_C1', 'P1'), ('IsCargo_C1', True) itp.
# Dla uproszczenia: użyjemy prostej reprezentacji logicznej
# Każdy predykat to string + argumenty jako klucz

def make_air_cargo_domain():
    """Tworzy domenę Air Cargo"""
    # Akcje będziemy generować dynamicznie na podstawie obiektów
    pass

def build_air_cargo_problem(cargos, planes, airports, at_cargo, at_planes, goals_cargo):
    """
    Buduje problem Air Cargo.
    cargos: lista ładunków
    planes: lista samolotów  
    airports: lista lotnisk
    at_cargo: dict {cargo: airport}
    at_planes: dict {plane: airport}
    goals_cargo: dict {cargo: airport} - gdzie cargo ma być
    """
    # Stan początkowy
    initial = set()
    for c in cargos:
        initial.add(('Cargo', c, True))
        initial.add(('At', c, at_cargo[c]))
    for p in planes:
        initial.add(('Plane', p, True))
        initial.add(('At', p, at_planes[p]))
    for a in airports:
        initial.add(('Airport', a, True))
    
    # Akcje
    actions = []
    
    # LOAD: załaduj cargo c na samolot p na lotnisku a
    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"LOAD({c},{p},{a})"
                preconds = frozenset([
                    ('At', c, a), ('At', p, a),
                    ('Cargo', c, True), ('Plane', p, True), ('Airport', a, True)
                ])
                effects = {('At', c): None, ('In', c): p}  # usuń At(c,a), dodaj In(c,p)
                actions.append(AirCargoAction(name, preconds, 
                                              add_effects=frozenset([('In', c, p)]),
                                              del_effects=frozenset([('At', c, a)])))
    
    # UNLOAD
    for c in cargos:
        for p in planes:
            for a in airports:
                name = f"UNLOAD({c},{p},{a})"
                preconds = frozenset([
                    ('In', c, p), ('At', p, a),
                    ('Cargo', c, True), ('Plane', p, True), ('Airport', a, True)
                ])
                actions.append(AirCargoAction(name, preconds,
                                              add_effects=frozenset([('At', c, a)]),
                                              del_effects=frozenset([('In', c, p)])))
    
    # FLY
    for p in planes:
        for f in airports:
            for t in airports:
                if f != t:
                    name = f"FLY({p},{f},{t})"
                    preconds = frozenset([
                        ('At', p, f),
                        ('Plane', p, True), ('Airport', f, True), ('Airport', t, True)
                    ])
                    actions.append(AirCargoAction(name, preconds,
                                                  add_effects=frozenset([('At', p, t)]),
                                                  del_effects=frozenset([('At', p, f)])))
    
    goal = frozenset([('At', c, a) for c, a in goals_cargo.items()])
    
    return ProblemSTRIPS(initial, goal, actions)


class AirCargoAction:
    """Akcja dla Air Cargo z add/del lists"""
    def __init__(self, name, preconds, add_effects, del_effects):
        self.name = name
        self.preconds = preconds
        self.add_effects = add_effects
        self.del_effects = del_effects
    
    def __repr__(self):
        return self.name

class ProblemSTRIPS:
    """Problem STRIPS z add/delete lists"""
    def __init__(self, initial_state, goal, actions):
        self.initial_state = frozenset(initial_state)
        self.goal = frozenset(goal)
        self.actions = actions
    
    def is_goal(self, state):
        return self.goal.issubset(state)
    
    def get_successors(self, state):
        successors = []
        for action in self.actions:
            if action.preconds.issubset(state):
                new_state = (state - action.del_effects) | action.add_effects
                successors.append((action, new_state))
        return successors


# =============================================================================
# DOMENA 2: DOCK WORKER ROBOT
# =============================================================================

def build_dwr_problem(robots, locations, cranes, piles, containers,
                      adjacent_pairs, attached_pile_loc, belong_crane_loc,
                      initial_stacks, robot_at, robot_loaded, crane_holding,
                      free_locations, goal_stacks):
    """
    Buduje problem Dock Worker Robot.
    initial_stacks: {pile: [bottom, ..., top]} - lista kontenerów od dołu
    goal_stacks: {pile: [bottom, ..., top]}
    """
    initial = set()
    
    # Typy
    for r in robots: initial.add(('robot', r, True))
    for l in locations: initial.add(('location', l, True))
    for k in cranes: initial.add(('crane', k, True))
    for p in piles: initial.add(('pile', p, True))
    for c in containers: initial.add(('container', c, True))
    
    # Sąsiedztwo
    for (l1, l2) in adjacent_pairs:
        initial.add(('adjacent', l1, l2))
    
    # Przypisanie pile do lokacji
    for p, l in attached_pile_loc.items():
        initial.add(('attached', p, l))
    
    # Przypisanie crane do lokacji
    for k, l in belong_crane_loc.items():
        initial.add(('belong', k, l))
    
    # Stosy kontenerów: in, on, top
    for pile, stack in initial_stacks.items():
        if not stack:
            initial.add(('top', 'pallet', pile))
        else:
            # pallet jest na dnie
            initial.add(('on', stack[0], 'pallet'))
            initial.add(('in', stack[0], pile))
            for i in range(1, len(stack)):
                initial.add(('on', stack[i], stack[i-1]))
                initial.add(('in', stack[i], pile))
            initial.add(('top', stack[-1], pile))
    
    # Roboty
    for r, l in robot_at.items():
        initial.add(('at', r, l))
    for r, loaded in robot_loaded.items():
        if loaded:
            initial.add(('loaded', r, loaded))
        else:
            initial.add(('unloaded', r))
    
    # Crane
    for k, holding in crane_holding.items():
        if holding:
            initial.add(('holding', k, holding))
        else:
            initial.add(('empty', k))
    
    # Wolne lokacje
    for l in free_locations:
        initial.add(('free', l))
    
    # Akcje
    actions = []
    
    # MOVE: robot r z from do to
    for r in robots:
        for l1 in locations:
            for l2 in locations:
                if l1 != l2:
                    name = f"move({r},{l1},{l2})"
                    preconds = frozenset([
                        ('adjacent', l1, l2), ('at', r, l1), ('free', l2),
                        ('robot', r, True), ('location', l1, True), ('location', l2, True)
                    ])
                    add_eff = frozenset([('at', r, l2), ('free', l1)])
                    del_eff = frozenset([('at', r, l1), ('free', l2)])
                    actions.append(AirCargoAction(name, preconds, add_eff, del_eff))
    
    # LOAD: crane k at l loads container c onto robot r
    for k in cranes:
        lk = belong_crane_loc[k]
        for r in robots:
            for c in containers:
                name = f"load({k},{lk},{c},{r})"
                preconds = frozenset([
                    ('at', r, lk), ('belong', k, lk),
                    ('holding', k, c), ('unloaded', r),
                    ('crane', k, True), ('robot', r, True), ('container', c, True)
                ])
                add_eff = frozenset([('loaded', r, c), ('empty', k)])
                del_eff = frozenset([('unloaded', r), ('holding', k, c)])
                actions.append(AirCargoAction(name, preconds, add_eff, del_eff))
    
    # UNLOAD: crane k at l unloads container c from robot r
    for k in cranes:
        lk = belong_crane_loc[k]
        for r in robots:
            for c in containers:
                name = f"unload({k},{lk},{c},{r})"
                preconds = frozenset([
                    ('belong', k, lk), ('at', r, lk),
                    ('loaded', r, c), ('empty', k),
                    ('crane', k, True), ('robot', r, True), ('container', c, True)
                ])
                add_eff = frozenset([('unloaded', r), ('holding', k, c)])
                del_eff = frozenset([('loaded', r, c), ('empty', k)])
                actions.append(AirCargoAction(name, preconds, add_eff, del_eff))
    
    # TAKE: crane k at l bierze kontener c (który leży na else) ze stosu p
    for k in cranes:
        lk = belong_crane_loc[k]
        # Pobieramy tylko pile dla tej lokacji
        local_piles = [p for p, l in attached_pile_loc.items() if l == lk]
        for p in local_piles:
            for c in containers + ['pallet']:
                for c_below in containers + ['pallet']:
                    if c != c_below:
                        name = f"take({k},{lk},{c},{c_below},{p})"
                        preconds = frozenset([
                            ('belong', k, lk), ('attached', p, lk),
                            ('empty', k), ('in', c, p),
                            ('top', c, p), ('on', c, c_below),
                            ('crane', k, True), ('container', c, True) if c != 'pallet' else ('container', 'pallet', True)
                        ])
                        add_eff = frozenset([('holding', k, c), ('top', c_below, p)])
                        del_eff = frozenset([('in', c, p), ('top', c, p), ('on', c, c_below), ('empty', k)])
                        actions.append(AirCargoAction(name, preconds, add_eff, del_eff))
    
    # PUT: crane k at l kładzie kontener c (na else) na stosie p
    for k in cranes:
        lk = belong_crane_loc[k]
        local_piles = [p for p, l in attached_pile_loc.items() if l == lk]
        for p in local_piles:
            for c in containers:
                for c_top in containers + ['pallet']:
                    if c != c_top:
                        name = f"put({k},{lk},{c},{c_top},{p})"
                        preconds = frozenset([
                            ('belong', k, lk), ('attached', p, lk),
                            ('holding', k, c), ('top', c_top, p),
                            ('crane', k, True), ('container', c, True)
                        ])
                        add_eff = frozenset([('in', c, p), ('top', c, p), ('on', c, c_top), ('empty', k)])
                        del_eff = frozenset([('top', c_top, p), ('holding', k, c)])
                        actions.append(AirCargoAction(name, preconds, add_eff, del_eff))
    
    # Cel
    goal_set = set()
    for pile, stack in goal_stacks.items():
        for c in stack:
            goal_set.add(('in', c, pile))
    
    return ProblemSTRIPS(initial, frozenset(goal_set), actions)


# =============================================================================
# DOMENA 3: 1D RUBIK'S CUBE
# =============================================================================

def build_rubik1d_problem(initial_positions, goal_positions=None):
    """
    Buduje problem 1D Rubik's Cube.
    initial_positions: lista 6 elementów, np. [1,3,2,6,5,4]
    goal_positions: lista 6 elementów (domyślnie [1,2,3,4,5,6])
    """
    if goal_positions is None:
        goal_positions = [1, 2, 3, 4, 5, 6]
    
    vals = list(range(1, 7))  # v1..v6
    
    def make_state(pos):
        return frozenset([('pos', i+1, pos[i]) for i in range(6)])
    
    initial = make_state(initial_positions)
    goal = make_state(goal_positions)
    
    # Akcje rot0, rot1, rot2
    # rot0: odwróć pozycje 1-4, tzn. [p1,p2,p3,p4] -> [p4,p3,p2,p1]
    # rot1: odwróć pozycje 2-5
    # rot2: odwróć pozycje 3-6
    
    actions = []
    
    for v1 in vals:
        for v2 in vals:
            for v3 in vals:
                for v4 in vals:
                    for v5 in vals:
                        for v6 in vals:
                            if len({v1,v2,v3,v4,v5,v6}) != 6:
                                continue
                            preconds = frozenset([
                                ('pos',1,v1),('pos',2,v2),('pos',3,v3),
                                ('pos',4,v4),('pos',5,v5),('pos',6,v6)
                            ])
                            
                            # rot0: odwróć 1-4
                            add0 = frozenset([('pos',1,v4),('pos',2,v3),('pos',3,v2),('pos',4,v1),('pos',5,v5),('pos',6,v6)])
                            del0 = frozenset([('pos',1,v1),('pos',2,v2),('pos',3,v3),('pos',4,v4)])
                            if add0 != preconds:  # unikaj brak-zmian
                                actions.append(AirCargoAction(f"rot0({v1},{v2},{v3},{v4},{v5},{v6})",
                                                               preconds, add0, del0))
                            
                            # rot1: odwróć 2-5
                            add1 = frozenset([('pos',1,v1),('pos',2,v5),('pos',3,v4),('pos',4,v3),('pos',5,v2),('pos',6,v6)])
                            del1 = frozenset([('pos',2,v2),('pos',3,v3),('pos',4,v4),('pos',5,v5)])
                            if add1 != preconds:
                                actions.append(AirCargoAction(f"rot1({v1},{v2},{v3},{v4},{v5},{v6})",
                                                               preconds, add1, del1))
                            
                            # rot2: odwróć 3-6
                            add2 = frozenset([('pos',1,v1),('pos',2,v2),('pos',3,v6),('pos',4,v5),('pos',5,v4),('pos',6,v3)])
                            del2 = frozenset([('pos',3,v3),('pos',4,v4),('pos',5,v5),('pos',6,v6)])
                            if add2 != preconds:
                                actions.append(AirCargoAction(f"rot2({v1},{v2},{v3},{v4},{v5},{v6})",
                                                               preconds, add2, del2))
    
    return ProblemSTRIPS(initial, goal, actions)


# =============================================================================
# HEURYSTYKI SPECYFICZNE
# =============================================================================

def heuristic_air_cargo(state, goal):
    """
    Heurystyka dla Air Cargo:
    Liczy ładunki, które jeszcze nie są na docelowym lotnisku.
    Każdy taki ładunek wymaga min. 1 akcji (UNLOAD lub LOAD+FLY+UNLOAD).
    Heurystyka jest dopuszczalna (nie przeszacowuje).
    """
    unsatisfied = len(goal - state)
    return unsatisfied

def heuristic_dwr(state, goal):
    """
    Heurystyka dla DWR:
    Liczy kontenery, które nie są jeszcze w docelowym stosie.
    Każdy taki kontener wymaga co najmniej 1 akcji.
    """
    return len(goal - state)

def heuristic_rubik(state, goal):
    """
    Heurystyka dla 1D Rubik:
    Liczy ile pozycji ma złą wartość.
    Każda rotacja zmienia 4 pozycje, więc dolna granica to ceil(bad_positions/4).
    """
    bad = sum(1 for (k, pos, v) in goal if ('pos', pos, v) not in state)
    return (bad + 3) // 4  # ceil(bad/4)


# =============================================================================
# RUNNER - uruchamia solver i zwraca wyniki
# =============================================================================

def solve_and_report(name, problem, heuristic=None, subgoals=None, timeout=300):
    """Rozwiązuje problem i zwraca dict z wynikami"""
    print(f"\n{'='*60}")
    print(f"PROBLEM: {name}")
    print(f"{'='*60}")
    
    results = {'name': name}
    
    # --- BFS (bez heurystyki) ---
    if subgoals is None:
        print(f"\n[BFS - bez heurystyki]")
        t0 = time.time()
        res = forward_planner_bfs(problem, timeout=timeout)
        if res[0] is not None:
            plan, elapsed, nodes = res
            print(f"  Rozwiązanie znalezione!")
            print(f"  Liczba akcji: {len(plan)}")
            print(f"  Czas: {elapsed:.4f}s")
            print(f"  Węzły rozwinięte: {nodes}")
            print(f"  Plan: {[str(a) for a in plan]}")
            results['bfs_plan'] = [str(a) for a in plan]
            results['bfs_time'] = elapsed
            results['bfs_nodes'] = nodes
        else:
            print(f"  TIMEOUT lub brak rozwiązania")
            results['bfs_plan'] = None
            results['bfs_time'] = timeout
            results['bfs_nodes'] = res[2]
    
    # --- A* z heurystyką ---
    if heuristic:
        print(f"\n[A* - z heurystyką]")
        if subgoals:
            print(f"  Podecele: {[str(sorted(list(sg))) for sg in subgoals]}")
            plan, elapsed, nodes = forward_planner_astar_subgoals(
                problem, heuristic, subgoals, timeout=timeout)
        else:
            plan, elapsed, nodes = forward_planner_astar(problem, heuristic, timeout=timeout)
        
        if plan is not None:
            print(f"  Rozwiązanie znalezione!")
            print(f"  Liczba akcji: {len(plan)}")
            print(f"  Czas: {elapsed:.4f}s")
            print(f"  Węzły rozwinięte: {nodes}")
            print(f"  Plan: {[str(a) for a in plan]}")
            results['astar_plan'] = [str(a) for a in plan]
            results['astar_time'] = elapsed
            results['astar_nodes'] = nodes
        else:
            print(f"  TIMEOUT lub brak rozwiązania")
            results['astar_plan'] = None
            results['astar_time'] = elapsed
    
    return results


# =============================================================================
# DEFINICJA PROBLEMÓW
# =============================================================================

def get_all_problems():
    problems = []
    
    # =========================================================================
    # DOMENA 1: AIR CARGO
    # =========================================================================
    # Problem AC1: Oryginalny (z zadania) - C1@SFO->JFK, C2@JFK->SFO
    # 2 cargo, 2 planes, 2 airports
    # Rozwiązanie: LOAD(C1,P1,SFO), LOAD(C2,P2,JFK), FLY(P1,SFO,JFK), FLY(P2,JFK,SFO), UNLOAD(C1,P1,JFK), UNLOAD(C2,P2,SFO)
    # = 6 akcji
    
    ac1 = build_air_cargo_problem(
        cargos=['C1','C2'], planes=['P1','P2'], airports=['SFO','JFK'],
        at_cargo={'C1':'SFO','C2':'JFK'}, at_planes={'P1':'SFO','P2':'JFK'},
        goals_cargo={'C1':'JFK','C2':'SFO'}
    )
    problems.append(('AC1_Bazowy', ac1, heuristic_air_cargo))
    
    # Problem AC2: 3 cargo, 2 planes, 3 airports
    # C1@SFO->JFK, C2@JFK->LAX, C3@LAX->SFO
    ac2 = build_air_cargo_problem(
        cargos=['C1','C2','C3'], planes=['P1','P2'],
        airports=['SFO','JFK','LAX'],
        at_cargo={'C1':'SFO','C2':'JFK','C3':'LAX'},
        at_planes={'P1':'SFO','P2':'JFK'},
        goals_cargo={'C1':'JFK','C2':'LAX','C3':'SFO'}
    )
    problems.append(('AC2_Rozszerzony', ac2, heuristic_air_cargo))
    
    # Problem AC3: 4 cargo, 2 planes, 2 airports (większa przestrzeń stanów, min. 8 akcji)
    # C1,C2,C3 z SFO do JFK; C4 z JFK do SFO
    ac3 = build_air_cargo_problem(
        cargos=['C1','C2','C3','C4'], planes=['P1','P2'],
        airports=['SFO','JFK'],
        at_cargo={'C1':'SFO','C2':'SFO','C3':'SFO','C4':'JFK'},
        at_planes={'P1':'SFO','P2':'JFK'},
        goals_cargo={'C1':'JFK','C2':'JFK','C3':'JFK','C4':'SFO'}
    )
    problems.append(('AC3_DuzyProb', ac3, heuristic_air_cargo))
    
    # =========================================================================
    # DOMENA 2: DOCK WORKER ROBOT
    # =========================================================================
    # Problem DWR1: Oryginalny z zadania
    # ca,cb,cc w p1 (od dołu: pallet,ca,cb,cc)
    # cd,ce,cf w q1 (od dołu: pallet,cd,ce,cf)
    # Cel: ca,cc w p2; cb,cd,ce,cf w q2
    
    dwr1 = build_dwr_problem(
        robots=['r1'],
        locations=['l1','l2'],
        cranes=['k1','k2'],
        piles=['p1','q1','p2','q2'],
        containers=['ca','cb','cc','cd','ce','cf'],
        adjacent_pairs=[('l1','l2'),('l2','l1')],
        attached_pile_loc={'p1':'l1','q1':'l1','p2':'l2','q2':'l2'},
        belong_crane_loc={'k1':'l1','k2':'l2'},
        initial_stacks={
            'p1': ['ca','cb','cc'],
            'q1': ['cd','ce','cf'],
            'p2': [],
            'q2': []
        },
        robot_at={'r1':'l1'},
        robot_loaded={'r1': None},
        crane_holding={'k1': None, 'k2': None},
        free_locations=['l2'],
        goal_stacks={
            'p2': ['ca','cc'],
            'q2': ['cb','cd','ce','cf']
        }
    )
    problems.append(('DWR1_Bazowy', dwr1, heuristic_dwr))
    
    # Problem DWR2: Prostszy - 4 kontenery, przenie wszystko z l1 do l2
    dwr2 = build_dwr_problem(
        robots=['r1'],
        locations=['l1','l2'],
        cranes=['k1','k2'],
        piles=['p1','q1','p2','q2'],
        containers=['ca','cb','cc','cd'],
        adjacent_pairs=[('l1','l2'),('l2','l1')],
        attached_pile_loc={'p1':'l1','q1':'l1','p2':'l2','q2':'l2'},
        belong_crane_loc={'k1':'l1','k2':'l2'},
        initial_stacks={
            'p1': ['ca','cb'],
            'q1': ['cc','cd'],
            'p2': [],
            'q2': []
        },
        robot_at={'r1':'l1'},
        robot_loaded={'r1': None},
        crane_holding={'k1': None, 'k2': None},
        free_locations=['l2'],
        goal_stacks={
            'p2': ['ca','cb'],
            'q2': ['cc','cd']
        }
    )
    problems.append(('DWR2_Sredni', dwr2, heuristic_dwr))
    
    # Problem DWR3: Mały - 2 kontenery, prostszy
    dwr3 = build_dwr_problem(
        robots=['r1'],
        locations=['l1','l2'],
        cranes=['k1','k2'],
        piles=['p1','p2'],
        containers=['ca','cb'],
        adjacent_pairs=[('l1','l2'),('l2','l1')],
        attached_pile_loc={'p1':'l1','p2':'l2'},
        belong_crane_loc={'k1':'l1','k2':'l2'},
        initial_stacks={
            'p1': ['ca','cb'],
            'p2': []
        },
        robot_at={'r1':'l1'},
        robot_loaded={'r1': None},
        crane_holding={'k1': None, 'k2': None},
        free_locations=['l2'],
        goal_stacks={
            'p2': ['ca','cb']
        }
    )
    problems.append(('DWR3_Maly', dwr3, heuristic_dwr))
    
    # =========================================================================
    # DOMENA 3: 1D RUBIK'S CUBE
    # =========================================================================
    # Problem RUB1: 1 3 2 6 5 4 -> 1 2 3 4 5 6 (z zadania, rozwiązanie: rot1,rot2,rot1)
    rub1 = build_rubik1d_problem([1,3,2,6,5,4])
    problems.append(('RUB1_Bazowy', rub1, heuristic_rubik))
    
    # Problem RUB2: 5 6 2 1 4 3 -> 1 2 3 4 5 6 (z zadania, rozwiązanie: rot0, rot2)
    rub2 = build_rubik1d_problem([5,6,2,1,4,3])
    problems.append(('RUB2_Sredni', rub2, heuristic_rubik))
    
    # Problem RUB3: 6 5 4 1 2 3 -> 1 2 3 4 5 6 (z zadania, rozwiązanie: rot0, rot1, rot2)
    rub3 = build_rubik1d_problem([6,5,4,1,2,3])
    problems.append(('RUB3_Trudny', rub3, heuristic_rubik))
    
    return problems


# =============================================================================
# PODECELE (SUBGOALS) - zadanie na 6 punktów
# =============================================================================

def get_subgoals():
    """
    Definiuje podecele dla każdej domeny.
    Podecele = pośrednie stany, które trzeba osiągnąć na drodze do celu.
    """
    subgoals = {}
    
    # AIR CARGO AC1: 
    # Podcel 1: załaduj oba samoloty (cargo jest IN samolotach)
    # Podcel 2: przesuń samoloty (pełne rozwiązanie)
    subgoals['AC1'] = [
        frozenset([('In','C1','P1'), ('In','C2','P2')]),    # oba załadowane
        frozenset([('At','P1','JFK'), ('At','P2','SFO')]),   # oba samoloty na miejscu
    ]
    
    # AIR CARGO AC2:
    # Podcel 1: C1 załadowany na P1 w SFO
    # Podcel 2: P1 przyleciał do JFK, C1 wyładowany
    subgoals['AC2'] = [
        frozenset([('In','C1','P1')]),                        # C1 załadowany
        frozenset([('At','C1','JFK')]),                       # C1 na miejscu
    ]
    
    # AIR CARGO AC3:
    # Podcel 1: wszystkie cargo załadowane
    # Podcel 2: samoloty na właściwych lotniskach
    subgoals['AC3'] = [
        frozenset([('In','C1','P1'), ('In','C2','P1')]),      # P1 załadowany
        frozenset([('At','C1','JFK'), ('At','C2','JFK')]),    # C1,C2 dostarczone
    ]
    
    # DWR - podecele: 
    # Podcel 1: zdjąć cc z p1 (cc jest top, więc można dostać do cb/ca)
    # Podcel 2: przemieścić przynajmniej ca do l2
    subgoals['DWR3'] = [
        frozenset([('holding', 'k1', 'cb')]),                 # cb w crane
        frozenset([('in', 'ca', 'p2')]),                      # ca dostarczone
    ]
    
    subgoals['DWR2'] = [
        frozenset([('in', 'cb', 'p2')]),                      # cb w p2
        frozenset([('in', 'ca', 'p2'), ('in', 'cb', 'p2')]),  # oba w p2
    ]
    
    # RUBIK - podecele:
    # Podcel 1: popraw pierwsze 3 pozycje
    # Podcel 2: popraw całość
    subgoals['RUB1'] = [
        frozenset([('pos',1,1), ('pos',2,2)]),                # pozycje 1,2 OK
    ]
    
    subgoals['RUB2'] = [
        frozenset([('pos',1,1), ('pos',2,2), ('pos',3,3)]),   # pierwsze 3 OK
    ]
    
    subgoals['RUB3'] = [
        frozenset([('pos',1,1)]),                              # pozycja 1 OK
        frozenset([('pos',1,1), ('pos',2,2)]),                # pozycje 1,2 OK
    ]
    
    return subgoals


# =============================================================================
# PROBLEMY NA 8 PUNKTÓW - min. 20 akcji
# =============================================================================

def get_hard_problems():
    """
    Trzy problemy z podecelami wymagające minimum 20 akcji.
    """
    hard = []
    
    # HARD1: Air Cargo z 4 cargo, 3 planes, 4 airports
    # Każde cargo musi "okrążyć" - trzeba wykonać wiele lotów
    # Minimalne rozwiązanie > 20 akcji
    ac_hard = build_air_cargo_problem(
        cargos=['C1','C2','C3','C4','C5'],
        planes=['P1','P2','P3'],
        airports=['SFO','JFK','LAX','ORD'],
        at_cargo={'C1':'SFO','C2':'SFO','C3':'JFK','C4':'JFK','C5':'LAX'},
        at_planes={'P1':'SFO','P2':'JFK','P3':'LAX'},
        goals_cargo={'C1':'ORD','C2':'LAX','C3':'SFO','C4':'LAX','C5':'ORD'}
    )
    
    # Podecele: 
    # Podcel 1: wszystkie cargo załadowane lub w tranzycie
    # Podcel 2: C1 i C3 dostarczone  
    subgoals_ac_hard = [
        frozenset([('At','C1','JFK')]),   # C1 w drodze - na JFK
        frozenset([('At','C3','SFO')]),   # C3 dostarczone do SFO
    ]
    
    hard.append(('HARD1_AirCargo5cargo', ac_hard, heuristic_air_cargo, subgoals_ac_hard))
    
    # HARD2: DWR z 5 kontenerami
    dwr_hard = build_dwr_problem(
        robots=['r1'],
        locations=['l1','l2'],
        cranes=['k1','k2'],
        piles=['p1','q1','p2','q2'],
        containers=['ca','cb','cc','cd','ce'],
        adjacent_pairs=[('l1','l2'),('l2','l1')],
        attached_pile_loc={'p1':'l1','q1':'l1','p2':'l2','q2':'l2'},
        belong_crane_loc={'k1':'l1','k2':'l2'},
        initial_stacks={
            'p1': ['ca','cb','cc'],
            'q1': ['cd','ce'],
            'p2': [],
            'q2': []
        },
        robot_at={'r1':'l1'},
        robot_loaded={'r1': None},
        crane_holding={'k1': None, 'k2': None},
        free_locations=['l2'],
        goal_stacks={
            'p2': ['ca','cc'],
            'q2': ['cb','cd','ce']
        }
    )
    
    subgoals_dwr_hard = [
        frozenset([('in','cc','p2')]),           # cc dostarczone
        frozenset([('in','ca','p2'), ('in','cc','p2')]),  # ca,cc dostarczone
    ]
    
    hard.append(('HARD2_DWR5cont', dwr_hard, heuristic_dwr, subgoals_dwr_hard))
    
    # HARD3: 1D Rubik - trudniejsze startowe ułożenie wymagające wielu kroków
    # Start: 3 1 4 2 6 5  - wymagające wielu rotacji
    rub_hard = build_rubik1d_problem([3,1,4,2,6,5])
    
    subgoals_rub_hard = [
        frozenset([('pos',1,1),('pos',2,2)]),
        frozenset([('pos',1,1),('pos',2,2),('pos',3,3),('pos',4,4)]),
    ]
    
    hard.append(('HARD3_Rubik3142x65', rub_hard, heuristic_rubik, subgoals_rub_hard))
    
    return hard


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_results = []
    
    print("\n" + "="*70)
    print("PLANOWANIE STRIPS - ROZWIĄZANIE ZADAŃ NA 4, 6 i 8 PUNKTÓW")
    print("="*70)
    
    subgoals_dict = get_subgoals()
    
    # =========================================================================
    # ZADANIE NA 4 PUNKTY: 3 problemy x 3 domeny
    # =========================================================================
    print("\n" + "#"*70)
    print("# ZADANIE NA 4 PUNKTY: Problemy bazowe - BFS i A* z heurystyką")
    print("#"*70)
    
    problems = get_all_problems()
    
    for name, problem, heuristic in problems:
        r = {}
        r['name'] = name
        
        print(f"\n{'='*60}")
        print(f"PROBLEM: {name}")
        print(f"Liczba akcji w domenie: {len(problem.actions)}")
        print(f"Stan początkowy: {len(problem.initial_state)} faktów")
        print(f"Cel: {problem.goal}")
        print(f"{'='*60}")
        
        # BFS
        print(f"\n[BFS - bez heurystyki]")
        t0 = time.time()
        res = forward_planner_bfs(problem, timeout=120)
        if res[0] is not None:
            plan, elapsed, nodes = res
            print(f"  ✓ Rozwiązanie znalezione!")
            print(f"  Liczba akcji: {len(plan)}")
            print(f"  Czas: {elapsed:.4f}s")
            print(f"  Węzły rozwinięte: {nodes}")
            print(f"  Plan:")
            for i, a in enumerate(plan, 1):
                print(f"    {i}. {a}")
            r['bfs_plan'] = [str(a) for a in plan]
            r['bfs_time'] = elapsed
            r['bfs_nodes'] = nodes
        else:
            print(f"  ✗ TIMEOUT")
            r['bfs_plan'] = None
            r['bfs_time'] = 120
        
        # A*
        print(f"\n[A* - z heurystyką: {heuristic.__name__}]")
        plan, elapsed, nodes = forward_planner_astar(problem, heuristic, timeout=120)
        if plan is not None:
            print(f"  ✓ Rozwiązanie znalezione!")
            print(f"  Liczba akcji: {len(plan)}")
            print(f"  Czas: {elapsed:.4f}s")
            print(f"  Węzły rozwinięte: {nodes}")
            print(f"  Plan:")
            for i, a in enumerate(plan, 1):
                print(f"    {i}. {a}")
            r['astar_plan'] = [str(a) for a in plan]
            r['astar_time'] = elapsed
            r['astar_nodes'] = nodes
        else:
            print(f"  ✗ TIMEOUT")
            r['astar_plan'] = None
            r['astar_time'] = elapsed
        
        all_results.append(r)
    
    # =========================================================================
    # ZADANIE NA 6 PUNKTÓW: Podecele
    # =========================================================================
    print("\n" + "#"*70)
    print("# ZADANIE NA 6 PUNKTÓW: Podecele + A* z heurystyką")
    print("#"*70)
    
    subgoal_configs = [
        ('AC1_Bazowy_subgoals', problems[0][1], problems[0][2], subgoals_dict['AC1']),
        ('AC2_Rozszerzony_subgoals', problems[1][1], problems[1][2], subgoals_dict['AC2']),
        ('AC3_DuzyProb_subgoals', problems[2][1], problems[2][2], subgoals_dict['AC3']),
        ('DWR3_Maly_subgoals', problems[5][1], problems[5][2], subgoals_dict['DWR3']),
        ('DWR2_Sredni_subgoals', problems[4][1], problems[4][2], subgoals_dict['DWR2']),
        ('RUB1_Bazowy_subgoals', problems[6][1], problems[6][2], subgoals_dict['RUB1']),
        ('RUB2_Sredni_subgoals', problems[7][1], problems[7][2], subgoals_dict['RUB2']),
        ('RUB3_Trudny_subgoals', problems[8][1], problems[8][2], subgoals_dict['RUB3']),
    ]
    
    for name, problem, heuristic, sgoals in subgoal_configs:
        print(f"\n{'='*60}")
        print(f"PROBLEM Z PODECELAMI: {name}")
        print(f"Podecele ({len(sgoals)}):")
        for i, sg in enumerate(sgoals, 1):
            print(f"  Podcel {i}: {sorted(list(sg))}")
        
        print(f"\n[A* z podecelami + heurystyką]")
        plan, elapsed, nodes = forward_planner_astar_subgoals(
            problem, heuristic, sgoals, timeout=120)
        
        r = {'name': name}
        if plan is not None:
            print(f"  ✓ Rozwiązanie znalezione!")
            print(f"  Liczba akcji: {len(plan)}")
            print(f"  Czas: {elapsed:.4f}s")
            print(f"  Węzły rozwinięte: {nodes}")
            print(f"  Plan:")
            for i, a in enumerate(plan, 1):
                print(f"    {i}. {a}")
            r['plan'] = [str(a) for a in plan]
            r['time'] = elapsed
            r['nodes'] = nodes
        else:
            print(f"  ✗ TIMEOUT lub brak rozwiązania")
            r['plan'] = None
        
        all_results.append(r)
    
    # =========================================================================
    # ZADANIE NA 8 PUNKTÓW: Trudne problemy min. 20 akcji
    # =========================================================================
    print("\n" + "#"*70)
    print("# ZADANIE NA 8 PUNKTÓW: Trudne problemy (min. 20 akcji) z podecelami")
    print("#"*70)
    
    hard_problems = get_hard_problems()
    
    for name, problem, heuristic, sgoals in hard_problems:
        print(f"\n{'='*60}")
        print(f"TRUDNY PROBLEM: {name}")
        print(f"Liczba akcji w domenie: {len(problem.actions)}")
        print(f"Podecele ({len(sgoals)}):")
        for i, sg in enumerate(sgoals, 1):
            print(f"  Podcel {i}: {sorted(list(sg))}")
        
        # A* bez podeceli
        print(f"\n[A* - bez podeceli]")
        plan, elapsed, nodes = forward_planner_astar(problem, heuristic, timeout=120)
        r = {'name': name}
        if plan is not None:
            print(f"  ✓ Rozwiązanie: {len(plan)} akcji, {elapsed:.4f}s, {nodes} węzłów")
            for i, a in enumerate(plan, 1):
                print(f"    {i}. {a}")
            r['astar_plan'] = [str(a) for a in plan]
            r['astar_time'] = elapsed
            r['astar_nodes'] = nodes
        else:
            print(f"  ✗ TIMEOUT")
            r['astar_plan'] = None
        
        # A* z podecelami
        print(f"\n[A* z podecelami]")
        plan, elapsed, nodes = forward_planner_astar_subgoals(
            problem, heuristic, sgoals, timeout=120)
        if plan is not None:
            print(f"  ✓ Rozwiązanie: {len(plan)} akcji, {elapsed:.4f}s, {nodes} węzłów")
            for i, a in enumerate(plan, 1):
                print(f"    {i}. {a}")
            r['subgoal_plan'] = [str(a) for a in plan]
            r['subgoal_time'] = elapsed
            r['subgoal_nodes'] = nodes
        else:
            print(f"  ✗ TIMEOUT")
            r['subgoal_plan'] = None
        
        all_results.append(r)
    
    # =========================================================================
    # ZAPIS WYNIKÓW DO PLIKU
    # =========================================================================
    write_report(all_results, subgoal_configs, hard_problems)
    print("\n\n✓ Raport zapisany do: wyniki_planowania.txt")
    
    return all_results


def write_report(results, subgoal_configs, hard_problems):
    """Zapisuje szczegółowy raport do pliku tekstowego"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("RAPORT: PLANOWANIE STRIPS")
    lines.append("Domeny: Air Cargo, Dock Worker Robot, 1D Rubik's Cube")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("OPIS HEURYSTYK")
    lines.append("-" * 40)
    lines.append("""
1. heuristic_air_cargo (dla Air Cargo):
   Liczy liczbę warunków celu (At cargo airport), które nie są jeszcze spełnione.
   Każdy niezaspokojony cel oznacza co najmniej 1 akcję (UNLOAD), więc heurystyka
   jest DOPUSZCZALNA (nie przeszacowuje rzeczywistego kosztu).
   Dlaczego pomocna: skupia się bezpośrednio na ostatecznym celu - dostarczeniu
   ładunków. Nawet jeśli cargo jest w samolocie lecącym właśnie we właściwe miejsce,
   heurystyka wciąż pokazuje 1 - bo UNLOAD jest jeszcze potrzebny.

2. heuristic_dwr (dla Dock Worker Robot):
   Liczy ile faktów 'in(container, pile)' z celu nie jest jeszcze spełnionych.
   Dopuszczalna, bo każdy brakujący kontener w stosie wymaga co najmniej 1 akcji put.
   Dlaczego pomocna: kieruje planer na "brakujące" kontenery, ignorując pośrednie
   stany (trzymanie przez crane, przewóz robotem), co znacznie redukuje przestrzeń.

3. heuristic_rubik (dla 1D Rubik):
   Liczy pozycje z niewłaściwą wartością i dzieli przez 4 (każda rotacja zmienia
   dokładnie 4 pozycje). Daje dolną granicę liczby ruchów - jest DOPUSZCZALNA.
   Dlaczego pomocna: w problemie Rubika kluczowe jest minimalizowanie ruchów,
   a ta heurystyka daje bezpośrednią ocenę "jak daleko jesteśmy od celu".
""")
    
    lines.append("")
    lines.append("OPIS PODECELI")
    lines.append("-" * 40)
    lines.append("""
Podecele dekomponują problem na mniejsze podproblemy, które planer rozwiązuje
sekwencyjnie. Każdy podcel musi być osiągnięty przed przejściem do następnego.

Air Cargo AC1:
  Podcel 1: Oba ładunki załadowane na samoloty (In C1 P1) AND (In C2 P2)
            -> zmusza do załadowania cargo przed lotem
  Podcel 2: Samoloty na właściwych lotniskach (At P1 JFK) AND (At P2 SFO)
            -> zmusza do wykonania lotów przed rozładunkiem

Dock Worker Robot:
  Podcel 1: Górny kontener wzięty przez crane (zdjęcie z wierzchu stosu)
  Podcel 2: Pierwszy kontener dostarczony do celu
  Uzasadnienie: Naturalny porządek - najpierw zdjąć blokujące, potem przewieźć.

1D Rubik's Cube:
  Podcel 1: Pierwsze 1-2 pozycje prawidłowe
  Podcel 2: Pierwsze 3-4 pozycje prawidłowe (jeśli 2 podecele)
  Uzasadnienie: Podobnie jak w prawdziwym Rubiku - poprawiamy warstwami.
""")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("WYNIKI ROZWIĄZAŃ")
    lines.append("=" * 80)
    
    for r in results:
        lines.append(f"\nPROBLEM: {r.get('name','?')}")
        lines.append("-" * 60)
        
        if 'bfs_plan' in r:
            if r['bfs_plan']:
                lines.append(f"  BFS: {len(r['bfs_plan'])} akcji w {r.get('bfs_time',0):.4f}s "
                             f"({r.get('bfs_nodes',0)} węzłów)")
                lines.append(f"  Plan BFS: {r['bfs_plan']}")
            else:
                lines.append(f"  BFS: TIMEOUT")
        
        if 'astar_plan' in r:
            if r['astar_plan']:
                lines.append(f"  A*:  {len(r['astar_plan'])} akcji w {r.get('astar_time',0):.4f}s "
                             f"({r.get('astar_nodes',0)} węzłów)")
                lines.append(f"  Plan A*: {r['astar_plan']}")
            else:
                lines.append(f"  A*: TIMEOUT")
        
        if 'plan' in r:
            if r['plan']:
                lines.append(f"  A*+Subgoals: {len(r['plan'])} akcji w {r.get('time',0):.4f}s "
                             f"({r.get('nodes',0)} węzłów)")
                lines.append(f"  Plan: {r['plan']}")
            else:
                lines.append(f"  A*+Subgoals: TIMEOUT")
        
        if 'subgoal_plan' in r:
            if r['subgoal_plan']:
                lines.append(f"  A*+Subgoals: {len(r['subgoal_plan'])} akcji w {r.get('subgoal_time',0):.4f}s")
            else:
                lines.append(f"  A*+Subgoals: TIMEOUT")
    
    lines.append("\n" + "=" * 80)
    lines.append("PODSUMOWANIE PORÓWNAWCZE: BFS vs A*")
    lines.append("=" * 80)
    lines.append("""
BFS (Breadth-First Search):
  + Gwarantuje optymalne (najkrótsze) rozwiązanie
  - Wykładnicza złożoność pamięciowa i czasowa
  - Dla dużych problemów (DWR z 6 kontenerami) może nie znaleźć w czasie

A* z heurystyką:
  + Znacznie szybszy dzięki kierowaniu poszukiwań
  + Z dopuszczalną heurystyką wciąż gwarantuje optymalność
  - Wymaga dobrej heurystyki

A* z podecelami:
  + Najszybszy - rozkłada problem na mniejsze podproblemy
  - Może nie znaleźć globalnego optimum (plan może być dłuższy)
  - Podecele muszą być osiągalne i sensowne
""")
    
    with open(r'C:\Users\pryce\Downloads\inteligencjaObliczeniowa\wyniki_planowania.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()