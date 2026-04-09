"""
MAGICWORLD (Magical Spells and Potions)
Domain: Fantasy world with wizards, spells, potions, and magical transformations
Source: https://github.com/primaryobjects/strips/tree/master/examples/magicworld

Problem: Achieve magical transformation through spells and potions
- Collect ingredients (herbs, crystals, reagents)
- Mix potions with effects
- Cast spells to transform objects
- Break/dispel existing magic
- Create enchanted items
"""
from aipython_strips import Strips, STRIPS_domain, Planning_problem

# ==================================================================================
# MAGICWORLD DOMAIN DEFINITION
# ==================================================================================

def create_magicworld_domain():
    """
    Features:
    - have_<ingredient>: {True, False} - whether we have ingredient
    - have_<potion>: {True, False} - whether we brewed the potion
    - cast_<spell>: {True, False} - whether spell was cast
    - <object>_state: {normal, enchanted, petrified, invisible} - state of object
    - player_level: {apprentice, journeyman, master} - wizard skill level
    - mana: {0..100} - wizard's magical energy
    """

    boolean = {True, False}
    object_states = {'normal', 'enchanted', 'petrified', 'invisible', 'transmuted'}
    skill_levels = {'apprentice', 'journeyman', 'master'}

    feature_domain_dict = {
        # Ingredients
        'have_moonstone': boolean,
        'have_dragon_scale': boolean,
        'have_phoenix_feather': boolean,
        'have_crystal': boolean,
        'have_herbs': boolean,
        'have_unicorn_horn': boolean,

        # Potions (created from ingredients)
        'brewed_invisibility_potion': boolean,
        'brewed_strength_potion': boolean,
        'brewed_transformation_potion': boolean,
        'brewed_healing_potion': boolean,

        # Spells cast
        'cast_fireball': boolean,
        'cast_freeze': boolean,
        'cast_lightning': boolean,
        'cast_transmute': boolean,

        # Object states
        'sword_state': object_states,
        'amulet_state': object_states,
        'ring_state': object_states,

        # Wizard status
        'player_level': skill_levels,
        'mana': {0, 25, 50, 75, 100},

        # Additional flags
        'magic_circle_drawn': boolean,
        'ritual_prepared': boolean,
    }

    actions = set()

    # ========== GATHER INGREDIENTS ==========
    # These actions acquire raw materials (no preconditions)
    actions.add(Strips('gather_moonstone',
                       {},
                       {'have_moonstone': True}))

    actions.add(Strips('gather_dragon_scale',
                       {},
                       {'have_dragon_scale': True}))

    actions.add(Strips('gather_phoenix_feather',
                       {},
                       {'have_phoenix_feather': True}))

    actions.add(Strips('gather_crystal',
                       {},
                       {'have_crystal': True}))

    actions.add(Strips('gather_herbs',
                       {},
                       {'have_herbs': True}))

    actions.add(Strips('gather_unicorn_horn',
                       {},
                       {'have_unicorn_horn': True}))

    # ========== BREW POTIONS ==========
    # Create potions from ingredients

    actions.add(Strips('brew_invisibility_potion',
                       {'have_phoenix_feather': True, 'have_herbs': True},
                       {'brewed_invisibility_potion': True,
                        'have_phoenix_feather': False, 'have_herbs': False}))

    actions.add(Strips('brew_strength_potion',
                       {'have_dragon_scale': True, 'have_crystal': True},
                       {'brewed_strength_potion': True,
                        'have_dragon_scale': False, 'have_crystal': False}))

    actions.add(Strips('brew_transformation_potion',
                       {'have_unicorn_horn': True, 'have_moonstone': True},
                       {'brewed_transformation_potion': True,
                        'have_unicorn_horn': False, 'have_moonstone': False}))

    actions.add(Strips('brew_healing_potion',
                       {'have_herbs': True, 'have_crystal': True},
                       {'brewed_healing_potion': True,
                        'have_herbs': False, 'have_crystal': False}))

    # ========== PREPARATION SPELLS ==========
    # These prepare the environment

    actions.add(Strips('draw_magic_circle',
                       {},
                       {'magic_circle_drawn': True}))

    actions.add(Strips('prepare_ritual',
                       {'magic_circle_drawn': True},
                       {'ritual_prepared': True}))

    # ========== TRAIN TO INCREASE SKILL ==========
    actions.add(Strips('train_to_journeyman',
                       {'player_level': 'apprentice', 'mana': 50},
                       {'player_level': 'journeyman'}))

    actions.add(Strips('train_to_master',
                       {'player_level': 'journeyman', 'mana': 100},
                       {'player_level': 'master'}))

    # ========== RESTORE MANA ==========
    actions.add(Strips('meditate_restore_25',
                       {'mana': 0},
                       {'mana': 25}))

    actions.add(Strips('meditate_restore_50',
                       {'mana': 25},
                       {'mana': 50}))

    actions.add(Strips('meditate_restore_75',
                       {'mana': 50},
                       {'mana': 75}))

    actions.add(Strips('meditate_restore_100',
                       {'mana': 75},
                       {'mana': 100}))

    # ========== CAST SPELLS ==========
    # Spells require mana and have effects

    actions.add(Strips('cast_fireball',
                       {'mana': 50, 'ritual_prepared': True},
                       {'cast_fireball': True, 'mana': 25}))

    actions.add(Strips('cast_freeze',
                       {'mana': 50, 'ritual_prepared': True},
                       {'cast_freeze': True, 'mana': 25}))

    actions.add(Strips('cast_lightning',
                       {'mana': 75, 'player_level': 'master'},
                       {'cast_lightning': True, 'mana': 50}))

    actions.add(Strips('cast_transmute',
                       {'mana': 100, 'player_level': 'master', 'ritual_prepared': True},
                       {'cast_transmute': True, 'mana': 50}))

    # ========== ENCHANT OBJECTS ==========
    # Use potions and spells to enchant objects

    actions.add(Strips('enchant_sword_with_invisibility',
                       {'sword_state': 'normal', 'brewed_invisibility_potion': True, 'cast_freeze': True},
                       {'sword_state': 'invisible', 'brewed_invisibility_potion': False}))

    actions.add(Strips('enchant_sword_with_strength',
                       {'sword_state': 'normal', 'brewed_strength_potion': True, 'cast_fireball': True},
                       {'sword_state': 'enchanted', 'brewed_strength_potion': False}))

    actions.add(Strips('enchant_amulet_with_protection',
                       {'amulet_state': 'normal', 'brewed_healing_potion': True, 'ritual_prepared': True},
                       {'amulet_state': 'enchanted', 'brewed_healing_potion': False}))

    actions.add(Strips('enchant_ring_with_transformation',
                       {'ring_state': 'normal', 'brewed_transformation_potion': True, 'cast_transmute': True},
                       {'ring_state': 'transmuted', 'brewed_transformation_potion': False}))

    # ========== DISPEL MAGIC ==========
    actions.add(Strips('dispel_sword_magic',
                       {'sword_state': 'enchanted'},
                       {'sword_state': 'normal'}))

    actions.add(Strips('remove_invisibility_from_sword',
                       {'sword_state': 'invisible'},
                       {'sword_state': 'normal'}))

    return STRIPS_domain(feature_domain_dict, actions)


def magicworld_problem_1():
    """
    PROBLEM 1 (Basic)
    Goal: Create an invisible sword
    - Gather ingredients
    - Brew invisibility potion
    - Cast freeze spell
    - Enchant sword
    - Requires ~6-8 actions minimum
    """
    domain = create_magicworld_domain()

    initial_state = {
        'have_moonstone': False,
        'have_dragon_scale': False,
        'have_phoenix_feather': False,
        'have_crystal': False,
        'have_herbs': False,
        'have_unicorn_horn': False,
        'brewed_invisibility_potion': False,
        'brewed_strength_potion': False,
        'brewed_transformation_potion': False,
        'brewed_healing_potion': False,
        'cast_fireball': False,
        'cast_freeze': False,
        'cast_lightning': False,
        'cast_transmute': False,
        'sword_state': 'normal',
        'amulet_state': 'normal',
        'ring_state': 'normal',
        'player_level': 'apprentice',
        'mana': 50,
        'magic_circle_drawn': False,
        'ritual_prepared': False,
    }

    goal = {
        'sword_state': 'invisible',
    }

    return Planning_problem(domain, initial_state, goal)


def magicworld_problem_2():
    """
    PROBLEM 2 (Medium)
    Goal: Create enchanted sword and protected amulet
    - Requires multiple potions
    - Multiple spells needed
    - Medium complexity
    - Requires ~10-12 actions minimum
    """
    domain = create_magicworld_domain()

    initial_state = {
        'have_moonstone': False,
        'have_dragon_scale': False,
        'have_phoenix_feather': False,
        'have_crystal': False,
        'have_herbs': False,
        'have_unicorn_horn': False,
        'brewed_invisibility_potion': False,
        'brewed_strength_potion': False,
        'brewed_transformation_potion': False,
        'brewed_healing_potion': False,
        'cast_fireball': False,
        'cast_freeze': False,
        'cast_lightning': False,
        'cast_transmute': False,
        'sword_state': 'normal',
        'amulet_state': 'normal',
        'ring_state': 'normal',
        'player_level': 'apprentice',
        'mana': 50,
        'magic_circle_drawn': False,
        'ritual_prepared': False,
    }

    goal = {
        'sword_state': 'enchanted',
        'amulet_state': 'enchanted',
    }

    return Planning_problem(domain, initial_state, goal)


def magicworld_problem_3():
    """
    PROBLEM 3 (Complex)
    Goal: Complex magical transformation
    - Enchant sword, amulet, and ring
    - Requires reaching master skill level
    - Multiple spell types needed
    - Requires ~16-20 actions minimum
    """
    domain = create_magicworld_domain()

    initial_state = {
        'have_moonstone': False,
        'have_dragon_scale': False,
        'have_phoenix_feather': False,
        'have_crystal': False,
        'have_herbs': False,
        'have_unicorn_horn': False,
        'brewed_invisibility_potion': False,
        'brewed_strength_potion': False,
        'brewed_transformation_potion': False,
        'brewed_healing_potion': False,
        'cast_fireball': False,
        'cast_freeze': False,
        'cast_lightning': False,
        'cast_transmute': False,
        'sword_state': 'normal',
        'amulet_state': 'normal',
        'ring_state': 'normal',
        'player_level': 'apprentice',
        'mana': 0,
        'magic_circle_drawn': False,
        'ritual_prepared': False,
    }

    goal = {
        'sword_state': 'enchanted',
        'amulet_state': 'enchanted',
        'ring_state': 'transmuted',
        'player_level': 'master',
        'mana': 50,
    }

    return Planning_problem(domain, initial_state, goal)


def get_magicworld_subgoals():
    """
    Subgoals for magicworld problems:
    1. Problem 1: (a) Gather ingredients, (b) Brew potion, (c) Cast spell
    2. Problem 2: (a) Gather all ingredients, (b) Brew potions, (c) Cast spells
    3. Problem 3: (a) Restore mana, (b) Train to master, (c) Prepare ritual
    """
    return {
        'magicworld_1': [
            {'have_phoenix_feather': True, 'have_herbs': True},
            {'brewed_invisibility_potion': True, 'cast_freeze': True}
        ],
        'magicworld_2': [
            {'have_phoenix_feather': True, 'have_herbs': True,
             'have_dragon_scale': True, 'have_crystal': True},
            {'brewed_invisibility_potion': True, 'brewed_strength_potion': True},
            {'cast_fireball': True, 'cast_freeze': True},
        ],
        'magicworld_3': [
            {'player_level': 'master'},
            {'mana': 100},
            {'ritual_prepared': True,
             'brewed_transformation_potion': True,
             'brewed_healing_potion': True},
            {'cast_fireball': True, 'cast_freeze': True},
        ],
    }


def heuristic_magicworld(state, goal):
    """
    Heuristic for magicworld:
    Count missing goal facts.
    Also consider wizard level - lower level = higher heuristic.
    """
    missing = 0
    for g in goal:
        if g not in state or state[g] != goal[g]:
            missing += 1

    # Add extra cost if need to level up
    if 'player_level' in goal and 'player_level' in state:
        level_cost = 0
        goal_level = goal['player_level']
        current_level = state['player_level']
        if goal_level == 'journeyman' and current_level == 'apprentice':
            level_cost = 2
        elif goal_level == 'master' and current_level == 'journeyman':
            level_cost = 2
        elif goal_level == 'master' and current_level == 'apprentice':
            level_cost = 4
        missing += level_cost

    return missing
