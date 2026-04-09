"""
DINNER (Restaurant Dinner Planning)
Domain: Planning a dinner - shopping, preparing ingredients, cooking meals
Source: https://github.com/primaryobjects/strips/tree/master/examples/dinner

Problem: Prepare a dinner with multiple guests
- Ingredients must be bought
- Items must be prepared
- Meals must be cooked
- Table must be set
- Guests must be invited
"""
from aipython_strips import Strips, STRIPS_domain, Planning_problem

# ==================================================================================
# DINNER DOMAIN DEFINITION
# ==================================================================================

def create_dinner_domain():
    """
    Features:
    - have_<ingredient>: {True, False} - whether we have the ingredient
    - prepared_<item>: {True, False} - whether item is prepared
    - cooked_<meal>: {True, False} - whether meal is cooked
    - table_set: {True, False} - whether table is set
    - <guest>_invited: {True, False} - whether guest is invited
    - <guest>_present: {True, False} - whether guest has arrived
    """

    boolean = {True, False}

    feature_domain_dict = {
        # Ingredients
        'have_chicken': boolean,
        'have_rice': boolean,
        'have_vegetables': boolean,
        'have_wine': boolean,
        'have_dessert_ingredients': boolean,

        # Preparation
        'prepared_chicken': boolean,
        'prepared_vegetables': boolean,
        'prepared_rice': boolean,
        'prepared_dessert': boolean,

        # Cooking
        'cooked_chicken': boolean,
        'cooked_rice': boolean,
        'cooked_dessert': boolean,

        # Setup
        'table_set': boolean,
        'wine_opened': boolean,

        # Guests status (3 guests)
        'alice_invited': boolean,
        'bob_invited': boolean,
        'carol_invited': boolean,
        'alice_present': boolean,
        'bob_present': boolean,
        'carol_present': boolean,
    }

    # Actions
    actions = set()

    # SHOP actions - buy ingredients
    actions.add(Strips('shop_chicken',
                       {},
                       {'have_chicken': True}))

    actions.add(Strips('shop_rice',
                       {},
                       {'have_rice': True}))

    actions.add(Strips('shop_vegetables',
                       {},
                       {'have_vegetables': True}))

    actions.add(Strips('shop_wine',
                       {},
                       {'have_wine': True}))

    actions.add(Strips('shop_dessert_ingredients',
                       {},
                       {'have_dessert_ingredients': True}))

    # PREPARE actions - prepare raw ingredients
    actions.add(Strips('prepare_chicken',
                       {'have_chicken': True},
                       {'prepared_chicken': True, 'have_chicken': False}))

    actions.add(Strips('prepare_vegetables',
                       {'have_vegetables': True},
                       {'prepared_vegetables': True, 'have_vegetables': False}))

    actions.add(Strips('prepare_rice',
                       {'have_rice': True},
                       {'prepared_rice': True, 'have_rice': False}))

    actions.add(Strips('prepare_dessert',
                       {'have_dessert_ingredients': True},
                       {'prepared_dessert': True, 'have_dessert_ingredients': False}))

    # COOK actions - transform prepared into cooked
    actions.add(Strips('cook_chicken',
                       {'prepared_chicken': True},
                       {'cooked_chicken': True}))

    actions.add(Strips('cook_rice',
                       {'prepared_rice': True},
                       {'cooked_rice': True}))

    actions.add(Strips('cook_dessert',
                       {'prepared_dessert': True},
                       {'cooked_dessert': True}))

    # SETUP actions
    actions.add(Strips('set_table',
                       {},
                       {'table_set': True}))

    actions.add(Strips('open_wine',
                       {'have_wine': True},
                       {'wine_opened': True}))

    # INVITE actions
    actions.add(Strips('invite_alice',
                       {},
                       {'alice_invited': True}))

    actions.add(Strips('invite_bob',
                       {},
                       {'bob_invited': True}))

    actions.add(Strips('invite_carol',
                       {},
                       {'carol_invited': True}))

    # GUEST ARRIVAL actions - guest arrives if invited
    actions.add(Strips('alice_arrives',
                       {'alice_invited': True, 'cooked_chicken': True,
                        'cooked_rice': True, 'cooked_dessert': True, 'table_set': True},
                       {'alice_present': True}))

    actions.add(Strips('bob_arrives',
                       {'bob_invited': True, 'cooked_chicken': True,
                        'cooked_rice': True, 'cooked_dessert': True, 'table_set': True},
                       {'bob_present': True}))

    actions.add(Strips('carol_arrives',
                       {'carol_invited': True, 'cooked_chicken': True,
                        'cooked_rice': True, 'cooked_dessert': True, 'table_set': True},
                       {'carol_present': True}))

    return STRIPS_domain(feature_domain_dict, actions)


def dinner_problem_1():
    """
    PROBLEM 1 (Basic - 4+ actions)
    Goal: Prepare simple dinner and have one guest arrive
    - Cook chicken and rice
    - Set table
    - Invite and have Alice arrive
    - Requires ~6 actions minimum
    """
    domain = create_dinner_domain()

    initial_state = {
        'have_chicken': False,
        'have_rice': False,
        'have_vegetables': False,
        'have_wine': False,
        'have_dessert_ingredients': False,
        'prepared_chicken': False,
        'prepared_vegetables': False,
        'prepared_rice': False,
        'prepared_dessert': False,
        'cooked_chicken': False,
        'cooked_rice': False,
        'cooked_dessert': False,
        'table_set': False,
        'wine_opened': False,
        'alice_invited': False,
        'bob_invited': False,
        'carol_invited': False,
        'alice_present': False,
        'bob_present': False,
        'carol_present': False,
    }

    goal = {
        'cooked_chicken': True,
        'cooked_rice': True,
        'cooked_dessert': True,
        'table_set': True,
        'alice_present': True,
    }

    return Planning_problem(domain, initial_state, goal)


def dinner_problem_2():
    """
    PROBLEM 2 (Medium)
    Goal: Prepare full dinner with 2 guests
    - Medium complexity with more guests
    - Requires wine to be opened
    - Requires ~10 actions minimum
    """
    domain = create_dinner_domain()

    initial_state = {
        'have_chicken': False,
        'have_rice': False,
        'have_vegetables': False,
        'have_wine': False,
        'have_dessert_ingredients': False,
        'prepared_chicken': False,
        'prepared_vegetables': False,
        'prepared_rice': False,
        'prepared_dessert': False,
        'cooked_chicken': False,
        'cooked_rice': False,
        'cooked_dessert': False,
        'table_set': False,
        'wine_opened': False,
        'alice_invited': False,
        'bob_invited': False,
        'carol_invited': False,
        'alice_present': False,
        'bob_present': False,
        'carol_present': False,
    }

    goal = {
        'cooked_chicken': True,
        'cooked_rice': True,
        'cooked_dessert': True,
        'wine_opened': True,
        'table_set': True,
        'alice_present': True,
        'bob_present': True,
    }

    return Planning_problem(domain, initial_state, goal)


def dinner_problem_3():
    """
    PROBLEM 3 (Complex)
    Goal: Complete dinner with all guests, vegetables, and wine
    - All 3 guests must arrive
    - All dishes must be cooked
    - Wine must be opened
    - Requires ~15 actions minimum
    """
    domain = create_dinner_domain()

    initial_state = {
        'have_chicken': False,
        'have_rice': False,
        'have_vegetables': False,
        'have_wine': False,
        'have_dessert_ingredients': False,
        'prepared_chicken': False,
        'prepared_vegetables': False,
        'prepared_rice': False,
        'prepared_dessert': False,
        'cooked_chicken': False,
        'cooked_rice': False,
        'cooked_dessert': False,
        'table_set': False,
        'wine_opened': False,
        'alice_invited': False,
        'bob_invited': False,
        'carol_invited': False,
        'alice_present': False,
        'bob_present': False,
        'carol_present': False,
    }

    goal = {
        'cooked_chicken': True,
        'cooked_rice': True,
        'cooked_dessert': True,
        'wine_opened': True,
        'table_set': True,
        'alice_present': True,
        'bob_present': True,
        'carol_present': True,
    }

    return Planning_problem(domain, initial_state, goal)


# Subgoals for problems (for 6-point task)
def get_dinner_subgoals():
    """
    Subgoals for dinner problems:
    1. Problem 1: (a) Prepare ingredients, (b) Cook main course
    2. Problem 2: (a) Prepare ingredients, (b) Cook meals, (c) Setup and open wine
    3. Problem 3: (a) Shop all ingredients, (b) Prepare all, (c) Cook main, (d) Setup
    """
    return {
        'dinner_1': [
            {'prepared_chicken': True, 'prepared_rice': True, 'prepared_dessert': True},
            {'cooked_chicken': True, 'cooked_rice': True, 'cooked_dessert': True},
        ],
        'dinner_2': [
            {'prepared_chicken': True, 'prepared_rice': True, 'prepared_dessert': True},
            {'cooked_chicken': True, 'cooked_rice': True},
            {'table_set': True, 'wine_opened': True},
            {'cooked_chicken': True, 'cooked_rice': True},
        ],
        'dinner_3': [
            {'have_chicken': True, 'have_rice': True, 'have_dessert_ingredients': True},
            {'prepared_chicken': True, 'prepared_rice': True, 'prepared_dessert': True},
            {'cooked_chicken': True, 'cooked_rice': True, 'cooked_dessert': True},
            {'table_set': True, 'wine_opened': True},
            {'have_chicken': True, 'have_rice': True, 'have_dessert_ingredients': True},
            {'prepared_chicken': True, 'prepared_rice': True, 'prepared_dessert': True},
        ],
    }


# Heuristic function
def heuristic_dinner(state, goal):
    """
    Heuristic for dinner planning:
    Count how many goal facts are not yet satisfied.
    Each missing fact needs at least one action to achieve.
    """
    return len([g for g in goal if g not in state or state[g] != goal[g]])
