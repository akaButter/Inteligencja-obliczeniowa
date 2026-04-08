from aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips
# https://github.com/primaryobjects/strips/blob/master/examples/blocksworld4
domain = STRIPS_domain({"block":{'a','b','b'}, 'table': {'t1','t2','t3'}})

boolean = {False, True}
def move(x,y,z):
    """string for the 'move' action"""
    return 'move_'+x+'_from_'+y+'_to_'+z
def on(x):
    """string for the 'on' feature"""
    return x+'_is_on'
def clear(x):
    """string for the 'clear' feature"""
    return 'clear_'+x

def create_blocks_world(blocks = {'a','b','c'}, tables = {'t1', 't2', 't3'}):
    blocks_and_tables = blocks | tables
    stmap =  {Strips(move(x,y,z),{on(x):y, clear(x):True, clear(z):True}, # poruszamy x z y do z
                                 {on(x):z, clear(y):True, clear(z):False})
                    for x in blocks
                    for y in blocks_and_tables
                    for z in blocks
                    if x!=y and y!=z and z!=x}
    stmap.update({Strips(move(x,y,z), {on(x):y, clear(x):True}, 
                                 {on(x):'table', clear(y):True})
                    for x in blocks
                    for y in blocks
                    for z in tables
                    if x!=y})
    feature_domain_dict = {on(x):blocks_and_tables-{x} for x in blocks}
    feature_domain_dict.update({clear(x):boolean for x in blocks_and_tables})
    return STRIPS_domain(feature_domain_dict, stmap)