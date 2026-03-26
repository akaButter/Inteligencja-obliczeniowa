from aipython.stripsProblem import STRIPS_domain, Planning_problem
# https://github.com/primaryobjects/strips/blob/master/examples/blocksworld4
domain = STRIPS_domain({"block":{'a','b','b'}, 'table': {'t1','t2','t3'}})

def on(x):
    """string for the 'on' feature"""
    return x+'_is_on'
def clear(x):
    """string for the 'clear' feature"""
    return 'clear_'+x