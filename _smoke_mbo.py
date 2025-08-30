import importlib, sys
sys.path.append('c:/Users/Kushtrimi/Desktop/AI')
try:
    mbo_mod = importlib.import_module('modules.memory.memory_budget_optimizer')
    cls = getattr(mbo_mod, 'MemoryBudgetOptimizer')
    inst = cls()
    print('OK', type(inst).__name__)
except Exception as e:
    print('ERR', repr(e))
