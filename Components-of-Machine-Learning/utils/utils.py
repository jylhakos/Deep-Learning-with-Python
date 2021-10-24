import importlib
import os
import sys

module_path = os.path.abspath(os.path.join('../../../coursedata'))

if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Temporarily hijack __file__ to avoid adding names at module scope;
# __file__ will be overwritten again during the reload() call.
__file__ = {'sys': sys, 'importlib': importlib}

del importlib
del os
del sys

__file__['importlib'].reload(__file__['sys'].modules[__name__])



