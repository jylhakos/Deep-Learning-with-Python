""" CSS style loader

 This script allows the user to define a custom CSS style to use in the Jupyter Notebooks.
You SHOULD put the call of this function as the last line in the notebook's cell to avoid
having to manually set the HTML in the kernel context.

"""

import os.path

from IPython.core.display import HTML

from utils import resources_dir


def load_styles():
    """Load the custom.css file from the resources directory

    Notice that in order to the CSS to be loaded properly, it MUST be wrapped in the `style` tag.
    """
    with open(os.path.join(resources_dir, 'custom.css')) as fin:
        css = fin.read()
        custom_css = f'<style>\n{css}\n</style>'
        return HTML(custom_css)
