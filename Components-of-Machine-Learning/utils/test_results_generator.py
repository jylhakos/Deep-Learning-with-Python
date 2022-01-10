"""Test results generator

 This script will generate the needed HTML code to show the results of the test in an HTML list.
You can check the content of the file `test_results_template.hmtl.j2` for details of the HTML
that will be generated.

Notice that there there is not `ul` or `ol` tag. This is because we are modeling the list as divs.
This is important to know, because the CSS style you use needs to work with the classes defined there.

The :func:`generate_test_results` is not intended to be used directly in the Notebooks' cells. Instead, you should
use it in the scripts where the test cases are executed.
"""

from typing import List
from IPython.core.display import HTML

from jinja2 import Environment, FileSystemLoader
from utils import resources_dir
from utils.test_result import TestResult

j2_env = Environment(loader=FileSystemLoader(resources_dir), trim_blocks=True)


def generate_test_results(test_results: List[TestResult]) -> HTML:
    """HTML List generator

    This function will pass the variables to the Jinja2 context in order to render the HTML template.
    If introduce or remove variables from the template, make sure that you also make the changes in the
    invocation of the `render` method

    :param test_results: A list of :class:`TestResult`
    """

    all_succeeded = check_all_succeeded(test_results)
    title = 'Congratulations!' if all_succeeded else 'Wrong Answer :('

    subtitle = 'You have passed the test cases.' \
        if all_succeeded else \
        f'{sum(not test.succeeded for test in test_results)}/{len(test_results)} test cases failed'

    return HTML(
        j2_env.get_template('test_results_template.hmtl.j2')
            .render(
               title=title,
               subtitle=subtitle,
               test_results=test_results
            ))


def check_all_succeeded(test_results: List[TestResult]):
    """Check whether all the tests are successful or not

    We could achieve the same result using functools.reduce.
    """
    result = True
    for test_result in test_results:
        result = result & test_result.succeeded

    return result
