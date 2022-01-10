"""Test Cases Generator for the Round 1

 This script allows to the instructors to generate data sets for the Round 1.
To make things easier, we are using the library `jsonpickle` to serialize the
objects in a readable JSON file.


"""

import os

import click
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

from reference_solutions import gradient_step_one_feature
from reference_solutions import gradient_step
from test_case_input import TestCaseInput

jsonpickle_numpy.register_handlers()

pwd = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(pwd, 'resources')
rng = np.random.RandomState(42)


@click.group()
def cli():
    pass


@cli.command(name='gradient_step_one_feature')

@click.option('--cases-count', '-n', default=5, help='Number of test cases to generate', show_default=True)
@click.option('--initial-input-size', '-m', default=5, help='Initial size for the input dataset. The consequent datasets will be a multiple of this value', show_default=True)
@click.option('--lrates', '-l', default=[0.5, 1.0, 1.5], multiple=True, help='Learning rates to use', show_default=True)
@click.option('--weights', '-w', default=[0.01, 0.1, 1.0], multiple=True, help='Weights to use', show_default=True)

def gradient_step_one_feature_test_generator(cases_count, initial_input_size, lrates, weights):
    """Test case generator for the function `gradient_step_one_feature`

    The help of the options will give a hint on how to use this function. If you are ok with the defaults
    parameters, just execute the script like:

    $ python3 generate_test_cases.py

    In the case of the `lrates` and `weights` if you want to pass multiple values, then you need to use the following
    way:

     $ python3 generate_test_cases.py -l 0.1 -l 0.01 -l 0.001

     This will execute the function with the lrates: [0.1, 0.01, 0.001]
    """
    file_name_template = 'gradient_step_one_feature_test_case'

    for i in range(cases_count):
        file_name = os.path.join(resources_dir, f'{file_name_template}_{i + 1}.json')
        x = rng.random((initial_input_size * (i + 1), 1))
        y = rng.random((initial_input_size * (i + 1),1))
        lrate = rng.choice(lrates, 1)[0]
        weight = rng.choice(weights, 1)[0]
        output = gradient_step_one_feature(x, y, weight, lrate)

        testCase = {
            'input': TestCaseInput(x, y, weight, lrate),
            'output': output
        }

        with open(file_name, 'w') as fout:
            serialized = jsonpickle.encode(testCase, indent=4)
            fout.write(serialized)

def gradient_step_test_generator(cases_count, initial_input_size, lrates, weights):
    """Test case generator for the function `gradient_step`

    The help of the options will give a hint on how to use this function. If you are ok with the defaults
    parameters, just execute the script like:

    $ python3 generate_test_cases.py

    In the case of the `lrates` and `weights` if you want to pass multiple values, then you need to use the following
    way:

     $ python3 generate_test_cases.py -l 0.1 -l 0.01 -l 0.001

     This will execute the function with the lrates: [0.1, 0.01, 0.001]
    """
    file_name_template = 'gradient_step_test_case'

    for i in range(cases_count):
        file_name = os.path.join(resources_dir, f'{file_name_template}_{i + 1}.json')
        x = rng.random((initial_input_size * (i + 1), (i + 1)))
        y = rng.random((initial_input_size * (i + 1),1))
        lrate = rng.choice(lrates, 1)[0]
        weight = np.random.rand((i + 1),1)
        output = gradient_step(x, y, weight, lrate)

        testCase = {
            'input': TestCaseInput(x, y, weight, lrate),
            'output': output
        }

        with open(file_name, 'w') as fout:
            serialized = jsonpickle.encode(testCase, indent=4)
            fout.write(serialized)




if __name__ == '__main__':
    cli()
