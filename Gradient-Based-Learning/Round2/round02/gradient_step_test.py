""" Round 1 test cases

The functions defined in this script are intended to be used in the Jupyter Notebook for the Round 1.

It will perform tests for the following assignments:
    * Gradient Descent one feature: Given a set of test datasets (contained in the resources folder) it will generate
    an HTML report with the result of the student solution for the assignment Gradient Descent one feature.
"""
import glob
import os
import sys

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

from utils.test_result import TestResult
from utils.test_results_generator import generate_test_results
from utils.test_results_generator import check_all_succeeded

pwd = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(pwd, 'resources')
jsonpickle_numpy.register_handlers()


def __execute_test_case__(test_case, student_solution):
    test_input = test_case['input']
    expected_output = test_case['output']
    
    student_output = student_solution(test_input.x, test_input.y, test_input.weight, test_input.lrate)
    test_result = TestResult(test_input, student_output, expected_output, True)
    
    if not np.allclose(expected_output[0], student_output[0]):
        test_result.succeeded = False
    elif not np.allclose(expected_output[1], student_output[1]):
        test_result.succeeded = False
    return test_result

def test_gradient_step_one_feature(student_solution):
    sys.path.append(pwd)
    test_files = glob.glob(os.path.join(resources_dir, 'gradient_step_one_feature_test_case_*.json'))
    test_results = []
    for test_file in test_files:
        with open(test_file, 'r') as fin:
            test_case_serialized = fin.read()
            test_case = jsonpickle.decode(test_case_serialized)
            test_results.append(__execute_test_case__(test_case, student_solution))

    sys.path.remove(pwd)
    return generate_test_results(test_results)

def test_gradient_step(student_solution):
    sys.path.append(pwd)
    test_files = glob.glob(os.path.join(resources_dir, 'gradient_step_test_case_*.json'))
    test_results = []
    for test_file in test_files:
        with open(test_file, 'r') as fin:
            test_case_serialized = fin.read()
            test_case = jsonpickle.decode(test_case_serialized)
            test_results.append(__execute_test_case__(test_case, student_solution))

    sys.path.remove(pwd)
    return generate_test_results(test_results)

def hidden_test_gradient_step_one_feature(student_solution):
    sys.path.append(pwd)
    test_files = glob.glob(os.path.join(resources_dir, 'gradient_step_one_feature_test_case_*.json'))
    test_results = []
    for test_file in test_files:
        with open(test_file, 'r') as fin:
            test_case_serialized = fin.read()
            test_case = jsonpickle.decode(test_case_serialized)
            test_results.append(__execute_test_case__(test_case, student_solution))

    sys.path.remove(pwd)
    return check_all_succeeded(test_results)

def hidden_test_gradient_step(student_solution):
    sys.path.append(pwd)
    test_files = glob.glob(os.path.join(resources_dir, 'gradient_step_test_case_*.json'))
    test_results = []
    for test_file in test_files:
        with open(test_file, 'r') as fin:
            test_case_serialized = fin.read()
            test_case = jsonpickle.decode(test_case_serialized)
            test_results.append(__execute_test_case__(test_case, student_solution))

    sys.path.remove(pwd)
    return check_all_succeeded(test_results)
