class TestResult:
    """ A class to represent a test result.

    To make things simple, we are also passing the result of the test, instead of
    trying to determining it from the parameters :param:`output` and :param:`expected_output`.
    We could use some library to perform this comparison, but we have to take care of the
    numpy data types, which could get a bit complex.

    :param test_input: The input to the test case.
    :param output: The output of the function applied to the :param:`test_input`
    :param expected_output: The real or expected output for the :param:`test_input`
    :param succeeded if :param:`output` is equal to :param:`expected_output`
    """
    def __init__(self, test_input, output, expected_output, succeeded):
        self.test_input = test_input
        self.output = output
        self.expected_output = expected_output
        self.succeeded = succeeded
