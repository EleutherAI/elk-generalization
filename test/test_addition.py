import math
import random
from elk_generalization.generate_sloppy_dataset import add
from hypothesis import given, settings
import hypothesis.strategies as st

@given(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))  # this tests random floats between 0 and 1
@settings(deadline=2000, max_examples=20)  # this tests at most 20 inputs to the function, with a deadline of 2 seconds
def test_addition_instance(err_rate):
    # check that the function works
    num_sloppy_correct = 0
    n=300_000
    for i in range(n):
        r1, r2 = int(2**(random.random() * 16)), int(2**(random.random() * 16))
        real_sum, sloppy_sum = add(r1, r2), add(r1, r2, err_rate)
        num_sloppy_correct += real_sum == sloppy_sum
    p_err = 1 - num_sloppy_correct / n
    assert math.isclose(p_err, err_rate, abs_tol=0.005)
