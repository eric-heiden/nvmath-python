import cupy as cp
import warp as wp

import nvmath

# Prepare sample input data.
n, m, k = 123, 456, 789
a = cp.random.rand(n, k)
b = cp.random.rand(k, m)

a_wp = wp.array(a)
b_wp = wp.array(b)

# Perform the multiplication.
result = nvmath.linalg.advanced.matmul(a_wp, b_wp)

# Synchronize the default stream, since by default the execution is non-blocking for GPU operands.
# cp.cuda.get_current_stream().synchronize()
wp.synchronize()

# Check if the result is cupy array as well.
print(f"Inputs were of types {type(a_wp)} and {type(b_wp)} and the result is of type {type(result)}.")
assert isinstance(result, wp.array)

print("Warp:")
print(result.numpy())


result = nvmath.linalg.advanced.matmul(a, b)
print("CuPy:")
print(result)