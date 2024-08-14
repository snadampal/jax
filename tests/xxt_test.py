# Copyright 2024 The JAX Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl.testing import absltest
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import linalg as lax_linalg

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
config.parse_flags_with_absl()


class XxtTest(jtu.JaxTestCase):

  @jtu.sample_product(
      shape=[
          (128, 12),
          (128, 64),
          (4096, 128),
      ],
      dtype=[jnp.float32, jnp.float64],
      symmetrize_output=[True, False],
  )
  def testRandomMatrix(self, shape, dtype, symmetrize_output):
    if dtype is jnp.float64 and not config.enable_x64.value:
      self.skipTest("Test disabled for x32 mode")
    rng = jtu.rand_default(self.rng())
    a_matrix = rng(shape, dtype)

    old_xxt = a_matrix @ a_matrix.T
    new_xxt = lax_linalg.xxt(a_matrix, symmetrize_output=symmetrize_output)
    atol = 1e-6 if dtype == jnp.float64 else 1e-3
    if not symmetrize_output:
      new_xxt = jnp.tril(new_xxt)
      old_xxt = jnp.tril(old_xxt)
    jtu._assert_numpy_allclose(new_xxt, old_xxt, atol=atol)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
