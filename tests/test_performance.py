"""
    Rosetta version 3-alpha (3a)
    Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
    Copyright (C) 2016  Marcel G. Schaap
    Copyright (C) 2021  Roger Lew <rogerlew@gmail.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Marcel G. Schaap can be contacted at:
    mschaap@cals.arizona.edu
"""

import os
import sys
import time
import statistics
import unittest

_TESTS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rosetta import Rosetta2, Rosetta3  # noqa: E402


class RosettaBenchmarkTest(unittest.TestCase):
    """Simple timing harness to track prediction performance."""

    #: Number of measured iterations (set via ROSETTA_BENCH_ITERS)
    iterations = int(os.getenv("ROSETTA_BENCH_ITERS", "50"))

    #: Warmup runs to prime caches and trigger first-use loading
    warmup_iterations = int(os.getenv("ROSETTA_BENCH_WARMUP", "1"))

    #: Environment toggle so the benchmark is opt-in
    run_benchmarks = os.getenv("ROSETTA_RUN_BENCHMARKS") == "1"

    sand = 45.0
    silt = 35.0
    clay = 20.0
    bd = 1.45

    @unittest.skipUnless(run_benchmarks, "Benchmarks disabled; set ROSETTA_RUN_BENCHMARKS=1 to enable")
    def test_prediction_hot_path_timings(self):
        metrics = {}

        def time_callable(label, func):
            for _ in range(self.warmup_iterations):
                func()

            durations = []
            perf_counter = time.perf_counter
            for _ in range(self.iterations):
                start = perf_counter()
                func()
                durations.append(perf_counter() - start)

            metrics[label] = dict(
                mean=statistics.mean(durations),
                stdev=statistics.pstdev(durations) if len(durations) > 1 else 0.0,
                minimum=min(durations),
                maximum=max(durations),
                runs=len(durations),
            )

        # Measure Rosetta3 construction (loads sqlite-backed models)
        time_callable("rosetta3_init", Rosetta3)

        # Keep a live instance around to isolate predict timings from init
        rosetta3_instance = Rosetta3()

        def rosetta3_predict():
            result = rosetta3_instance.predict_kwargs(
                sand=self.sand,
                silt=self.silt,
                clay=self.clay,
                bd=self.bd,
            )
            # Basic sanity check so we know predictions completed
            self.assertIn("theta_r", result)

        time_callable("rosetta3_predict", rosetta3_predict)

        # Measure Rosetta2 init
        time_callable("rosetta2_init", Rosetta2)

        rosetta2_instance = Rosetta2()

        def rosetta2_predict():
            result = rosetta2_instance.predict_kwargs(
                sand=self.sand,
                silt=self.silt,
                clay=self.clay,
            )
            self.assertIn("theta_r", result)

        time_callable("rosetta2_predict", rosetta2_predict)

        # Emit metrics so they are captured in test output / CI logs
        for label, data in metrics.items():
            print(
                f"BENCH {label}: mean={data['mean']:.6f}s stdev={data['stdev']:.6f}s "
                f"min={data['minimum']:.6f}s max={data['maximum']:.6f}s runs={data['runs']}"
            )

        # Assertions keep the test green while still guaranteeing timings ran
        self.assertTrue(metrics)


if __name__ == "__main__":
    unittest.main()


# ROSETTA_RUN_BENCHMARKS=1 python -m unittest tests.test_performance
# original sqlite-based benchmarks:
# BENCH rosetta3_init: mean=0.055598s stdev=0.004485s min=0.050961s max=0.076069s runs=50
# BENCH rosetta3_predict: mean=0.031704s stdev=0.001106s min=0.029135s max=0.034549s runs=50
# BENCH rosetta2_init: mean=0.053628s stdev=0.003374s min=0.049884s max=0.067250s runs=50
# BENCH rosetta2_predict: mean=0.030979s stdev=0.001475s min=0.029148s max=0.035426s runs=50

# duckdb
# BENCH rosetta3_init: mean=0.092406s stdev=0.006220s min=0.085315s max=0.117160s runs=50
# BENCH rosetta3_predict: mean=0.030783s stdev=0.002894s min=0.028342s max=0.046806s runs=50
# BENCH rosetta2_init: mean=0.088270s stdev=0.006863s min=0.081356s max=0.112032s runs=50
# BENCH rosetta2_predict: mean=0.031067s stdev=0.000890s min=0.029429s max=0.034822s runs=50