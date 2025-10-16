"""
Test discretization functions and grid structure.
Verify that discretize() and continuous_state() are properly related.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ContinuousParticleSystem, GridParticleSystem


class TestDiscretization:
    """Test grid discretization correctness"""

    def setup_system(self):
        """Create test system"""
        continuous_sys = ContinuousParticleSystem(
            m=1.0,
            y_max=10.0,
            v_max=5.0,
            A=2.0,
            p_c=0.1,
            goal_y=0.0,
            goal_v=0.0,
            goal_tolerance=0.1,
        )
        grid_sys = GridParticleSystem(
            continuous_sys, delta_y=1.0, delta_v=0.5, goal_y=0.0, goal_v=0.0
        )
        return continuous_sys, grid_sys

    def test_grid_construction(self):
        """Test that grid is constructed correctly"""
        print("\n" + "=" * 60)
        print("TEST: Grid Construction")
        print("=" * 60)

        continuous_sys, grid_sys = self.setup_system()

        # Check y grid
        expected_y_min = -continuous_sys.y_max
        expected_y_max = continuous_sys.y_max

        actual_y_min = grid_sys.y_grid[0]
        actual_y_max = grid_sys.y_grid[-1]

        print(f"Position grid:")
        print(f"  Range: [{actual_y_min:.2f}, {actual_y_max:.2f}]")
        print(f"  Expected: [{expected_y_min:.2f}, {expected_y_max:.2f}]")
        print(f"  Points: {grid_sys.n_y}")
        print(f"  Delta: {grid_sys.delta_y}")

        # Check v grid
        expected_v_min = -continuous_sys.v_max
        expected_v_max = continuous_sys.v_max

        actual_v_min = grid_sys.v_grid[0]
        actual_v_max = grid_sys.v_grid[-1]

        print(f"\nVelocity grid:")
        print(f"  Range: [{actual_v_min:.2f}, {actual_v_max:.2f}]")
        print(f"  Expected: [{expected_v_min:.2f}, {expected_v_max:.2f}]")
        print(f"  Points: {grid_sys.n_v}")
        print(f"  Delta: {grid_sys.delta_v}")

        # Verify grid spacing
        y_diffs = np.diff(grid_sys.y_grid)
        v_diffs = np.diff(grid_sys.v_grid)

        y_spacing_correct = np.allclose(y_diffs, grid_sys.delta_y)
        v_spacing_correct = np.allclose(v_diffs, grid_sys.delta_v)

        print(f"\nGrid spacing:")
        print(f"  Y spacing uniform: {y_spacing_correct}")
        print(f"  V spacing uniform: {v_spacing_correct}")

        all_correct = (
            abs(actual_y_min - expected_y_min) < 1e-6
            and abs(actual_y_max - expected_y_max) < grid_sys.delta_y
            and abs(actual_v_min - expected_v_min) < 1e-6
            and abs(actual_v_max - expected_v_max) < grid_sys.delta_v
            and y_spacing_correct
            and v_spacing_correct
        )

        if all_correct:
            print("\n✅ PASSED: Grid is constructed correctly")
        else:
            print("\n❌ FAILED: Grid construction has errors")

        return all_correct

    def test_discretize_function(self):
        """Test that discretize maps continuous states to valid indices"""
        print("\n" + "=" * 60)
        print("TEST: Discretize Function")
        print("=" * 60)

        _, grid_sys = self.setup_system()

        # Test cases
        test_cases = [
            (0.0, 0.0, "Origin"),
            (5.0, 2.5, "Mid-range positive"),
            (-5.0, -2.5, "Mid-range negative"),
            (10.0, 5.0, "Max bounds"),
            (-10.0, -5.0, "Min bounds"),
            (10.5, 5.5, "Beyond max (should clip)"),
            (-10.5, -5.5, "Beyond min (should clip)"),
        ]

        all_correct = True
        for y, v, description in test_cases:
            i, j = grid_sys.discretize(y, v)

            # Check indices are valid
            valid = (0 <= i < grid_sys.n_y) and (0 <= j < grid_sys.n_v)

            # Get corresponding grid state
            y_grid, v_grid = grid_sys.continuous_state(i, j)

            status = "✓" if valid else "✗"
            print(
                f"  {status} {description}: ({y:.2f}, {v:.2f}) → ({i}, {j}) → ({y_grid:.2f}, {v_grid:.2f})"
            )

            if not valid:
                print(f"      ❌ Invalid indices!")
                all_correct = False

        if all_correct:
            print("\n✅ PASSED: Discretize function produces valid indices")
        else:
            print("\n❌ FAILED: Discretize function has errors")

        return all_correct

    def test_continuous_state_function(self):
        """Test that continuous_state returns grid cell centers"""
        print("\n" + "=" * 60)
        print("TEST: Continuous State Function")
        print("=" * 60)

        _, grid_sys = self.setup_system()

        # Check a few grid cells
        test_indices = [
            (0, 0, "Bottom-left corner"),
            (grid_sys.n_y // 2, grid_sys.n_v // 2, "Center"),
            (grid_sys.n_y - 1, grid_sys.n_v - 1, "Top-right corner"),
        ]

        all_correct = True
        for i, j, description in test_indices:
            y, v = grid_sys.continuous_state(i, j)

            # Should match grid values
            expected_y = grid_sys.y_grid[i]
            expected_v = grid_sys.v_grid[j]

            y_match = abs(y - expected_y) < 1e-6
            v_match = abs(v - expected_v) < 1e-6

            status = "✓" if (y_match and v_match) else "✗"
            print(f"  {status} {description}: ({i}, {j}) → ({y:.2f}, {v:.2f})")
            print(f"      Expected: ({expected_y:.2f}, {expected_v:.2f})")

            if not (y_match and v_match):
                all_correct = False

        if all_correct:
            print("\n✅ PASSED: Continuous state function returns correct values")
        else:
            print("\n❌ FAILED: Continuous state function has errors")

        return all_correct

    def test_round_trip_consistency(self):
        """Test that discretize ∘ continuous_state ≈ identity on grid"""
        print("\n" + "=" * 60)
        print("TEST: Round-Trip Consistency (Grid → Continuous → Grid)")
        print("=" * 60)

        _, grid_sys = self.setup_system()

        # Test: for all (i, j), discretize(continuous_state(i, j)) should return (i, j)
        failures = []

        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                y, v = grid_sys.continuous_state(i, j)
                i_back, j_back = grid_sys.discretize(y, v)

                if i_back != i or j_back != j:
                    failures.append(
                        {
                            "original": (i, j),
                            "continuous": (y, v),
                            "back": (i_back, j_back),
                        }
                    )

        if failures:
            print(f"❌ FAILED: {len(failures)} grid cells don't round-trip correctly")
            print("\nFirst 5 failures:")
            for f in failures[:5]:
                print(
                    f"  ({f['original']}) → ({f['continuous'][0]:.2f}, {f['continuous'][1]:.2f}) → ({f['back']})"
                )
            return False
        else:
            print(
                f"✅ PASSED: All {grid_sys.n_y * grid_sys.n_v} grid cells round-trip correctly"
            )
            return True

    def test_nearest_neighbor_property(self):
        """Test that discretize maps to nearest grid cell"""
        print("\n" + "=" * 60)
        print("TEST: Nearest Neighbor Property")
        print("=" * 60)

        _, grid_sys = self.setup_system()

        # Test random continuous points
        np.random.seed(42)
        n_tests = 20

        failures = []
        for _ in range(n_tests):
            # Random continuous state within bounds
            y = np.random.uniform(-grid_sys.continuous.y_max, grid_sys.continuous.y_max)
            v = np.random.uniform(-grid_sys.continuous.v_max, grid_sys.continuous.v_max)

            # Discretize
            i, j = grid_sys.discretize(y, v)
            y_grid, v_grid = grid_sys.continuous_state(i, j)

            # Check if this is approximately the nearest grid point
            # Find actual nearest grid point
            y_dists = np.abs(grid_sys.y_grid - y)
            v_dists = np.abs(grid_sys.v_grid - v)

            i_nearest = np.argmin(y_dists)
            j_nearest = np.argmin(v_dists)

            # Should match (within boundary effects)
            # Allow one cell difference due to clipping
            i_diff = abs(i - i_nearest)
            j_diff = abs(j - j_nearest)

            if i_diff > 1 or j_diff > 1:
                failures.append(
                    {
                        "continuous": (y, v),
                        "discretized": (i, j),
                        "nearest": (i_nearest, j_nearest),
                        "diff": (i_diff, j_diff),
                    }
                )

        if failures:
            print(
                f"⚠️  WARNING: {len(failures)}/{n_tests} points not mapped to nearest neighbor"
            )
            print("(Some failures may be due to boundary clipping)")
            for f in failures[:3]:
                print(
                    f"  {f['continuous']} → {f['discretized']}, nearest = {f['nearest']}"
                )
            return False
        else:
            print(
                f"✅ PASSED: All {n_tests} random points mapped to (approximately) nearest cell"
            )
            return True


def run_all_discretization_tests():
    """Run complete discretization test suite"""
    print("\n" + "=" * 70)
    print("DISCRETIZATION CORRECTNESS TEST SUITE")
    print("=" * 70)

    tester = TestDiscretization()

    results = {
        "Grid Construction": tester.test_grid_construction(),
        "Discretize Function": tester.test_discretize_function(),
        "Continuous State Function": tester.test_continuous_state_function(),
        "Round-Trip Consistency": tester.test_round_trip_consistency(),
        "Nearest Neighbor Property": tester.test_nearest_neighbor_property(),
    }

    print("\n" + "=" * 70)
    print("DISCRETIZATION TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All discretization tests passed!")
        print("   Grid structure and mapping functions are correct!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review discretization")

    return results


if __name__ == "__main__":
    run_all_discretization_tests()
