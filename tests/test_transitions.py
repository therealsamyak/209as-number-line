"""
Test suite for verifying analytical transition probability computation.
These tests verify the mathematical correctness of the grid discretization.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ContinuousParticleSystem, GridParticleSystem


class TestTransitionProbabilities:
    """Verify that transition probabilities are computed correctly"""

    def setup_simple_system(self):
        """Create a simple test system"""
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
            continuous_sys, delta_y=1.0, delta_v=1.0, goal_y=0.0, goal_v=0.0
        )
        return continuous_sys, grid_sys

    def test_probabilities_sum_to_one(self):
        """Critical: All transition probabilities must sum to 1.0"""
        print("\n" + "=" * 60)
        print("TEST 1: Probabilities Sum to 1.0")
        print("=" * 60)

        _, grid_sys = self.setup_simple_system()

        actions = [-1, 0, 1]
        failures = []

        # Check every state-action pair
        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                for a in actions:
                    trans = grid_sys.transitions.get((i, j, a), {})
                    prob_sum = sum(trans.values())

                    # Allow small numerical error
                    if abs(prob_sum - 1.0) > 1e-6:
                        failures.append(
                            {
                                "state": (i, j),
                                "action": a,
                                "sum": prob_sum,
                                "error": abs(prob_sum - 1.0),
                            }
                        )

        if failures:
            print(
                f"❌ FAILED: {len(failures)} state-action pairs have invalid probability sums"
            )
            print("\nFirst 5 failures:")
            for f in failures[:5]:
                print(
                    f"  State {f['state']}, action {f['action']}: sum = {f['sum']:.8f}, error = {f['error']:.2e}"
                )
            return False
        else:
            print(
                f"✅ PASSED: All {grid_sys.n_y * grid_sys.n_v * len(actions)} state-action pairs sum to 1.0"
            )
            return True

    def test_no_negative_probabilities(self):
        """Probabilities must be non-negative"""
        print("\n" + "=" * 60)
        print("TEST 2: No Negative Probabilities")
        print("=" * 60)

        _, grid_sys = self.setup_simple_system()

        actions = [-1, 0, 1]
        failures = []

        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                for a in actions:
                    trans = grid_sys.transitions.get((i, j, a), {})
                    for (i_next, j_next), prob in trans.items():
                        if prob < 0:
                            failures.append(
                                {
                                    "state": (i, j),
                                    "action": a,
                                    "next_state": (i_next, j_next),
                                    "prob": prob,
                                }
                            )

        if failures:
            print(f"❌ FAILED: Found {len(failures)} negative probabilities")
            for f in failures[:5]:
                print(
                    f"  {f['state']} -[a={f['action']}]-> {f['next_state']}: p = {f['prob']}"
                )
            return False
        else:
            print("✅ PASSED: All probabilities are non-negative")
            return True

    def test_zero_velocity_no_wobble(self):
        """When v=0, wobble should be zero (deterministic velocity transition)"""
        print("\n" + "=" * 60)
        print("TEST 3: Zero Velocity → Deterministic Update")
        print("=" * 60)

        continuous_sys, grid_sys = self.setup_simple_system()

        # Find grid cell closest to v=0
        j_zero = np.argmin(np.abs(grid_sys.v_grid))
        v_zero = grid_sys.v_grid[j_zero]

        print(f"Testing at v ≈ {v_zero:.4f} (grid index j={j_zero})")

        # Pick a middle position
        i_mid = grid_sys.n_y // 2
        y_mid = grid_sys.y_grid[i_mid]

        # Test each action
        test_passed = True
        for a in [-1, 0, 1]:
            trans = grid_sys.transitions.get((i_mid, j_zero, a), {})

            # Compute expected deterministic transition
            f_phi = continuous_sys.phi(y_mid)
            f_net = a + f_phi
            v_next_expected = v_zero + (f_net / continuous_sys.m)
            v_next_expected = np.clip(
                v_next_expected, -continuous_sys.v_max, continuous_sys.v_max
            )

            # Position update (with current velocity v≈0)
            y_next_expected = y_mid + v_zero

            print(f"\n  Action a={a}:")
            print(f"    Expected: y'={y_next_expected:.4f}, v'={v_next_expected:.4f}")

            # Check if most probability mass is at expected location
            if len(trans) > 0:
                max_prob_state = max(trans.items(), key=lambda x: x[1])
                (i_max, j_max), max_prob = max_prob_state
                y_max, v_max_actual = grid_sys.continuous_state(i_max, j_max)

                print(
                    f"    Most likely: y'={y_max:.4f}, v'={v_max_actual:.4f} (p={max_prob:.4f})"
                )
                print(f"    Entropy: {len(trans)} non-zero transitions")

                # For v≈0, we expect low entropy (few possible next states)
                if len(trans) > 5:
                    print(f"    ⚠️  Warning: High entropy for v≈0 case")
                    test_passed = False
            else:
                print(f"    ❌ No transitions found!")
                test_passed = False

        if test_passed:
            print("\n✅ PASSED: Zero velocity case behaves reasonably")
        else:
            print("\n⚠️  WARNING: Zero velocity case may have issues")

        return test_passed

    def test_crash_probability_correctness(self):
        """Verify crash probability is computed correctly"""
        print("\n" + "=" * 60)
        print("TEST 4: Crash Probability Correctness")
        print("=" * 60)

        continuous_sys, grid_sys = self.setup_simple_system()

        # Test at high velocity (should have high crash prob)
        j_high = grid_sys.n_v - 1  # Highest velocity
        v_high = grid_sys.v_grid[j_high]

        # Expected crash probability
        p_crash_expected = continuous_sys.p_c * abs(v_high) / continuous_sys.v_max
        p_crash_expected = min(p_crash_expected, 1.0)

        print(f"At v={v_high:.2f}:")
        print(f"  Expected crash probability: {p_crash_expected:.4f}")

        # Pick a position and action
        i_mid = grid_sys.n_y // 2
        y_mid = grid_sys.y_grid[i_mid]
        a = 0  # Neutral action

        # Get transitions
        trans = grid_sys.transitions.get((i_mid, j_high, a), {})

        # Find the crash state (v'=0, y unchanged)
        i_crash, j_crash = grid_sys.discretize(y_mid, 0.0)
        p_crash_actual = trans.get((i_crash, j_crash), 0.0)

        print(f"  Actual crash probability in transitions: {p_crash_actual:.4f}")
        print(f"  Error: {abs(p_crash_actual - p_crash_expected):.4f}")

        # Allow some tolerance due to discretization
        if abs(p_crash_actual - p_crash_expected) < 0.1:
            print("✅ PASSED: Crash probability is approximately correct")
            return True
        else:
            print("❌ FAILED: Crash probability differs significantly from expected")
            return False

    def test_boundary_states(self):
        """Test that boundary states are handled correctly"""
        print("\n" + "=" * 60)
        print("TEST 5: Boundary State Handling")
        print("=" * 60)

        _, grid_sys = self.setup_simple_system()

        # Test corner states
        corners = [
            (0, 0, "bottom-left"),
            (0, grid_sys.n_v - 1, "top-left"),
            (grid_sys.n_y - 1, 0, "bottom-right"),
            (grid_sys.n_y - 1, grid_sys.n_v - 1, "top-right"),
        ]

        test_passed = True
        for i, j, name in corners:
            y, v = grid_sys.continuous_state(i, j)
            print(f"\n  Testing {name}: ({y:.2f}, {v:.2f})")

            for a in [-1, 0, 1]:
                trans = grid_sys.transitions.get((i, j, a), {})
                prob_sum = sum(trans.values())

                if abs(prob_sum - 1.0) > 1e-6:
                    print(f"    ❌ Action {a}: probability sum = {prob_sum:.6f}")
                    test_passed = False
                else:
                    print(f"    ✓ Action {a}: probability sum = {prob_sum:.6f}")

        if test_passed:
            print("\n✅ PASSED: Boundary states handled correctly")
        else:
            print("\n❌ FAILED: Some boundary states have invalid probabilities")

        return test_passed


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 70)
    print("TRANSITION PROBABILITY VERIFICATION TEST SUITE")
    print("=" * 70)

    tester = TestTransitionProbabilities()

    results = {
        "Probabilities Sum to 1.0": tester.test_probabilities_sum_to_one(),
        "No Negative Probabilities": tester.test_no_negative_probabilities(),
        "Zero Velocity (No Wobble)": tester.test_zero_velocity_no_wobble(),
        "Crash Probability": tester.test_crash_probability_correctness(),
        "Boundary States": tester.test_boundary_states(),
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review implementation")

    return results


if __name__ == "__main__":
    run_all_tests()
