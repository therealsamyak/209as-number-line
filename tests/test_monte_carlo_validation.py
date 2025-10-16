"""
Monte Carlo validation of analytical transition probabilities.

This test compares the analytical computation against empirical sampling
from the actual continuous system dynamics.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ContinuousParticleSystem, GridParticleSystem
from collections import defaultdict


class MonteCarloValidator:
    """Validate analytical transitions using Monte Carlo sampling"""

    def __init__(self, continuous_sys, grid_sys, n_samples=10000):
        self.continuous = continuous_sys
        self.grid = grid_sys
        self.n_samples = n_samples

    def sample_transitions(self, i, j, a):
        """
        Sample transitions from continuous system to build empirical distribution

        Args:
            i, j: grid indices for state
            a: action

        Returns:
            dict: empirical transition probabilities {(i', j'): count/n_samples}
        """
        y, v = self.grid.continuous_state(i, j)

        next_state_counts = defaultdict(int)

        # Run many samples
        for _ in range(self.n_samples):
            # Reset to same initial state
            state = (y, v)

            # Manually step through dynamics (without reward/done)
            f_net = a + self.continuous.phi(y)

            # Check crash
            p_crash = self.continuous.p_c * abs(v) / self.continuous.v_max
            if np.random.rand() < p_crash:
                # Crash
                v_next = 0.0
                y_next = y
            else:
                # No crash
                speed_wobble = np.random.normal(0, 0.1 * abs(v))
                v_next = v + (f_net / self.continuous.m) + speed_wobble
                v_next = np.clip(v_next, -self.continuous.v_max, self.continuous.v_max)
                y_next = y + v
                y_next = np.clip(y_next, -self.continuous.y_max, self.continuous.y_max)

            # Discretize next state
            i_next, j_next = self.grid.discretize(y_next, v_next)
            next_state_counts[(i_next, j_next)] += 1

        # Convert counts to probabilities
        empirical_trans = {k: v / self.n_samples for k, v in next_state_counts.items()}

        return empirical_trans

    def compare_distributions(self, analytical, empirical, tolerance=0.05):
        """
        Compare analytical and empirical distributions

        Args:
            analytical: dict of analytical probabilities
            empirical: dict of empirical probabilities
            tolerance: acceptable difference threshold

        Returns:
            bool: whether distributions match within tolerance
            dict: detailed comparison metrics
        """
        # Get all states that appear in either distribution
        all_states = set(analytical.keys()) | set(empirical.keys())

        differences = []
        for state in all_states:
            p_analytical = analytical.get(state, 0.0)
            p_empirical = empirical.get(state, 0.0)
            diff = abs(p_analytical - p_empirical)
            differences.append((state, p_analytical, p_empirical, diff))

        # Compute metrics
        max_diff = max(d[3] for d in differences) if differences else 0.0
        avg_diff = np.mean([d[3] for d in differences]) if differences else 0.0

        # Total Variation Distance: 0.5 * sum |p - q|
        tv_distance = 0.5 * sum(d[3] for d in differences)

        matches = max_diff <= tolerance

        return matches, {
            "max_diff": max_diff,
            "avg_diff": avg_diff,
            "tv_distance": tv_distance,
            "differences": sorted(differences, key=lambda x: x[3], reverse=True),
        }

    def validate_state_action(self, i, j, a, tolerance=0.05, verbose=True):
        """Validate a single state-action pair"""
        y, v = self.grid.continuous_state(i, j)

        if verbose:
            print(f"\nValidating state ({i}, {j}) = ({y:.2f}, {v:.2f}), action = {a}")

        # Get analytical transitions
        analytical = self.grid.transitions.get((i, j, a), {})

        # Sample empirical transitions
        empirical = self.sample_transitions(i, j, a)

        # Compare
        matches, metrics = self.compare_distributions(analytical, empirical, tolerance)

        if verbose:
            print(f"  Max difference: {metrics['max_diff']:.4f}")
            print(f"  Avg difference: {metrics['avg_diff']:.4f}")
            print(f"  TV distance: {metrics['tv_distance']:.4f}")

            if not matches:
                print(f"  ⚠️  Largest discrepancies:")
                for state, p_a, p_e, diff in metrics["differences"][:3]:
                    print(
                        f"    State {state}: analytical={p_a:.4f}, empirical={p_e:.4f}, diff={diff:.4f}"
                    )

        return matches, metrics


def test_random_states_monte_carlo():
    """Test random states using Monte Carlo validation"""
    print("\n" + "=" * 70)
    print("MONTE CARLO VALIDATION TEST")
    print("=" * 70)

    # Create systems
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

    validator = MonteCarloValidator(continuous_sys, grid_sys, n_samples=10000)

    # Test a few random states
    np.random.seed(42)
    n_tests = 5
    tolerance = 0.05  # 5% tolerance

    print(f"\nTesting {n_tests} random state-action pairs")
    print(f"Monte Carlo samples per test: {validator.n_samples}")
    print(f"Tolerance: {tolerance}")

    results = []
    for test_num in range(n_tests):
        # Random state and action
        i = np.random.randint(0, grid_sys.n_y)
        j = np.random.randint(0, grid_sys.n_v)
        a = np.random.choice([-1, 0, 1])

        print(f"\n--- Test {test_num + 1}/{n_tests} ---")
        matches, metrics = validator.validate_state_action(
            i, j, a, tolerance, verbose=True
        )
        results.append(matches)

        if matches:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")

    # Summary
    print("\n" + "=" * 70)
    print("MONTE CARLO VALIDATION SUMMARY")
    print("=" * 70)
    passed = sum(results)
    print(f"Passed: {passed}/{n_tests}")

    if passed == n_tests:
        print("✅ All Monte Carlo tests passed!")
        print("   Analytical computation matches empirical sampling.")
    else:
        print(f"⚠️  {n_tests - passed} test(s) failed")
        print("   Analytical transitions may have errors.")

    return results


def test_edge_cases_monte_carlo():
    """Test specific edge cases with Monte Carlo"""
    print("\n" + "=" * 70)
    print("MONTE CARLO EDGE CASE TESTS")
    print("=" * 70)

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

    validator = MonteCarloValidator(continuous_sys, grid_sys, n_samples=10000)

    # Test cases
    test_cases = [
        (
            "Zero velocity (middle)",
            grid_sys.n_y // 2,
            np.argmin(np.abs(grid_sys.v_grid)),
            0,
        ),
        ("High velocity", grid_sys.n_y // 2, grid_sys.n_v - 1, 1),
        ("Boundary position", 0, grid_sys.n_v // 2, -1),
        ("Corner state", 0, 0, 1),
    ]

    results = []
    for name, i, j, a in test_cases:
        print(f"\n--- {name} ---")
        matches, metrics = validator.validate_state_action(
            i, j, a, tolerance=0.05, verbose=True
        )
        results.append((name, matches))

        if matches:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")

    # Summary
    print("\n" + "=" * 70)
    print("EDGE CASE SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    passed = sum(r[1] for r in results)
    total = len(results)

    if passed == total:
        print(f"\n✅ All {total} edge cases passed!")
    else:
        print(f"\n⚠️  {total - passed}/{total} edge case(s) failed")

    return results


if __name__ == "__main__":
    print("Note: This test uses random sampling and may take 1-2 minutes...")

    # Run tests
    random_results = test_random_states_monte_carlo()
    edge_results = test_edge_cases_monte_carlo()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Random states: {sum(random_results)}/{len(random_results)} passed")
    print(f"Edge cases: {sum(r[1] for r in edge_results)}/{len(edge_results)} passed")

    all_passed = all(random_results) and all(r[1] for r in edge_results)
    if all_passed:
        print("\n🎉 All Monte Carlo validation tests passed!")
        print("   Analytical implementation is correct!")
    else:
        print("\n⚠️  Some tests failed - review analytical computation")
