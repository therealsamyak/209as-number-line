"""
Test mathematical correctness of Value Iteration.
Verify that the computed policy satisfies the Bellman optimality equation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ContinuousParticleSystem, GridParticleSystem, ValueIteration


class TestBellman:
    """Test Bellman equation satisfaction"""

    def setup_system(self):
        """Create test system with solved value function"""
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

        # Use coarser grid for faster testing
        grid_sys = GridParticleSystem(
            continuous_sys, delta_y=1.0, delta_v=1.0, goal_y=0.0, goal_v=0.0
        )

        # Solve
        vi = ValueIteration(grid_sys, gamma=0.95)
        vi.solve(max_iterations=200, threshold=1e-4)

        return continuous_sys, grid_sys, vi

    def test_bellman_optimality_equation(self):
        """
        Test that V*(s) = max_a Q*(s,a) for all states
        where Q*(s,a) = Σ P(s'|s,a)[R(s,a,s') + γV*(s')]
        """
        print("\n" + "=" * 60)
        print("TEST: Bellman Optimality Equation")
        print("=" * 60)
        print("Verifying: V*(s) = max_a Q*(s,a) for all states")

        _, grid_sys, vi = self.setup_system()

        max_error = 0.0
        errors = []

        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                # Compute Q-values for all actions
                q_values = []
                for a in vi.actions:
                    q = vi._compute_q_value(i, j, a)
                    q_values.append(q)

                # Bellman optimality: V*(s) should equal max_a Q*(s,a)
                v_star = vi.V[i, j]
                max_q = max(q_values)

                error = abs(v_star - max_q)
                max_error = max(max_error, error)

                if error > 1e-3:
                    y, v = grid_sys.continuous_state(i, j)
                    errors.append(
                        {
                            "state": (i, j),
                            "continuous": (y, v),
                            "V": v_star,
                            "max_Q": max_q,
                            "error": error,
                        }
                    )

        print(f"\nResults:")
        print(f"  Max error: {max_error:.6f}")
        print(f"  States with error > 1e-3: {len(errors)}")

        if len(errors) > 0:
            print(f"\n  Largest errors:")
            sorted_errors = sorted(errors, key=lambda x: x["error"], reverse=True)
            for e in sorted_errors[:5]:
                print(
                    f"    State {e['state']}: V={e['V']:.6f}, max_Q={e['max_Q']:.6f}, error={e['error']:.6f}"
                )

        if max_error < 1e-3:
            print("\n✅ PASSED: Bellman optimality equation satisfied (error < 1e-3)")
            return True
        else:
            print("\n❌ FAILED: Bellman optimality equation has significant errors")
            return False

    def test_policy_greedy_wrt_value_function(self):
        """Test that policy π*(s) = argmax_a Q*(s,a)"""
        print("\n" + "=" * 60)
        print("TEST: Policy is Greedy w.r.t. Value Function")
        print("=" * 60)

        _, grid_sys, vi = self.setup_system()

        mismatches = []

        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                # Compute Q-values
                q_values = []
                for a in vi.actions:
                    q = vi._compute_q_value(i, j, a)
                    q_values.append(q)

                # Greedy action
                greedy_action_idx = np.argmax(q_values)

                # Stored policy
                policy_action_idx = vi.policy[i, j]

                if greedy_action_idx != policy_action_idx:
                    y, v = grid_sys.continuous_state(i, j)
                    mismatches.append(
                        {
                            "state": (i, j),
                            "continuous": (y, v),
                            "policy_action": vi.actions[policy_action_idx],
                            "greedy_action": vi.actions[greedy_action_idx],
                            "q_values": q_values,
                        }
                    )

        print(f"\nResults:")
        print(
            f"  Policy-greedy mismatches: {len(mismatches)}/{grid_sys.n_y * grid_sys.n_v}"
        )

        if len(mismatches) > 0:
            print(f"\n  First few mismatches:")
            for m in mismatches[:3]:
                print(
                    f"    State {m['state']}: policy={m['policy_action']}, greedy={m['greedy_action']}"
                )
                print(f"      Q-values: {m['q_values']}")

        if len(mismatches) == 0:
            print("\n✅ PASSED: Policy is greedy with respect to value function")
            return True
        else:
            print("\n❌ FAILED: Policy contains non-greedy actions")
            return False

    def test_value_function_convergence(self):
        """Test that value iteration actually converged"""
        print("\n" + "=" * 60)
        print("TEST: Value Function Convergence")
        print("=" * 60)

        _, grid_sys, vi = self.setup_system()

        # Perform one more Bellman backup
        V_new = np.zeros_like(vi.V)

        for i in range(grid_sys.n_y):
            for j in range(grid_sys.n_v):
                q_values = []
                for a in vi.actions:
                    q = vi._compute_q_value(i, j, a)
                    q_values.append(q)
                V_new[i, j] = max(q_values)

        # Measure difference
        delta = np.max(np.abs(V_new - vi.V))
        avg_delta = np.mean(np.abs(V_new - vi.V))

        print(f"\nResults (one more Bellman backup):")
        print(f"  Max change: {delta:.6f}")
        print(f"  Avg change: {avg_delta:.6f}")

        threshold = 1e-3
        if delta < threshold:
            print(f"\n✅ PASSED: Value function converged (delta < {threshold})")
            return True
        else:
            print(f"\n⚠️  WARNING: Value function may not have fully converged")
            print(f"   Consider more iterations or lower threshold")
            return False

    def test_value_function_bounds(self):
        """Test that value function is bounded correctly"""
        print("\n" + "=" * 60)
        print("TEST: Value Function Bounds")
        print("=" * 60)

        _, grid_sys, vi = self.setup_system()

        # Theoretical bounds
        # Min: all states give 0 reward → V = 0
        # Max: if we always got reward 1, V = 1/(1-γ)
        r_max = 1.0
        v_theoretical_max = r_max / (1 - vi.gamma)

        v_min = np.min(vi.V)
        v_max = np.max(vi.V)

        print(f"\nValue function statistics:")
        print(f"  Min: {v_min:.4f}")
        print(f"  Max: {v_max:.4f}")
        print(f"  Mean: {np.mean(vi.V):.4f}")
        print(f"  Std: {np.std(vi.V):.4f}")
        print(f"\nTheoretical bounds:")
        print(f"  Min (expected): 0.0")
        print(f"  Max (theoretical): {v_theoretical_max:.4f}")

        bounds_ok = (v_min >= -1e-6) and (v_max <= v_theoretical_max + 1e-6)

        if bounds_ok:
            print("\n✅ PASSED: Value function within theoretical bounds")
            return True
        else:
            print("\n❌ FAILED: Value function violates bounds")
            if v_min < -1e-6:
                print(f"  Min value {v_min} is negative!")
            if v_max > v_theoretical_max + 1e-6:
                print(
                    f"  Max value {v_max} exceeds theoretical max {v_theoretical_max}"
                )
            return False

    def test_goal_state_value(self):
        """Test that goal state has high value"""
        print("\n" + "=" * 60)
        print("TEST: Goal State Value")
        print("=" * 60)

        _, grid_sys, vi = self.setup_system()

        # Find goal state
        i_goal, j_goal = grid_sys.discretize(grid_sys.goal_y, grid_sys.goal_v)
        v_goal = vi.V[i_goal, j_goal]

        y_goal, v_goal_actual = grid_sys.continuous_state(i_goal, j_goal)

        print(f"\nGoal state:")
        print(f"  Target: ({grid_sys.goal_y}, {grid_sys.goal_v})")
        print(f"  Grid cell: ({i_goal}, {j_goal})")
        print(f"  Actual: ({y_goal:.2f}, {v_goal_actual:.2f})")
        print(f"  Value: {v_goal:.4f}")

        # Goal state should have high value (at least above average)
        avg_value = np.mean(vi.V)
        print(f"\nAverage value across all states: {avg_value:.4f}")

        # Goal should be significantly above average
        # Check if goal is in top 10% of state values
        value_90th_percentile = np.percentile(vi.V, 90)

        if v_goal >= value_90th_percentile:
            print(
                f"\n✅ PASSED: Goal state has high value ({v_goal:.4f} >= 90th percentile {value_90th_percentile:.4f})"
            )
            return True
        else:
            print(f"\n⚠️  WARNING: Goal state value may be lower than expected")
            print(
                f"   Goal value: {v_goal:.4f}, 90th percentile: {value_90th_percentile:.4f}"
            )
            return False


def run_all_bellman_tests():
    """Run complete Bellman equation test suite"""
    print("\n" + "=" * 70)
    print("BELLMAN EQUATION & VALUE ITERATION TEST SUITE")
    print("=" * 70)
    print("(This may take 1-2 minutes...)")

    tester = TestBellman()

    results = {
        "Bellman Optimality Equation": tester.test_bellman_optimality_equation(),
        "Policy is Greedy": tester.test_policy_greedy_wrt_value_function(),
        "Value Convergence": tester.test_value_function_convergence(),
        "Value Function Bounds": tester.test_value_function_bounds(),
        "Goal State Value": tester.test_goal_state_value(),
    }

    print("\n" + "=" * 70)
    print("BELLMAN TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Bellman tests passed!")
        print("   Value iteration is mathematically correct!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review value iteration")

    return results


if __name__ == "__main__":
    run_all_bellman_tests()
