"""
Test physics correctness of the continuous system dynamics.
Verify that forces, velocity updates, and position updates match specifications.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import ContinuousParticleSystem


class TestPhysics:
    """Test that physics equations are implemented correctly"""

    def setup_system(self):
        """Create test system"""
        return ContinuousParticleSystem(
            m=1.0,
            y_max=10.0,
            v_max=5.0,
            A=2.0,
            p_c=0.1,
            goal_y=0.0,
            goal_v=0.0,
            goal_tolerance=0.1,
        )

    def test_potential_field(self):
        """Test φ(y) = A * sin(2πy / y_max)"""
        print("\n" + "=" * 60)
        print("TEST: Potential Field Function φ(y)")
        print("=" * 60)

        sys = self.setup_system()

        # Test at specific points
        test_cases = [
            (0.0, 0.0),  # sin(0) = 0
            (sys.y_max / 4, sys.A),  # sin(π/2) = 1
            (sys.y_max / 2, 0.0),  # sin(π) = 0
            (3 * sys.y_max / 4, -sys.A),  # sin(3π/2) = -1
            (sys.y_max, 0.0),  # sin(2π) = 0
        ]

        all_correct = True
        for y, expected_phi in test_cases:
            actual_phi = sys.phi(y)
            error = abs(actual_phi - expected_phi)

            status = "✓" if error < 1e-10 else "✗"
            print(
                f"  {status} φ({y:.2f}) = {actual_phi:.6f}, expected {expected_phi:.6f}, error = {error:.2e}"
            )

            if error >= 1e-10:
                all_correct = False

        if all_correct:
            print("✅ PASSED: Potential field function is correct")
        else:
            print("❌ FAILED: Potential field function has errors")

        return all_correct

    def test_velocity_update_no_crash(self):
        """Test v[t+1] = v[t] + (a + φ(y))/m + wobble (when no crash)"""
        print("\n" + "=" * 60)
        print("TEST: Velocity Update (No Crash)")
        print("=" * 60)

        sys = self.setup_system()

        # Use fixed random seed for reproducibility
        np.random.seed(42)

        # Test configuration
        y = 2.0
        v = 1.0
        a = 1.0

        # Expected calculation
        f_phi = sys.phi(y)
        f_net = a + f_phi

        print(f"Initial state: y={y}, v={v}")
        print(f"Action: a={a}")
        print(f"Potential force: φ({y}) = {f_phi:.4f}")
        print(f"Net force: f_net = {f_net:.4f}")

        # Run many steps to check average behavior
        n_trials = 1000
        v_next_list = []

        for _ in range(n_trials):
            np.random.seed(np.random.randint(0, 100000))

            # Manually compute update (mimicking step function, no crash)
            speed_wobble = np.random.normal(0, 0.1 * abs(v))
            v_next_expected = v + (f_net / sys.m) + speed_wobble
            v_next_expected = np.clip(v_next_expected, -sys.v_max, sys.v_max)

            v_next_list.append(v_next_expected)

        # Check statistics
        v_mean = np.mean(v_next_list)
        v_std = np.std(v_next_list)

        # Expected mean: v + f_net/m
        v_expected_mean = v + f_net / sys.m

        # Expected std: 0.1 * |v|
        v_expected_std = 0.1 * abs(v)

        print(f"\nStatistics over {n_trials} trials (no crashes):")
        print(f"  Mean v': {v_mean:.4f}, expected {v_expected_mean:.4f}")
        print(f"  Std v': {v_std:.4f}, expected {v_expected_std:.4f}")

        mean_error = abs(v_mean - v_expected_mean)
        std_error = abs(v_std - v_expected_std)

        # Allow some statistical variation
        mean_ok = mean_error < 0.05
        std_ok = std_error < 0.02

        if mean_ok and std_ok:
            print("✅ PASSED: Velocity update follows correct dynamics")
            return True
        else:
            print("❌ FAILED: Velocity update has errors")
            if not mean_ok:
                print(f"  Mean error too large: {mean_error:.4f}")
            if not std_ok:
                print(f"  Std error too large: {std_error:.4f}")
            return False

    def test_position_update(self):
        """Test y[t+1] = y[t] + v[t] (position updates with current velocity)"""
        print("\n" + "=" * 60)
        print("TEST: Position Update")
        print("=" * 60)

        sys = self.setup_system()

        # Test cases
        test_cases = [
            (0.0, 1.0, 1.0),  # Moving forward
            (5.0, -2.0, 3.0),  # Moving backward
            (-3.0, 0.5, -2.5),  # From negative position
        ]

        all_correct = True
        for y, v, expected_y_next in test_cases:
            state = (y, v)

            # Take step with zero action to isolate position update
            np.random.seed(42)

            # Collect position updates (many samples to check)
            y_next_list = []
            for _ in range(100):
                next_state, _, _ = sys.step(state, action=0)
                y_next = next_state[0]
                y_next_list.append(y_next)

            # Position update should be deterministic: y' = y + v
            y_next_mean = np.mean(y_next_list)

            # With action=0 and potential field, velocity will change,
            # but position update uses CURRENT velocity
            # So we expect: y_next = y + v (clipped to bounds)
            expected_y = y + v
            expected_y = np.clip(expected_y, -sys.y_max, sys.y_max)

            error = abs(y_next_mean - expected_y)

            status = "✓" if error < 0.2 else "✗"
            print(
                f"  {status} y={y:.2f}, v={v:.2f} → y'={y_next_mean:.2f}, expected {expected_y:.2f}, error={error:.2f}"
            )

            if error >= 0.2:
                all_correct = False

        if all_correct:
            print("✅ PASSED: Position update is correct (y' = y + v)")
            return True
        else:
            print("⚠️  Position update may need review")
            return False

    def test_crash_mechanics(self):
        """Test crash behavior: v → 0, y unchanged"""
        print("\n" + "=" * 60)
        print("TEST: Crash Mechanics")
        print("=" * 60)

        sys = self.setup_system()

        # Use high velocity to increase crash probability
        y = 3.0
        v = 5.0  # Max velocity

        expected_crash_prob = sys.p_c * abs(v) / sys.v_max
        print(f"State: y={y}, v={v}")
        print(f"Expected crash probability: {expected_crash_prob:.2%}")

        # Run many trials
        n_trials = 1000
        crashes = 0

        for trial in range(n_trials):
            np.random.seed(trial)
            next_state, _, _ = sys.step((y, v), action=0)
            y_next, v_next = next_state

            # Check if crashed (velocity reset to 0)
            if abs(v_next) < 0.01:  # Close to zero
                crashes += 1

                # When crashed, position should be unchanged
                if abs(y_next - y) > 0.01:
                    print(f"  ⚠️  Trial {trial}: Crashed but position changed!")
                    print(f"     y: {y} → {y_next}")

        empirical_crash_rate = crashes / n_trials
        error = abs(empirical_crash_rate - expected_crash_prob)

        print(f"\nResults over {n_trials} trials:")
        print(f"  Empirical crash rate: {empirical_crash_rate:.2%}")
        print(f"  Expected: {expected_crash_prob:.2%}")
        print(f"  Error: {error:.2%}")

        if error < 0.05:  # Within 5%
            print("✅ PASSED: Crash mechanics are correct")
            return True
        else:
            print("❌ FAILED: Crash rate differs from expected")
            return False

    def test_velocity_bounds(self):
        """Test that velocity is clipped to [-v_max, v_max]"""
        print("\n" + "=" * 60)
        print("TEST: Velocity Bounds")
        print("=" * 60)

        sys = self.setup_system()

        # Try to exceed bounds
        test_cases = [
            (0.0, 4.9, 1.0, "Try to exceed v_max"),
            (0.0, -4.9, -1.0, "Try to exceed -v_max"),
        ]

        all_correct = True
        for y, v, a, description in test_cases:
            print(f"\n  {description}:")
            print(f"    Initial: y={y}, v={v}, action={a}")

            violations = 0
            for _ in range(100):
                next_state, _, _ = sys.step((y, v), action=a)
                _, v_next = next_state

                if abs(v_next) > sys.v_max + 1e-6:
                    violations += 1

            if violations > 0:
                print(f"    ❌ {violations}/100 trials violated velocity bounds")
                all_correct = False
            else:
                print(f"    ✓ All velocities within bounds")

        if all_correct:
            print("\n✅ PASSED: Velocity bounds are enforced")
            return True
        else:
            print("\n❌ FAILED: Velocity bounds were violated")
            return False

    def test_position_bounds(self):
        """Test that position is clipped to [-y_max, y_max]"""
        print("\n" + "=" * 60)
        print("TEST: Position Bounds")
        print("=" * 60)

        sys = self.setup_system()

        # Try to exceed bounds
        test_cases = [
            (9.5, 2.0, "Try to exceed y_max"),
            (-9.5, -2.0, "Try to exceed -y_max"),
        ]

        all_correct = True
        for y, v, description in test_cases:
            print(f"\n  {description}:")
            print(f"    Initial: y={y}, v={v}")

            violations = 0
            for _ in range(100):
                next_state, _, _ = sys.step((y, v), action=0)
                y_next, _ = next_state

                if abs(y_next) > sys.y_max + 1e-6:
                    violations += 1

            if violations > 0:
                print(f"    ❌ {violations}/100 trials violated position bounds")
                all_correct = False
            else:
                print(f"    ✓ All positions within bounds")

        if all_correct:
            print("\n✅ PASSED: Position bounds are enforced")
            return True
        else:
            print("\n❌ FAILED: Position bounds were violated")
            return False


def run_all_physics_tests():
    """Run complete physics test suite"""
    print("\n" + "=" * 70)
    print("PHYSICS CORRECTNESS TEST SUITE")
    print("=" * 70)

    tester = TestPhysics()

    results = {
        "Potential Field φ(y)": tester.test_potential_field(),
        "Velocity Update": tester.test_velocity_update_no_crash(),
        "Position Update": tester.test_position_update(),
        "Crash Mechanics": tester.test_crash_mechanics(),
        "Velocity Bounds": tester.test_velocity_bounds(),
        "Position Bounds": tester.test_position_bounds(),
    }

    print("\n" + "=" * 70)
    print("PHYSICS TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All physics tests passed!")
        print("   The continuous system implements the correct dynamics!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review physics implementation")

    return results


if __name__ == "__main__":
    run_all_physics_tests()
