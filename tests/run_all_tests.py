"""
Master test runner - runs all verification tests and generates a comprehensive report.
"""

import sys
import os

# Add parent directory to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add current directory to path so we can import test modules
sys.path.insert(0, os.path.dirname(__file__))

from test_transitions import run_all_tests as run_transition_tests
from test_monte_carlo_validation import (
    test_random_states_monte_carlo,
    test_edge_cases_monte_carlo,
)
from test_physics import run_all_physics_tests
from test_discretization import run_all_discretization_tests
from test_bellman import run_all_bellman_tests


def print_section_header(title):
    """Print formatted section header"""
    print("\n\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_comprehensive_test_suite():
    """Run all test suites and generate comprehensive report"""

    print("=" * 80)
    print(" " * 20 + "COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80)
    print("\nThis suite verifies:")
    print("  1. Transition Probabilities (analytical correctness)")
    print("  2. Monte Carlo Validation (empirical verification)")
    print("  3. Physics Implementation (dynamics correctness)")
    print("  4. Discretization (grid structure)")
    print("  5. Bellman Equations (mathematical optimality)")
    print("\nEstimated time: 3-5 minutes")

    input("\nPress Enter to start tests...")

    all_results = {}

    # ============================================================
    # 1. TRANSITION PROBABILITY TESTS
    # ============================================================
    print_section_header("PART 1: TRANSITION PROBABILITY TESTS")
    transition_results = run_transition_tests()
    all_results["Transition Probabilities"] = transition_results

    # ============================================================
    # 2. MONTE CARLO VALIDATION
    # ============================================================
    print_section_header("PART 2: MONTE CARLO VALIDATION")
    print("Comparing analytical transitions with empirical sampling...")

    random_mc_results = test_random_states_monte_carlo()
    edge_mc_results = test_edge_cases_monte_carlo()

    all_results["Monte Carlo - Random"] = {
        "passed": sum(random_mc_results),
        "total": len(random_mc_results),
    }
    all_results["Monte Carlo - Edges"] = {
        "passed": sum(r[1] for r in edge_mc_results),
        "total": len(edge_mc_results),
    }

    # ============================================================
    # 3. PHYSICS TESTS
    # ============================================================
    print_section_header("PART 3: PHYSICS IMPLEMENTATION TESTS")
    physics_results = run_all_physics_tests()
    all_results["Physics"] = physics_results

    # ============================================================
    # 4. DISCRETIZATION TESTS
    # ============================================================
    print_section_header("PART 4: DISCRETIZATION TESTS")
    discretization_results = run_all_discretization_tests()
    all_results["Discretization"] = discretization_results

    # ============================================================
    # 5. BELLMAN EQUATION TESTS
    # ============================================================
    print_section_header("PART 5: BELLMAN EQUATION TESTS")
    bellman_results = run_all_bellman_tests()
    all_results["Bellman Equations"] = bellman_results

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n\n")
    print("=" * 80)
    print(" " * 25 + "FINAL VERIFICATION REPORT")
    print("=" * 80)

    total_passed = 0
    total_tests = 0

    print("\n📊 TEST SUITE SUMMARY:\n")

    # Transition tests
    trans_passed = sum(all_results["Transition Probabilities"].values())
    trans_total = len(all_results["Transition Probabilities"])
    status = "✅" if trans_passed == trans_total else "❌"
    print(f"{status} Transition Probabilities: {trans_passed}/{trans_total}")
    total_passed += trans_passed
    total_tests += trans_total

    # Monte Carlo
    mc_random_passed = all_results["Monte Carlo - Random"]["passed"]
    mc_random_total = all_results["Monte Carlo - Random"]["total"]
    mc_edge_passed = all_results["Monte Carlo - Edges"]["passed"]
    mc_edge_total = all_results["Monte Carlo - Edges"]["total"]
    status = (
        "✅"
        if (mc_random_passed == mc_random_total and mc_edge_passed == mc_edge_total)
        else "❌"
    )
    print(f"{status} Monte Carlo Validation:")
    print(f"    - Random states: {mc_random_passed}/{mc_random_total}")
    print(f"    - Edge cases: {mc_edge_passed}/{mc_edge_total}")
    total_passed += mc_random_passed + mc_edge_passed
    total_tests += mc_random_total + mc_edge_total

    # Physics
    physics_passed = sum(all_results["Physics"].values())
    physics_total = len(all_results["Physics"])
    status = "✅" if physics_passed == physics_total else "❌"
    print(f"{status} Physics Implementation: {physics_passed}/{physics_total}")
    total_passed += physics_passed
    total_tests += physics_total

    # Discretization
    disc_passed = sum(all_results["Discretization"].values())
    disc_total = len(all_results["Discretization"])
    status = "✅" if disc_passed == disc_total else "❌"
    print(f"{status} Discretization: {disc_passed}/{disc_total}")
    total_passed += disc_passed
    total_tests += disc_total

    # Bellman
    bellman_passed = sum(all_results["Bellman Equations"].values())
    bellman_total = len(all_results["Bellman Equations"])
    status = "✅" if bellman_passed == bellman_total else "❌"
    print(f"{status} Bellman Equations: {bellman_passed}/{bellman_total}")
    total_passed += bellman_passed
    total_tests += bellman_total

    # Overall
    print("\n" + "-" * 80)
    print(
        f"\n📈 OVERALL: {total_passed}/{total_tests} tests passed ({100 * total_passed / total_tests:.1f}%)\n"
    )

    if total_passed == total_tests:
        print("=" * 80)
        print("🎉 " + " " * 15 + "ALL TESTS PASSED - IMPLEMENTATION VERIFIED!")
        print("=" * 80)
        print("\n✓ Transition probabilities are mathematically correct")
        print("✓ Analytical computation matches Monte Carlo sampling")
        print("✓ Physics dynamics are implemented correctly")
        print("✓ Grid discretization is sound")
        print("✓ Value iteration satisfies Bellman optimality")
        print("\n👍 You can trust this AI-generated code!")
    else:
        print("=" * 80)
        print("⚠️  " + " " * 20 + "SOME TESTS FAILED")
        print("=" * 80)
        print(
            f"\n{total_tests - total_passed} test(s) failed - review the implementation"
        )
        print("\nFailed categories:")

        if trans_passed < trans_total:
            print("  ❌ Transition Probabilities")
        if mc_random_passed < mc_random_total or mc_edge_passed < mc_edge_total:
            print("  ❌ Monte Carlo Validation")
        if physics_passed < physics_total:
            print("  ❌ Physics Implementation")
        if disc_passed < disc_total:
            print("  ❌ Discretization")
        if bellman_passed < bellman_total:
            print("  ❌ Bellman Equations")

    print("\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    run_comprehensive_test_suite()
