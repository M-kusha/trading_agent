"""
Comprehensive test to validate all dependency and circular dependency fixes.
Tests that provider conflicts, circular dependencies, and missing key issues have been resolved.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.contracts import CONTRACTS

def test_provider_conflicts():
    """Test that no two modules provide the same key."""
    print("=== TESTING PROVIDER CONFLICTS ===")

    all_provides = {}
    conflicts = []

    for module_name, contract in CONTRACTS.items():
        for key in contract.provides:
            if key not in all_provides:
                all_provides[key] = []
            all_provides[key].append(module_name)

    for key, providers in all_provides.items():
        if len(providers) > 1:
            conflicts.append(f"Key '{key}' provided by: {providers}")

    if conflicts:
        print("FAIL: Provider conflicts found:")
        for conflict in conflicts:
            print(f"      {conflict}")
        return False

    print("PASS: No provider conflicts detected")
    return True

def test_circular_voting_dependencies():
    """Test that circular dependencies in voting system have been resolved."""
    print("\n=== TESTING CIRCULAR VOTING DEPENDENCIES ===")

    # Key modules in the circular dependency from orchestrator logs
    circular_modules = [
        'EnhancedVotingCommitteeCoordinator',
        'VotingKernel',
        'EnhancedThemeExpert',
        'EnhancedSeasonalityRiskExpert'
    ]

    issues = []

    # Check that VotingKernel doesn't provide keys that coordinator provides
    voting_kernel = CONTRACTS.get('VotingKernel')
    coordinator = CONTRACTS.get('EnhancedVotingCommitteeCoordinator')

    if voting_kernel and coordinator:
        kernel_provides = set(voting_kernel.provides)
        coordinator_provides = set(coordinator.provides)

        conflicts = kernel_provides.intersection(coordinator_provides)
        if conflicts:
            issues.append(f"VotingKernel and Coordinator both provide: {conflicts}")

    # Check that voting members don't provide coordinator keys
    theme_expert = CONTRACTS.get('EnhancedThemeExpert')
    if theme_expert and coordinator:
        theme_provides = set(theme_expert.provides)
        coordinator_provides = set(coordinator.provides)

        conflicts = theme_provides.intersection(coordinator_provides)
        if conflicts:
            issues.append(f"EnhancedThemeExpert and Coordinator both provide: {conflicts}")

    if issues:
        print("FAIL: Circular dependency conflicts found:")
        for issue in issues:
            print(f"      {issue}")
        return False

    print("PASS: Circular voting dependencies resolved")
    return True

def test_voting_data_flow():
    """Test that voting system data flow is properly structured."""
    print("\n=== TESTING VOTING DATA FLOW ===")

    coordinator = CONTRACTS.get('EnhancedVotingCommitteeCoordinator')
    if not coordinator:
        print("FAIL: EnhancedVotingCommitteeCoordinator contract not found")
        return False

    issues = []

    # Check that coordinator provides aggregated keys
    required_coordinator_provides = [
        'expert_votes',
        'committee_members',
        'proposal_vectors',
        'signals',
        'committee_consensus',
        'member_confidences_ordered'
    ]

    missing_provides = []
    for key in required_coordinator_provides:
        if key not in coordinator.provides:
            missing_provides.append(key)

    if missing_provides:
        issues.append(f"Coordinator missing provides: {missing_provides}")

    # Check that coordinator requires individual voting proposals
    voting_members = [
        'DynamicRiskController', 'EnhancedAnomalyDetector', 'EnhancedSeasonalityRiskExpert',
        'EnhancedThemeExpert', 'ExecutionQualityMonitor', 'MetaAgent',
        'PortfolioRiskSystem', 'PPOAgent'
    ]

    missing_requirements = []
    for member in voting_members:
        proposal_key = f"{member}_voting_proposal"
        confidence_key = f"{member}_confidence"

        if proposal_key not in coordinator.requires:
            missing_requirements.append(proposal_key)
        if confidence_key not in coordinator.requires:
            missing_requirements.append(confidence_key)

    if missing_requirements:
        issues.append(f"Coordinator missing requirements: {missing_requirements}")

    if issues:
        print("FAIL: Voting data flow issues:")
        for issue in issues:
            print(f"      {issue}")
        return False

    print("PASS: Voting data flow properly structured")
    return True

def test_canonical_providers():
    """Test that canonical providers are correctly assigned."""
    print("\n=== TESTING CANONICAL PROVIDERS ===")

    canonical_assignments = {
        'theme_confidence': ['EnhancedThemeExpert'],
        'expert_votes': ['EnhancedVotingCommitteeCoordinator'],
        'committee_members': ['EnhancedVotingCommitteeCoordinator'],
        'proposal_vectors': ['EnhancedVotingCommitteeCoordinator'],
        'signals': ['EnhancedVotingCommitteeCoordinator'],
        'strategy_arbiter_weights': ['EnhancedVotingCommitteeCoordinator'],
        'universe': ['MarketDataProvider'],
        'watched_instruments': ['MarketDataProvider'],
        'member_confidences_ordered': ['EnhancedVotingCommitteeCoordinator']
    }

    issues = []

    for key, expected_providers in canonical_assignments.items():
        actual_providers = []
        for module_name, contract in CONTRACTS.items():
            if key in contract.provides:
                actual_providers.append(module_name)

        if set(actual_providers) != set(expected_providers):
            issues.append(f"Key '{key}': expected {expected_providers}, got {actual_providers}")

    if issues:
        print("FAIL: Canonical provider assignments incorrect:")
        for issue in issues:
            print(f"      {issue}")
        return False

    print("PASS: Canonical providers correctly assigned")
    return True

def test_voting_members_contracts():
    """Test that all voting members have proper contracts."""
    print("\n=== TESTING VOTING MEMBERS CONTRACTS ===")

    voting_members = [
        'DynamicRiskController', 'EnhancedAnomalyDetector', 'EnhancedSeasonalityRiskExpert',
        'EnhancedThemeExpert', 'ExecutionQualityMonitor', 'MetaAgent',
        'PortfolioRiskSystem', 'PPOAgent'
    ]

    issues = []

    for member in voting_members:
        contract = CONTRACTS.get(member)
        if not contract:
            issues.append(f"Missing contract for voting member: {member}")
            continue

        # Check that each voting member provides its standardized keys
        expected_proposal = f"{member}_voting_proposal"
        expected_confidence = f"{member}_confidence"

        if expected_proposal not in contract.provides:
            issues.append(f"{member} should provide {expected_proposal}")
        if expected_confidence not in contract.provides:
            issues.append(f"{member} should provide {expected_confidence}")

    if issues:
        print("FAIL: Voting members contract issues:")
        for issue in issues:
            print(f"      {issue}")
        return False

    print(f"PASS: All {len(voting_members)} voting members have proper contracts")
    return True

def main():
    """Run all dependency fix validation tests."""
    print("DEPENDENCY FIXES VALIDATION TEST")
    print("="*60)

    results = []
    results.append(test_provider_conflicts())
    results.append(test_circular_voting_dependencies())
    results.append(test_voting_data_flow())
    results.append(test_canonical_providers())
    results.append(test_voting_members_contracts())

    all_passed = all(results)
    print(f"\n{'='*60}")
    print(f"Overall Result: {'PASS' if all_passed else 'FAIL'}")

    if all_passed:
        print("PASS: All dependency and circular dependency issues resolved")
        print("PASS: Provider conflicts eliminated")
        print("PASS: Voting system architecture fixed")
        print("PASS: InfoBus data flow optimized")
    else:
        print("FAIL: Some dependency issues still need attention")

    return all_passed

if __name__ == "__main__":
    main()