"""
Comprehensive test for voting system data flow after fixing ownership conflicts.
Tests that EnhancedVotingCommitteeCoordinator properly consumes individual voting proposals
and produces aggregated outputs without violating InfoBus ownership rules.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.contracts import CONTRACTS

def test_voting_data_flow_contracts():
    """Test that voting contracts follow proper data flow architecture."""
    print("=== TESTING VOTING DATA FLOW CONTRACTS ===")

    contracts = CONTRACTS

    # Get coordinator contract
    coordinator_contract = contracts.get('EnhancedVotingCommitteeCoordinator')
    if not coordinator_contract:
        print("FAIL: EnhancedVotingCommitteeCoordinator contract not found")
        return False

    print(f"Coordinator provides: {coordinator_contract.provides}")
    print(f"Coordinator requires: {coordinator_contract.requires}")

    # Check that coordinator does NOT provide member_confidences (ownership violation)
    if 'member_confidences' in coordinator_contract.provides:
        print("FAIL: EnhancedVotingCommitteeCoordinator should not provide 'member_confidences'")
        print("      This key is owned by EnhancedThemeExpert")
        return False

    # Check that coordinator DOES provide member_confidences_ordered
    if 'member_confidences_ordered' not in coordinator_contract.provides:
        print("FAIL: EnhancedVotingCommitteeCoordinator should provide 'member_confidences_ordered'")
        return False

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

        if proposal_key not in coordinator_contract.requires:
            missing_requirements.append(proposal_key)
        if confidence_key not in coordinator_contract.requires:
            missing_requirements.append(confidence_key)

    if missing_requirements:
        print(f"FAIL: Coordinator missing requirements: {missing_requirements}")
        return False

    print("PASS: Coordinator contract follows proper data flow architecture")

    # Check voting members provide their individual keys
    print("\n=== CHECKING VOTING MEMBERS ===")
    voting_member_issues = []

    for member_name in voting_members:
        contract = contracts.get(member_name)
        if not contract:
            voting_member_issues.append(f"Missing contract for {member_name}")
            continue

        expected_proposal = f"{member_name}_voting_proposal"
        expected_confidence = f"{member_name}_confidence"

        if expected_proposal not in contract.provides:
            voting_member_issues.append(f"{member_name} should provide {expected_proposal}")
        if expected_confidence not in contract.provides:
            voting_member_issues.append(f"{member_name} should provide {expected_confidence}")

    if voting_member_issues:
        print(f"FAIL: Voting member issues:")
        for issue in voting_member_issues:
            print(f"      {issue}")
        return False

    print(f"PASS: All {len(voting_members)} voting members provide required keys")
    return True

def test_voting_architecture():
    """Test the overall voting system architecture."""
    print("\n=== TESTING VOTING ARCHITECTURE ===")

    contracts = CONTRACTS

    # Check that there's no duplicate ownership conflicts
    member_confidences_providers = []
    member_confidences_ordered_providers = []

    for name, contract in contracts.items():
        if 'member_confidences' in contract.provides:
            member_confidences_providers.append(name)
        if 'member_confidences_ordered' in contract.provides:
            member_confidences_ordered_providers.append(name)

    print(f"Modules providing 'member_confidences': {member_confidences_providers}")
    print(f"Modules providing 'member_confidences_ordered': {member_confidences_ordered_providers}")

    # Only EnhancedThemeExpert should provide member_confidences (as individual expert)
    if len(member_confidences_providers) > 1:
        print(f"FAIL: Multiple providers for 'member_confidences': {member_confidences_providers}")
        print("      This will cause InfoBus ownership conflicts")
        return False

    # Only EnhancedVotingCommitteeCoordinator should provide member_confidences_ordered
    if 'EnhancedVotingCommitteeCoordinator' not in member_confidences_ordered_providers:
        print("FAIL: EnhancedVotingCommitteeCoordinator should be the provider of 'member_confidences_ordered'")
        return False

    if len(member_confidences_ordered_providers) > 1:
        print(f"FAIL: Multiple providers for 'member_confidences_ordered': {member_confidences_ordered_providers}")
        return False

    print("PASS: No ownership conflicts in voting architecture")
    return True

def main():
    """Run all voting data flow tests."""
    print("VOTING SYSTEM DATA FLOW TEST")
    print("="*50)

    results = []
    results.append(test_voting_data_flow_contracts())
    results.append(test_voting_architecture())

    all_passed = all(results)
    print(f"\n{'='*50}")
    print(f"Overall Result: {'PASS' if all_passed else 'FAIL'}")

    if all_passed:
        print("PASS: Voting system data flow architecture is correct")
        print("PASS: No InfoBus ownership conflicts")
        print("PASS: Coordinator properly consumes individual proposals")
        print("PASS: All voting members provide required keys")
    else:
        print("FAIL: Voting system has architectural issues that need fixing")

    return all_passed

if __name__ == "__main__":
    main()