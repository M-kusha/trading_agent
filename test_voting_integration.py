#!/usr/bin/env python3
"""
Test voting proposal integration after fixes
"""
import sys
import os
sys.path.insert(0, os.getcwd())

def test_voting_coordinator_requirements():
    print("=== Testing Voting Coordinator Requirements ===")
    try:
        from modules.contracts import CONTRACTS

        evcc_contract = CONTRACTS.get("EnhancedVotingCommitteeCoordinator")
        if not evcc_contract:
            print("FAIL: EnhancedVotingCommitteeCoordinator contract not found")
            return False

        requires = evcc_contract.requires
        print(f"Total requirements: {len(requires)}")

        # Check for voting proposals
        voting_proposals = [req for req in requires if "_voting_proposal" in req]
        confidences = [req for req in requires if req.endswith("_confidence")]

        print(f"Voting proposals required: {len(voting_proposals)}")
        print(f"Confidence scores required: {len(confidences)}")

        # Expected voting members
        expected_voters = [
            "DynamicRiskController",
            "EnhancedAnomalyDetector", 
            "EnhancedSeasonalityRiskExpert",
            "EnhancedThemeExpert",
            "ExecutionQualityMonitor",
            "MetaAgent",
            "PortfolioRiskSystem",
            "PPOAgent"
        ]

        missing_voters = []
        for voter in expected_voters:
            proposal_key = f"{voter}_voting_proposal"
            confidence_key = f"{voter}_confidence"

            if proposal_key not in requires:
                missing_voters.append(f"{voter} proposal")
            if confidence_key not in requires:
                missing_voters.append(f"{voter} confidence")

        if missing_voters:
            print(f"FAIL: Missing requirements: {missing_voters}")
            return False
        else:
            print("PASS: All voting proposals properly required")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    print("VOTING INTEGRATION TEST")
    print("="*40)
    result = test_voting_coordinator_requirements()
    print(f"\nResult: {'PASS' if result else 'FAIL'}")

