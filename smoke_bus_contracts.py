# #!/usr/bin/env python3
# import sys
# import time

# # Ensure local repo is on path
# import os
# from pathlib import Path
# ROOT = Path(__file__).resolve().parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from modules.utils.info_bus import InfoBusManager

# # Target modules
# from modules.risk.correlated_risk_controller import CorrelatedRiskController
# from modules.voting.voting_wrappers import (
#     EnhancedThemeExpert,
#     EnhancedSeasonalityRiskExpert,
#     EnhancedVotingCommitteeCoordinator,
# )
# from modules.position.position_1 import PositionManager


# def ok(label: str):
#     print(f"[OK] {label}")

# def fail(label: str, err: Exception | str):
#     print(f"[FAIL] {label}: {err}")


# def main():
#     bus = InfoBusManager.get_instance()

#     # Instantiate providers (their _initialize should publish baselines)
#     try:
#         risk_ctrl = CorrelatedRiskController(config={})
#         ok("CorrelatedRiskController instantiated")
#     except Exception as e:
#         fail("CorrelatedRiskController init", e)
#         return 2

#     try:
#         theme_expert = EnhancedThemeExpert(config={})
#         ok("EnhancedThemeExpert instantiated")
#     except Exception as e:
#         fail("EnhancedThemeExpert init", e)
#         return 2

#     try:
#         season_expert = EnhancedSeasonalityRiskExpert(config={})
#         ok("EnhancedSeasonalityRiskExpert instantiated")
#     except Exception as e:
#         fail("EnhancedSeasonalityRiskExpert init", e)
#         return 2

#     try:
#         committee = EnhancedVotingCommitteeCoordinator(config={})
#         ok("EnhancedVotingCommitteeCoordinator instantiated")
#     except Exception as e:
#         fail("EnhancedVotingCommitteeCoordinator init", e)
#         return 2

#     # PositionManager isn't required for baseline checks, but instantiate to ensure consumers don't crash
#     try:
#         pm = PositionManager(config={}, instruments=["EUR/USD", "XAU/USD"])  # noqa: F841
#         ok("PositionManager instantiated")
#     except Exception as e:
#         fail("PositionManager init", e)
#         return 2

#     # Allow any async health monitors to tick once (non-critical)
#     time.sleep(0.5)

#     # Validate baseline keys exist on the bus
#     exit_code = 0

#     try:
#         corr = bus.get("correlation_matrix", "smoke")
#         assert isinstance(corr, dict), f"correlation_matrix must be dict, got {type(corr)}"
#         ok("correlation_matrix present (dict)")
#     except Exception as e:
#         fail("correlation_matrix missing/invalid", e)
#         exit_code = 1

#     for expert in ["EnhancedThemeExpert", "EnhancedSeasonalityRiskExpert"]:
#         try:
#             proposal = bus.get(f"{expert}_voting_proposal", "smoke")
#             confidence = bus.get(f"{expert}_confidence", "smoke")
#             assert isinstance(proposal, dict), f"{expert}_voting_proposal must be dict"
#             assert isinstance(confidence, (int, float)), f"{expert}_confidence must be number"
#             ok(f"{expert} baseline provides present")
#         except Exception as e:
#             fail(f"{expert} baseline provides missing/invalid", e)
#             exit_code = 1

#     try:
#         votes = bus.get("committee_votes", "smoke")
#         assert isinstance(votes, list), f"committee_votes must be list, got {type(votes)}"
#         ok("committee_votes present (list)")
#     except Exception as e:
#         fail("committee_votes missing/invalid", e)
#         exit_code = 1

#     if exit_code == 0:
#         print("\nAll baseline contract keys present. ✅")
#     else:
#         print("\nSome baseline keys missing. ❌")

#     return exit_code


# if __name__ == "__main__":
#     raise SystemExit(main())
