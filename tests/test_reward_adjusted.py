# test_reward_adjusted.py

from modules.reward.risk_adjusted_reward import RiskAdjustedReward

def test_reward_module():
    print("Running built-in test for RiskAdjustedReward...\n")
    reward_module = RiskAdjustedReward(initial_balance=10000.0, debug=True)
    result = reward_module.test_reward_calculation()
    print("\nTest finished. Results (reward1, reward2, reward3):", result)
    # Optionally, assert types or value ranges if you want:
    assert isinstance(result, tuple), "Result should be a tuple"
    assert all(isinstance(x, float) for x in result), "Each test reward should be a float"

if __name__ == "__main__":
    test_reward_module()
