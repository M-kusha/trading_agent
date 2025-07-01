# ─────────────────────────────────────────────────────────────
# File: modules/memory/historical_replay_analyzer.py
# ─────────────────────────────────────────────────────────────

import logging
import os
from typing import List, Optional
import numpy as np
import random
from collections import deque
from modules.core.core import Module

class HistoricalReplayAnalyzer(Module):

    def __init__(self, interval: int=10, bonus: float=0.1, sequence_len: int=5, debug=False):
        self.interval = interval
        self.bonus = bonus
        self.sequence_len = sequence_len
        self.debug = debug
        
        # Enhanced Logger Setup - FIXED
        log_dir = os.path.join("logs", "memory")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"HistoricalReplayAnalyzer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "replay_analyzer.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"HistoricalReplayAnalyzer initialized - interval={interval}, bonus={bonus}, sequence_len={sequence_len}")
        
        self.reset()

    def reset(self):
        # FIX: Store successful sequences
        self.episode_buffer = deque(maxlen=100)
        self.profitable_sequences = []
        self.sequence_patterns = {}  # pattern -> (count, avg_pnl)
        self.current_sequence = []
        self.replay_bonus = 0.0
        self.best_sequence_pnl = 0.0
        self._step_count = 0
        self._episode_count = 0
        
        self.logger.info("HistoricalReplayAnalyzer reset - all sequences and patterns cleared")

    def step(self, **kwargs):
        """Track current trading sequence"""
        self._step_count += 1
        
        try:
            if "action" in kwargs and "features" in kwargs:
                step_data = {
                    "action": kwargs["action"],
                    "features": kwargs["features"],
                    "timestamp": kwargs.get("timestamp", self._step_count)
                }
                self.current_sequence.append(step_data)
                
                # Keep sequence bounded
                if len(self.current_sequence) > self.sequence_len:
                    removed = self.current_sequence.pop(0)
                    self.logger.debug(f"Sequence bounded: removed step {removed.get('timestamp', 'unknown')}")
                
                self.logger.debug(f"Step {self._step_count}: Added to sequence (length={len(self.current_sequence)})")
            else:
                self.logger.debug(f"Step {self._step_count}: Insufficient data for sequence tracking")
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def record_episode(self, data: dict, actions: np.ndarray, pnl: float):
        """FIX: Analyze episode for profitable patterns"""
        try:
            self._episode_count += 1
            episode_data = {
                "data": data,
                "actions": actions,
                "pnl": pnl,
                "sequence": list(self.current_sequence),
                "episode": self._episode_count
            }
            self.episode_buffer.append(episode_data)
            
            self.logger.info(f"Episode {self._episode_count} recorded: PnL=€{pnl:.2f}, sequence_length={len(self.current_sequence)}")
            
            # Identify profitable sequences
            if pnl > 10:  # €10+ profit
                if len(self.current_sequence) >= 3:
                    # Extract sequence pattern
                    pattern = self._extract_sequence_pattern(self.current_sequence)
                    
                    # Update pattern statistics
                    if pattern in self.sequence_patterns:
                        count, avg_pnl = self.sequence_patterns[pattern]
                        new_avg = (avg_pnl * count + pnl) / (count + 1)
                        self.sequence_patterns[pattern] = (count + 1, new_avg)
                        self.logger.info(f"Updated pattern '{pattern}': count={count + 1}, avg_pnl=€{new_avg:.2f}")
                    else:
                        self.sequence_patterns[pattern] = (1, pnl)
                        self.logger.info(f"New profitable pattern '{pattern}': €{pnl:.2f}")
                    
                    # Store if exceptional
                    if pnl > self.best_sequence_pnl:
                        old_best = self.best_sequence_pnl
                        self.best_sequence_pnl = pnl
                        self.profitable_sequences.append({
                            "sequence": list(self.current_sequence),
                            "pnl": pnl,
                            "pattern": pattern,
                            "episode": self._episode_count
                        })
                        
                        # Keep only top 10 sequences
                        self.profitable_sequences.sort(key=lambda x: x["pnl"], reverse=True)
                        self.profitable_sequences = self.profitable_sequences[:10]
                        
                        self.logger.info(f"New best sequence: €{pnl:.2f} (previous: €{old_best:.2f})")
                else:
                    self.logger.warning(f"Profitable episode but sequence too short: {len(self.current_sequence)} < 3")
            else:
                self.logger.debug(f"Episode {self._episode_count}: Not profitable enough (€{pnl:.2f} <= €10)")
                    
            # Reset for next episode
            self.current_sequence = []
            self.logger.debug(f"Episode {self._episode_count} processing complete, sequence reset")
            
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _extract_sequence_pattern(self, sequence: List[dict]) -> str:
        """Extract pattern from action sequence"""
        try:
            if not sequence:
                return "empty"
                
            # Simple pattern based on action directions
            action_pattern = []
            for step in sequence[-3:]:  # Last 3 actions
                action = step.get("action", np.array([0, 0]))
                if isinstance(action, np.ndarray) and len(action) >= 2:
                    direction = "buy" if action[0] > 0 else "sell" if action[0] < 0 else "hold"
                    action_pattern.append(direction)
                    
            pattern = "_".join(action_pattern)
            self.logger.debug(f"Extracted pattern: {pattern}")
            return pattern
        except Exception as e:
            self.logger.error(f"Error extracting sequence pattern: {e}")
            return "error"

    def maybe_replay(self, episode: int) -> float:
        """FIX: Dynamic replay bonus based on profitable patterns"""
        try:
            base_bonus = self.bonus if episode % self.interval == 0 else 0.0
            
            # Additional bonus if we have profitable patterns
            if self.sequence_patterns:
                # Find most profitable pattern
                best_pattern = max(self.sequence_patterns.items(), 
                                 key=lambda x: x[1][1])  # Sort by avg PnL
                pattern_name, (count, avg_pnl) = best_pattern
                
                # Bonus proportional to pattern success
                pattern_bonus = min(0.2, avg_pnl / 100.0)  # Cap at 0.2
                self.replay_bonus = base_bonus + pattern_bonus
                
                self.logger.info(f"Episode {episode}: Replay bonus={self.replay_bonus:.3f} (base={base_bonus:.3f}, pattern={pattern_bonus:.3f})")
                self.logger.info(f"Best pattern '{pattern_name}': {count} times, avg €{avg_pnl:.2f}")
            else:
                self.replay_bonus = base_bonus
                self.logger.debug(f"Episode {episode}: No patterns, base bonus={base_bonus:.3f}")
                
            return self.replay_bonus
        except Exception as e:
            self.logger.error(f"Error in maybe_replay: {e}")
            return 0.0

    def get_best_sequence_for_replay(self) -> Optional[List[dict]]:
        """Get the most profitable sequence for learning"""
        try:
            if self.profitable_sequences:
                best_seq = self.profitable_sequences[0]["sequence"]
                self.logger.info(f"Returning best sequence: €{self.profitable_sequences[0]['pnl']:.2f}, length={len(best_seq)}")
                return best_seq
            else:
                self.logger.debug("No profitable sequences available")
                return None
        except Exception as e:
            self.logger.error(f"Error getting best sequence: {e}")
            return None

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return replay analysis metrics"""
        try:
            n_patterns = float(len(self.sequence_patterns))
            avg_pattern_pnl = 0.0
            if self.sequence_patterns:
                avg_pattern_pnl = np.mean([pnl for _, pnl in self.sequence_patterns.values()])
                
            result = np.array([
                self.replay_bonus,
                n_patterns,
                avg_pattern_pnl,
                self.best_sequence_pnl
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: bonus={self.replay_bonus:.3f}, patterns={n_patterns}, avg_pnl={avg_pattern_pnl:.2f}, best={self.best_sequence_pnl:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(4, np.float32)

    def mutate(self, noise_std=0.05):
        """Evolve replay parameters"""
        try:
            old_bonus = self.bonus
            old_seq_len = self.sequence_len
            
            self.bonus = float(np.clip(self.bonus + np.random.normal(0, noise_std), 0.0, 0.5))
            self.sequence_len = int(np.clip(self.sequence_len + np.random.randint(-1, 2), 3, 10))
            
            self.logger.info(f"Mutated: bonus {old_bonus:.3f}->{self.bonus:.3f}, seq_len {old_seq_len}->{self.sequence_len}")
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
        
    def crossover(self, other: "HistoricalReplayAnalyzer"):
        child = HistoricalReplayAnalyzer(
            self.interval, 
            random.choice([self.bonus, other.bonus]),
            random.choice([self.sequence_len, other.sequence_len]),
            self.debug
        )
        self.logger.info("Crossover completed")
        return child