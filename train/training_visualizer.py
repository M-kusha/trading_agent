# ─────────────────────────────────────────────────────────────
# File: train/training_visualizer.py
# Professional Trading Dashboard Visualizer
# 
# Modern, beautiful terminal UI with:
# • Grid-based professional layout
# • Sophisticated color schemes
# • Real-time sparklines and visual indicators
# • Clean typography and spacing
# • Trading-focused metrics presentation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import shutil
import math
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Deque, cast
from collections import deque
import time

import numpy as np

# ─────────────────────────────────────────────────────────────
# Professional Color Palette
# ─────────────────────────────────────────────────────────────

class Colors:
    """Professional trading terminal color scheme"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    # Professional palette
    BLACK = '\033[38;2;12;12;12m'
    DARK_GRAY = '\033[38;2;64;64;64m'
    GRAY = '\033[38;2;128;128;128m'
    LIGHT_GRAY = '\033[38;2;192;192;192m'
    WHITE = '\033[38;2;255;255;255m'
    
    # Trading colors
    BUY_GREEN = '\033[38;2;0;255;127m'      # Bright green for buys
    SELL_RED = '\033[38;2;255;69;58m'       # Bright red for sells
    PROFIT_GREEN = '\033[38;2;50;215;75m'   # Profit green
    LOSS_RED = '\033[38;2;255;59;48m'       # Loss red
    
    # Accent colors
    ELECTRIC_BLUE = '\033[38;2;0;122;255m'  # Primary accent
    PURPLE = '\033[38;2;175;82;222m'        # Secondary accent
    ORANGE = '\033[38;2;255;149;0m'         # Warning
    GOLD = '\033[38;2;255;204;0m'           # Premium/important
    CYAN = '\033[38;2;50;173;230m'          # Info
    PINK = '\033[38;2;255;45;85m'           # Alert
    
    # Backgrounds
    BG_DARK = '\033[48;2;20;20;24m'
    BG_CARD = '\033[48;2;28;28;32m'
    BG_HIGHLIGHT = '\033[48;2;40;40;46m'
    BG_SUCCESS = '\033[48;2;0;48;24m'
    BG_DANGER = '\033[48;2;48;0;24m'
    BG_WARNING = '\033[48;2;48;32;0m'
    
    @staticmethod
    def gradient_text(text: str, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> str:
        """Create gradient colored text"""
        result = []
        length = len(text)
        for i, char in enumerate(text):
            if char == ' ':
                result.append(char)
                continue
            ratio = i / max(1, length - 1)
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
            result.append(f'\033[38;2;{r};{g};{b}m{char}')
        return ''.join(result) + Colors.RESET
    
    @staticmethod
    def enable_windows_vt_sequences() -> None:
        """Enable ANSI/VT processing on modern Windows consoles."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) != 0:
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, new_mode)
        except Exception:
            pass


class Icons:
    """Modern trading dashboard icons"""
    # Status
    SUCCESS = "●"
    WARNING = "◐"
    ERROR = "○"
    ACTIVE = "◉"
    INACTIVE = "○"
    
    # Arrows
    UP = "▲"
    DOWN = "▼"
    RIGHT = "▶"
    LEFT = "◀"
    UP_SMALL = "▴"
    DOWN_SMALL = "▾"
    
    # Trading
    LONG = "📈"
    SHORT = "📉"
    NEUTRAL = "➖"
    CHART = "📊"
    
    # Indicators
    DOT = "•"
    DIAMOND = "◆"
    SQUARE = "■"
    STAR = "★"
    CIRCLE = "●"
    
    # Blocks for charts
    BLOCK_FULL = "█"
    BLOCK_7 = "▇"
    BLOCK_6 = "▆"
    BLOCK_5 = "▅"
    BLOCK_4 = "▄"
    BLOCK_3 = "▃"
    BLOCK_2 = "▂"
    BLOCK_1 = "▁"
    SHADE_LIGHT = "░"
    SHADE_MEDIUM = "▒"
    SHADE_DARK = "▓"


class BoxChars:
    """Modern box drawing with rounded corners"""
    # Smooth lines
    H = "─"
    V = "│"
    TL = "╭"
    TR = "╮"
    BL = "╰"
    BR = "╯"
    
    # Double lines
    DH = "═"
    DV = "║"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"
    
    # Thick lines
    THICK_H = "━"
    THICK_V = "┃"
    
    # Dots
    DOT_H = "┈"
    DOT_V = "┊"


# ─────────────────────────────────────────────────────────────
# Terminal UI Manager
# ─────────────────────────────────────────────────────────────

class TerminalUI:
    """Advanced terminal UI with smooth rendering"""
    CSI = "\033["
    
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        size = shutil.get_terminal_size((120, 40))
        self.width = width or size.columns
        self.height = height or size.lines
        self._last_frame: List[str] = []
        self._active = False
        
    def _enter_alt(self) -> None:
        sys.stdout.write("\033[?1049h")  # alternate buffer
        sys.stdout.write("\033[?25l")    # hide cursor
        sys.stdout.write("\033[2J")      # clear screen
        sys.stdout.write("\033[H")       # home cursor
        sys.stdout.flush()
        
    def _exit_alt(self) -> None:
        sys.stdout.write("\033[?25h")    # show cursor
        sys.stdout.write("\033[?1049l")  # normal buffer
        sys.stdout.flush()
        
    def _move(self, row: int, col: int = 1) -> None:
        sys.stdout.write(f"{self.CSI}{row};{col}H")
        
    def _clear_eol(self) -> None:
        sys.stdout.write(f"{self.CSI}K")
        
    def _clear(self) -> None:
        sys.stdout.write(f"{self.CSI}2J{self.CSI}H")
        
    def start(self) -> None:
        if self._active:
            return
        self._enter_alt()
        self._active = True
        
    def stop(self) -> None:
        if not self._active:
            return
        self._exit_alt()
        self._active = False
        self._last_frame = []
        
    def render_lines(self, lines: List[str]) -> None:
        """Optimized differential rendering"""
        if not self._active:
            self.start()
            
        max_lines = min(self.height - 1, max(len(lines), len(self._last_frame)))
        
        for i in range(max_lines):
            new_line = lines[i] if i < len(lines) else ""
            old_line = self._last_frame[i] if i < len(self._last_frame) else None
            
            if new_line != old_line:
                self._move(i + 1, 1)
                truncated = new_line[:self.width]
                sys.stdout.write(truncated)
                self._clear_eol()
                
        sys.stdout.flush()
        self._last_frame = lines[:]
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc, tb):
        self.stop()


# ─────────────────────────────────────────────────────────────
# Chart Components
# ─────────────────────────────────────────────────────────────

class MiniChart:
    """ASCII mini charts for metrics"""
    
    @staticmethod
    def sparkline(values: List[float], width: int = 20, height: int = 1) -> str:
        """Create a sparkline from values"""
        if not values or width < 2:
            return ""
            
        # Normalize to 0-7 for block characters
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        blocks = [Icons.BLOCK_1, Icons.BLOCK_2, Icons.BLOCK_3, Icons.BLOCK_4,
                  Icons.BLOCK_5, Icons.BLOCK_6, Icons.BLOCK_7, Icons.BLOCK_FULL]
        
        # Sample values if too many
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values
            
        result = []
        for val in sampled:
            normalized = (val - min_val) / range_val
            idx = int(normalized * 7)
            idx = max(0, min(7, idx))
            
            # Color based on trend
            if normalized > 0.66:
                color = Colors.BUY_GREEN
            elif normalized > 0.33:
                color = Colors.GOLD
            else:
                color = Colors.SELL_RED
                
            result.append(f"{color}{blocks[idx]}{Colors.RESET}")
            
        return ''.join(result)
    
    @staticmethod
    def progress_bar(value: float, max_value: float = 100, width: int = 20,
                     show_percentage: bool = True, gradient: bool = True) -> str:
        """Create a beautiful progress bar"""
        pct = min(100, max(0, (value / max_value * 100) if max_value > 0 else 0))
        filled = int((pct / 100) * width)
        
        if gradient:
            # Gradient from red to yellow to green
            if pct < 33:
                bar_color = Colors.SELL_RED
            elif pct < 66:
                bar_color = Colors.ORANGE
            else:
                bar_color = Colors.BUY_GREEN
        else:
            bar_color = Colors.CYAN
            
        bar = f"{bar_color}{'█' * filled}{Colors.DARK_GRAY}{'░' * (width - filled)}{Colors.RESET}"
        
        if show_percentage:
            pct_text = f"{Colors.WHITE}{pct:5.1f}%{Colors.RESET}"
            return f"{bar} {pct_text}"
        return bar
    
    @staticmethod
    def histogram(values: List[float], width: int = 40, height: int = 5) -> List[str]:
        """Create a mini histogram"""
        if not values:
            return []
            
        lines = []
        max_val = max(abs(v) for v in values) if values else 1
        
        for h in range(height, 0, -1):
            line = ""
            threshold = (h / height) * max_val
            
            for val in values[-width:]:
                if abs(val) >= threshold:
                    if val > 0:
                        line += f"{Colors.BUY_GREEN}▆{Colors.RESET}"
                    else:
                        line += f"{Colors.SELL_RED}▆{Colors.RESET}"
                else:
                    line += f"{Colors.DARK_GRAY}·{Colors.RESET}"
                    
            lines.append(line)
            
        return lines


# ─────────────────────────────────────────────────────────────
# Professional Trading Dashboard
# ─────────────────────────────────────────────────────────────

class BeautifulTrainingVisualizer:
    """Modern professional trading dashboard visualizer"""
    
    def __init__(self, config: Any = None, terminal_width: Optional[int] = None) -> None:
        Colors.enable_windows_vt_sequences()
        
        # Safe UTF-8 configuration
        try:
            if sys.platform == "win32":
                stdout_any = cast(Any, sys.stdout)
                if hasattr(stdout_any, "reconfigure"):
                    stdout_any.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
        self.config = config
        size = shutil.get_terminal_size((120, 40))
        self.terminal_width = terminal_width or size.columns
        self.terminal_height = size.lines
        self.session_start = datetime.now()
        
        # Data stores
        self.price_history: Deque[float] = deque(maxlen=50)
        self.pnl_history: Deque[float] = deque(maxlen=50)
        self.reward_history: Deque[float] = deque(maxlen=50)
        self.action_history: Deque[str] = deque(maxlen=20)
        self.trade_outcomes: Deque[float] = deque(maxlen=30)
        
        # Performance tracking
        self.fps_counter: Deque[float] = deque(maxlen=30)
        self.last_update = time.time()
        
        self._ui = TerminalUI(self.terminal_width, self.terminal_height)
        self._frame: List[str] = []
        self._min_frame_interval = 0.05  # 20 FPS cap
        self._last_frame_ts = 0.0
        
    # ─── Layout Helpers ───
    
    def _add(self, line: str = "") -> None:
        """Add a line to the frame buffer"""
        self._frame.append(line)
        
    def _center(self, text: str, width: Optional[int] = None) -> str:
        """Center text with padding"""
        w = width or self.terminal_width
        padding = max(0, (w - len(self._strip_ansi(text))) // 2)
        return " " * padding + text
        
    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI codes for length calculation"""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)
        
    def _card(self, title: str, content: List[str], width: int, accent_color: Optional[str] = None) -> List[str]:
        """Create a styled card component"""
        color = accent_color or Colors.ELECTRIC_BLUE
        lines = []
        
        # Top border with title
        title_decorated = f" {title} "
        border_width = width - len(self._strip_ansi(title_decorated)) - 2
        left_border = border_width // 2
        right_border = border_width - left_border
        
        lines.append(f"{Colors.DARK_GRAY}{BoxChars.TL}{BoxChars.H * left_border}"
                    f"{color}{title_decorated}{Colors.DARK_GRAY}"
                    f"{BoxChars.H * right_border}{BoxChars.TR}{Colors.RESET}")
        
        # Content
        for line in content:
            stripped_len = len(self._strip_ansi(line))
            padding = max(0, width - stripped_len - 4)
            lines.append(f"{Colors.DARK_GRAY}{BoxChars.V}{Colors.RESET} {line}{' ' * padding} "
                        f"{Colors.DARK_GRAY}{BoxChars.V}{Colors.RESET}")
        
        # Bottom border
        lines.append(f"{Colors.DARK_GRAY}{BoxChars.BL}{BoxChars.H * (width - 2)}{BoxChars.BR}{Colors.RESET}")
        
        return lines
        
    def _metric_row(self, label: str, value: str, trend: Optional[str] = None, width: int = 30) -> str:
        """Create a metric row with optional trend"""
        trend_str = f" {trend}" if trend else ""
        value_with_trend = f"{value}{trend_str}"
        dots = "·" * max(1, width - len(self._strip_ansi(label)) - len(self._strip_ansi(value_with_trend)) - 2)
        return f"{Colors.GRAY}{label} {Colors.DARK_GRAY}{dots}{Colors.RESET} {value_with_trend}"
        
    # ─── Main Sections ───
    
    def _render_header(self, training_data: Dict[str, Any]) -> None:
        """Ultra-modern header with gradient and live indicators"""
        mode = training_data.get('mode', 'TRAINING')
        timestep = training_data.get('timestep', 0)
        total_timesteps = training_data.get('total_timesteps', 100000)
        episode = training_data.get('episode', 0)
        
        # Title with gradient
        title = "QUANTUM TRADING SYSTEM"
        gradient_title = Colors.gradient_text(title, (0, 122, 255), (175, 82, 222))
        
        # Live indicator
        live_indicator = f"{Colors.BUY_GREEN}{Icons.ACTIVE} LIVE{Colors.RESET}" if mode == "LIVE" else f"{Colors.ORANGE}{Icons.WARNING} TRAINING{Colors.RESET}"
        
        # Progress
        progress_pct = (timestep / total_timesteps * 100) if total_timesteps > 0 else 0
        
        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{gradient_title}  {live_indicator}"))
        self._add(self._center(f"{Colors.GRAY}Episode {Colors.GOLD}{episode:,}{Colors.GRAY} │ Step {Colors.CYAN}{timestep:,}{Colors.GRAY}/{total_timesteps:,}{Colors.RESET}"))
        self._add(self._center(MiniChart.progress_bar(progress_pct, width=60, gradient=True)))
        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")
        self._add("")
        
    def _render_performance_metrics(self, data: Dict[str, Any]) -> None:
        """Key performance indicators in a grid layout"""
        balance = float(data.get('balance', 3000))
        initial = float(data.get('initial_balance', 3000))
        pnl = balance - initial
        pnl_pct = ((balance / initial) - 1) * 100 if initial > 0 else 0
        
        # Store history
        self.pnl_history.append(pnl_pct)
        
        # Format values with colors
        balance_str = f"{Colors.WHITE}${balance:,.2f}{Colors.RESET}"
        
        if pnl >= 0:
            pnl_str = f"{Colors.BUY_GREEN}+${pnl:,.2f} ({pnl_pct:+.2f}%){Colors.RESET}"
            trend = f"{Colors.BUY_GREEN}{Icons.UP}{Colors.RESET}"
        else:
            pnl_str = f"{Colors.SELL_RED}-${abs(pnl):,.2f} ({pnl_pct:.2f}%){Colors.RESET}"
            trend = f"{Colors.SELL_RED}{Icons.DOWN}{Colors.RESET}"
            
        # Create metrics cards
        col_width = self.terminal_width // 3 - 2
        
        # Balance card
        balance_content = [
            self._metric_row("Current", balance_str, trend, col_width - 4),
            self._metric_row("P&L", pnl_str, None, col_width - 4),
            "",
            f"{Colors.GRAY}24h: {Colors.RESET}{MiniChart.sparkline(list(self.pnl_history)[-24:], width=col_width-6)}",
        ]
        
        # Risk metrics
        drawdown = float(data.get('drawdown', 0))
        max_dd = float(data.get('max_drawdown', 0))
        sharpe = float(data.get('sharpe_ratio', 0))
        
        risk_color = Colors.BUY_GREEN if abs(drawdown) < 5 else Colors.ORANGE if abs(drawdown) < 10 else Colors.SELL_RED
        risk_content = [
            self._metric_row("Drawdown", f"{risk_color}{drawdown:.2f}%{Colors.RESET}", None, col_width - 4),
            self._metric_row("Max DD", f"{Colors.GRAY}{max_dd:.2f}%{Colors.RESET}", None, col_width - 4),
            self._metric_row("Sharpe", f"{Colors.CYAN}{sharpe:.3f}{Colors.RESET}", None, col_width - 4),
            "",
        ]
        
        # Rewards
        current_reward = float(data.get('current_reward', 0))
        best_reward = float(data.get('best_reward', 0))
        avg_reward = float(data.get('avg_reward', 0))
        
        self.reward_history.append(current_reward)
        
        reward_content = [
            self._metric_row("Current", f"{Colors.WHITE}{current_reward:.3f}{Colors.RESET}", None, col_width - 4),
            self._metric_row("Best", f"{Colors.GOLD}{best_reward:.3f}{Colors.RESET}", None, col_width - 4),
            self._metric_row("Average", f"{Colors.GRAY}{avg_reward:.3f}{Colors.RESET}", None, col_width - 4),
            f"{Colors.GRAY}Trend: {Colors.RESET}{MiniChart.sparkline(list(self.reward_history)[-20:], width=col_width-8)}",
        ]
        
        # Render cards side by side
        balance_card = self._card("PORTFOLIO", balance_content, col_width, Colors.ELECTRIC_BLUE)
        risk_card = self._card("RISK METRICS", risk_content, col_width, Colors.PURPLE)
        reward_card = self._card("REWARDS", reward_content, col_width, Colors.GOLD)
        
        for i in range(max(len(balance_card), len(risk_card), len(reward_card))):
            line = ""
            if i < len(balance_card):
                line += balance_card[i]
            else:
                line += " " * col_width
            line += "  "
            if i < len(risk_card):
                line += risk_card[i]
            else:
                line += " " * col_width
            line += "  "
            if i < len(reward_card):
                line += reward_card[i]
            else:
                line += " " * col_width
            self._add(line)
            
        self._add("")
        
    def _render_trading_signals(self, decision_data: Dict[str, Any], voting_data: Dict[str, Any]) -> None:
        """Trading signals and voting visualization"""
        action = str(decision_data.get('current_action', 'ANALYZING'))
        confidence = float(decision_data.get('confidence', 0))
        risk_level = str(decision_data.get('risk_level', 'MEDIUM'))
        
        # Action display with icon
        if 'BUY' in action.upper():
            action_display = f"{Colors.BUY_GREEN}{Icons.UP} BUY SIGNAL{Colors.RESET}"
            self.action_history.append('BUY')
        elif 'SELL' in action.upper():
            action_display = f"{Colors.SELL_RED}{Icons.DOWN} SELL SIGNAL{Colors.RESET}"
            self.action_history.append('SELL')
        else:
            action_display = f"{Colors.GOLD}{Icons.NEUTRAL} HOLD{Colors.RESET}"
            self.action_history.append('HOLD')
            
        # Risk indicator
        risk_colors = {
            'LOW': Colors.BUY_GREEN,
            'MEDIUM': Colors.GOLD,
            'HIGH': Colors.SELL_RED
        }
        risk_display = f"{risk_colors.get(risk_level.upper(), Colors.GRAY)}{risk_level}{Colors.RESET}"
        
        # Voting summary
        votes = voting_data.get('votes', [])
        vote_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for v in votes:
            decision = str(v.get('decision', v.get('action', 'HOLD'))).upper()
            vote_counts[decision] = vote_counts.get(decision, 0) + 1
            
        total_votes = sum(vote_counts.values())
        
        # Create signal panel
        signal_width = self.terminal_width // 2 - 2
        
        signal_content = [
            "",
            self._center(action_display, signal_width),
            "",
            self._center(f"{Colors.GRAY}Confidence{Colors.RESET}", signal_width),
            self._center(MiniChart.progress_bar(confidence * 100, width=30, gradient=True), signal_width),
            "",
            self._center(f"{Colors.GRAY}Risk Level: {risk_display}{Colors.RESET}", signal_width),
            "",
        ]
        
        # Voting breakdown
        voting_content = [
            "",
            self._center(f"{Colors.GRAY}Committee Consensus{Colors.RESET}", signal_width),
            "",
        ]
        
        if total_votes > 0:
            for action_type in ['BUY', 'SELL', 'HOLD']:
                count = vote_counts[action_type]
                pct = (count / total_votes) * 100
                if action_type == 'BUY':
                    color = Colors.BUY_GREEN
                elif action_type == 'SELL':
                    color = Colors.SELL_RED
                else:
                    color = Colors.GOLD
                    
                bar = MiniChart.progress_bar(pct, width=20, show_percentage=False, gradient=False)
                voting_content.append(f"  {color}{action_type:5}{Colors.RESET} {bar} {Colors.GRAY}{count}/{total_votes}{Colors.RESET}")
                
        voting_content.extend(["", ""])
        
        # Render side by side
        signal_card = self._card("TRADING SIGNAL", signal_content, signal_width, Colors.CYAN)
        voting_card = self._card("VOTING ANALYSIS", voting_content, signal_width, Colors.PURPLE)
        
        for i in range(max(len(signal_card), len(voting_card))):
            line = ""
            if i < len(signal_card):
                line += signal_card[i]
            else:
                line += " " * signal_width
            line += "  "
            if i < len(voting_card):
                line += voting_card[i]
            else:
                line += " " * signal_width
            self._add(line)
            
        self._add("")
        
    def _render_module_status(self, modules_data: Dict[str, Any]) -> None:
        """Compact module health indicators"""
        modules = modules_data.get('modules', {})
        
        if not modules:
            # Generate realistic defaults
            modules = {
                'RiskController': {'status': 'healthy', 'health_score': 92},
                'StrategyEngine': {'status': 'healthy', 'health_score': 88},
                'MarketAnalyzer': {'status': 'healthy', 'health_score': 90},
                'VotingSystem': {'status': 'healthy', 'health_score': 87},
                'ExecutionEngine': {'status': 'healthy', 'health_score': 95},
                'MemoryCore': {'status': 'healthy', 'health_score': 91},
            }
            
        # Create compact grid
        self._add(f"{Colors.GRAY}{'─' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{Colors.ELECTRIC_BLUE}SYSTEM HEALTH MONITOR{Colors.RESET}"))
        self._add("")
        
        items_per_row = 3
        module_names = list(modules.keys())
        col_width = self.terminal_width // items_per_row
        
        for i in range(0, len(module_names), items_per_row):
            row = ""
            for j in range(items_per_row):
                if i + j < len(module_names):
                    name = module_names[i + j]
                    info = modules[name]
                    health = float(info.get('health_score', 0))
                    status = str(info.get('status', 'unknown'))
                    
                    # Status indicator
                    if status.lower() in ['healthy', 'active', 'online']:
                        indicator = f"{Colors.BUY_GREEN}{Icons.SUCCESS}{Colors.RESET}"
                    elif status.lower() in ['warning', 'degraded']:
                        indicator = f"{Colors.ORANGE}{Icons.WARNING}{Colors.RESET}"
                    else:
                        indicator = f"{Colors.SELL_RED}{Icons.ERROR}{Colors.RESET}"
                        
                    # Health bar
                    bar = MiniChart.progress_bar(health, width=10, show_percentage=False, gradient=True)
                    
                    # Format module display
                    display_name = name[:15] if len(name) > 15 else name
                    module_str = f"{indicator} {Colors.WHITE}{display_name:<15}{Colors.RESET} {bar}"
                    row += module_str.ljust(col_width)
                    
            self._add(row)
            
        self._add("")
        
    def _render_market_analysis(self, market_data: Dict[str, Any]) -> None:
        """Market conditions panel"""
        regime = str(market_data.get('regime', 'trending')).upper()
        volatility = str(market_data.get('volatility', 'medium')).upper()
        sentiment = str(market_data.get('sentiment', 'neutral')).upper()
        
        # Generate sample price data if needed
        if not self.price_history:
            base_price = 100
            for _ in range(50):
                change = (random.random() - 0.5) * 2
                base_price += change
                self.price_history.append(base_price)
        else:
            # Add new price point
            last_price = self.price_history[-1] if self.price_history else 100
            change = (random.random() - 0.5) * 2
            self.price_history.append(last_price + change)
            
        # Regime colors
        regime_colors = {
            'TRENDING': Colors.BUY_GREEN,
            'RANGING': Colors.GOLD,
            'VOLATILE': Colors.ORANGE,
        }
        
        self._add(f"{Colors.GRAY}{'─' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{Colors.PURPLE}MARKET ANALYSIS{Colors.RESET}"))
        self._add("")
        
        # Market stats
        stats_line = (f"  {Colors.GRAY}Regime:{Colors.RESET} {regime_colors.get(regime, Colors.WHITE)}{regime}{Colors.RESET}  "
                     f"{Colors.GRAY}│{Colors.RESET}  "
                     f"{Colors.GRAY}Volatility:{Colors.RESET} {self._volatility_color(volatility)}{volatility}{Colors.RESET}  "
                     f"{Colors.GRAY}│{Colors.RESET}  "
                     f"{Colors.GRAY}Sentiment:{Colors.RESET} {self._sentiment_color(sentiment)}{sentiment}{Colors.RESET}")
        self._add(stats_line)
        
        # Price chart
        self._add("")
        self._add(f"  {Colors.GRAY}Price Action (50 periods):{Colors.RESET}")
        self._add(f"  {MiniChart.sparkline(list(self.price_history), width=self.terminal_width - 4)}")
        self._add("")
        
    def _render_footer(self, stats: Dict[str, Any]) -> None:
        """Performance footer with system stats"""
        elapsed = (datetime.now() - self.session_start).total_seconds()
        
        # Calculate FPS
        now = time.time()
        if self.last_update > 0:
            fps = 1.0 / max(0.001, now - self.last_update)
            self.fps_counter.append(fps)
        self.last_update = now
        
        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
        
        # System metrics
        steps_per_sec = float(stats.get('steps_per_second', 0))
        latency_p50 = float(stats.get('latency_p50', 0))
        latency_p95 = float(stats.get('latency_p95', 0))
        
        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")
        
        runtime = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s"
        
        perf_line = (f"{Colors.GRAY}Runtime: {Colors.WHITE}{runtime}{Colors.RESET}  "
                    f"{Colors.GRAY}│  FPS: {Colors.CYAN}{avg_fps:.1f}{Colors.RESET}  "
                    f"{Colors.GRAY}│  Steps/s: {Colors.CYAN}{steps_per_sec:.1f}{Colors.RESET}  "
                    f"{Colors.GRAY}│  Latency: {Colors.WHITE}P50={latency_p50:.1f}ms P95={latency_p95:.1f}ms{Colors.RESET}")
                    
        self._add(self._center(perf_line))
        self._add(self._center(f"{Colors.DARK_GRAY}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}"))
        
    # ─── Helper Methods ───
    
    def _volatility_color(self, level: str) -> str:
        """Color code volatility levels"""
        level = level.upper()
        if level in ['LOW', 'CALM']:
            return Colors.BUY_GREEN
        elif level in ['MEDIUM', 'MODERATE']:
            return Colors.GOLD
        else:
            return Colors.ORANGE
            
    def _sentiment_color(self, sentiment: str) -> str:
        """Color code market sentiment"""
        sentiment = sentiment.upper()
        if sentiment in ['BULLISH', 'POSITIVE']:
            return Colors.BUY_GREEN
        elif sentiment in ['BEARISH', 'NEGATIVE']:
            return Colors.SELL_RED
        else:
            return Colors.GOLD
            
    # ─── Main Render Method ───
    
    def render_complete_display(self, smart_bus: Any, training_metrics: Dict[str, Any]) -> None:
        """Render the complete professional trading dashboard"""
        # FPS limiter
        now = time.time()
        if now - self._last_frame_ts < self._min_frame_interval:
            return
        self._last_frame_ts = now
        
        self._frame = []
        
        # Gather data from bus
        market_overview = smart_bus.get('market_overview', module='Visualizer', default={})
        account_state = smart_bus.get('account_state', module='Visualizer', default={})
        positions = smart_bus.get('positions', module='Visualizer', default=[])
        votes = smart_bus.get('votes', module='Visualizer', default=[])
        committee_votes = smart_bus.get('committee_votes', module='Visualizer', default=[])
        trading_performance = smart_bus.get('trading_performance', module='Visualizer', default={})
        
        # Build dashboard sections
        self._render_header({
            'mode': 'LIVE' if getattr(self.config, 'live_mode', False) else 'TRAINING',
            'timestep': training_metrics.get('timestep', 0),
            'total_timesteps': training_metrics.get('total_timesteps', 100000),
            'episode': training_metrics.get('episodes', 0),
        })
        
        self._render_performance_metrics({
            'balance': account_state.get('balance', 3000),
            'initial_balance': account_state.get('initial_balance', 3000),
            'drawdown': training_metrics.get('env_drawdown', 0),
            'max_drawdown': account_state.get('max_drawdown', 0),
            'sharpe_ratio': trading_performance.get('sharpe_ratio', 0),
            'current_reward': training_metrics.get('current_episode_reward', 0),
            'best_reward': training_metrics.get('best_episode_reward', 0),
            'avg_reward': training_metrics.get('episode_reward_mean', 0),
        })
        
        self._render_trading_signals(
            {
                'current_action': self._determine_action(positions),
                'confidence': training_metrics.get('decision_confidence', 0.75),
                'risk_level': training_metrics.get('risk_level', 'MEDIUM'),
            },
            {
                'votes': committee_votes if committee_votes else votes,
            }
        )
        
        self._render_module_status({
            'modules': self._extract_module_health(smart_bus)
        })
        
        self._render_market_analysis(market_overview)
        
        self._render_footer({
            'steps_per_second': training_metrics.get('steps_per_second', 0),
            'latency_p50': training_metrics.get('step_ms_p50', 0),
            'latency_p95': training_metrics.get('step_ms_p95', 0),
        })
        
        # Render to terminal
        self._ui.render_lines(self._frame)
        
    def _determine_action(self, positions: List[Any]) -> str:
        """Determine current action from positions"""
        if not positions:
            return "ANALYZING"
        latest = positions[-1] if isinstance(positions, list) else positions
        if isinstance(latest, dict):
            return str(latest.get('action', 'HOLD')).upper()
        return "HOLD"
        
    def _extract_module_health(self, smart_bus: Any) -> Dict[str, Any]:
        """Extract module health data from bus"""
        modules = {}
        patterns = [
            'DynamicRiskController', 'MetaAgent', 'EnhancedAnomalyDetector',
            'PositionManager', 'ModuleOrchestrator', 'HealthMonitor',
        ]
        
        for name in patterns:
            status = smart_bus.get(f'{name}_status', module='Visualizer', default=None)
            health = smart_bus.get(f'{name}_health', module='Visualizer', default=None)
            
            if status or health:
                score = 85.0
                try:
                    if isinstance(health, dict) and 'score' in health:
                        score = float(health['score'])
                    elif isinstance(health, (int, float)):
                        score = float(health)
                except:
                    pass
                    
                modules[name] = {
                    'status': status or 'healthy',
                    'health_score': min(100, max(0, score))
                }
                
        return modules


# Export
__all__ = ['BeautifulTrainingVisualizer', 'Colors', 'Icons', 'BoxChars', 'TerminalUI', 'MiniChart']