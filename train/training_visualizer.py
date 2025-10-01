# ─────────────────────────────────────────────────────────────
# File: train/training_visualizer.py
# Professional Trading Dashboard Visualizer (responsive & ANSI-safe)
#
# • Full-screen, in-place updates (alternate buffer)
# • Frame-wide repaint each tick (no duplicates/ghosting)
# • ANSI-safe width fitting (no broken escape codes)
# • Unicode visual-width aware (emojis, box chars)
# • Auto-resizes every frame (responsive layout)
# • Uses full terminal height with safe clearing
# • Pylance-clean: guarded sys.stdout.reconfigure usage
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import shutil
import math
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Deque, cast
from collections import deque
import time
import re
import unicodedata

import numpy as np

# ─────────────────────────────────────────────────────────────
# Low-level ANSI / width utilities (no external deps)
# ─────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

def _char_display_width(ch: str) -> int:
    """
    Approximate visual width for terminal:
    - Fullwidth/Wide chars count as 2 (CJK)
    - Some emojis and symbols treated as width 2
    - Combining marks width 0
    - Default 1
    """
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me"):  # combining marks
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    # Treat common emoji blocks as width 2
    code = ord(ch)
    if (
        0x1F300 <= code <= 0x1FAFF  # Misc/Emoticons/Symbols
        or 0x1F900 <= code <= 0x1F9FF
        or 0x1F000 <= code <= 0x1F02F
        or ch in {"●", "◉", "○", "◆", "★", "◐", "▲", "▼", "▶", "◀"}
    ):
        return 2
    return 1

def _visible_width(text: str) -> int:
    """Compute printed width ignoring ANSI escapes and counting unicode width."""
    width = 0
    i = 0
    while i < len(text):
        if text[i] == "\033":
            # skip ANSI sequence until 'm'
            m = _ANSI_RE.match(text, i)
            if m:
                i = m.end()
                continue
        width += _char_display_width(text[i])
        i += 1
    return width

def _ansi_safe_truncate(text: str, max_width: int) -> str:
    """Truncate to max visible width without breaking ANSI codes. Appends RESET."""
    if max_width <= 0:
        return ""
    out = []
    width = 0
    i = 0
    stack_has_color = False

    while i < len(text) and width < max_width:
        if text[i] == "\033":
            m = _ANSI_RE.match(text, i)
            if m:
                seq = text[i:m.end()]
                out.append(seq)
                # naive: track that we printed some color sequences
                stack_has_color = True
                i = m.end()
                continue
        ch = text[i]
        w = _char_display_width(ch)
        if width + w > max_width:
            break
        out.append(ch)
        width += w
        i += 1

    # ensure we don't leave terminal colored
    out.append(Colors.RESET)
    return "".join(out)

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
        length = max(1, len(text))
        for i, char in enumerate(text):
            if char == ' ':
                result.append(char)
                continue
            ratio = i / (length - 1) if length > 1 else 0.0
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
            import ctypes  # type: ignore
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
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
    H = "─"
    V = "│"
    TL = "╭"
    TR = "╮"
    BL = "╰"
    BR = "╯"
    DH = "═"
    DV = "║"
    DTL = "╔"
    DTR = "╗"
    DBL = "╚"
    DBR = "╝"
    THICK_H = "━"
    THICK_V = "┃"
    DOT_H = "┈"
    DOT_V = "┊"

# ─────────────────────────────────────────────────────────────
# Terminal UI Manager (full-frame repaint, ANSI-safe)
# ─────────────────────────────────────────────────────────────

class TerminalUI:
    """Advanced terminal UI with smooth, safe rendering"""
    CSI = "\033["

    def __init__(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        size = shutil.get_terminal_size((120, 40))
        self.width = width or size.columns
        self.height = height or size.lines
        self._active = False
        self._last_frame_line_count = 0

    def _refresh_size(self) -> None:
        size = shutil.get_terminal_size((120, 40))
        self.width = size.columns if self.width is None else size.columns
        self.height = size.lines if self.height is None else size.lines

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
        self._last_frame_line_count = 0

    def render_lines(self, lines: List[str]) -> None:
        """Full repaint with ANSI-safe truncation and clearing."""
        if not self._active:
            self.start()

        # Always recalc size for responsiveness
        self._refresh_size()

        # Cap to available height (leave at least 0 lines for footer if needed)
        max_lines = min(self.height, len(lines))

        # Move to home, clear screen, print lines trimmed to width
        sys.stdout.write("\033[H")   # home
        for i in range(max_lines):
            # Truncate safely to current width
            safe = _ansi_safe_truncate(lines[i], self.width)
            sys.stdout.write(safe)
            # Clear to end of line to remove leftovers
            sys.stdout.write(f"{self.CSI}K")
            if i != max_lines - 1:
                sys.stdout.write("\n")

        # If previous frame had more lines, clear the remainder
        leftover = max(0, self._last_frame_line_count - max_lines)
        for _ in range(leftover):
            sys.stdout.write("\n")
            sys.stdout.write(f"{self.CSI}K")

        sys.stdout.flush()
        self._last_frame_line_count = max_lines

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
        range_val = max(max_val - min_val, 1e-12)

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
            idx = min(7, max(0, int(normalized * 7)))
            # Color based on relative height
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
        pct = 0.0
        if max_value > 0:
            pct = max(0.0, min(100.0, (value / max_value) * 100.0))
        filled = int((pct / 100.0) * width)

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
        if not values or width <= 0 or height <= 0:
            return []

        lines: List[str] = []
        max_val = max(abs(v) for v in values) if values else 1.0
        max_val = max(max_val, 1e-9)

        series = values[-width:]
        for h in range(height, 0, -1):
            line = ""
            threshold = (h / height) * max_val
            for val in series:
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
        self._frame.append(line)

    def _center(self, text: str, width: Optional[int] = None) -> str:
        w = width or self.terminal_width
        pad = max(0, (w - _visible_width(text)) // 2)
        return (" " * pad) + text

    def _card(self, title: str, content: List[str], width: int, accent_color: Optional[str] = None) -> List[str]:
        """Create a styled card component (ANSI-safe padding)"""
        color = accent_color or Colors.ELECTRIC_BLUE
        lines: List[str] = []

        title_decorated = f" {title} "
        border_width = max(2, width - _visible_width(title_decorated) - 2)
        left_border = border_width // 2
        right_border = border_width - left_border

        # Top border with title
        lines.append(
            f"{Colors.DARK_GRAY}{BoxChars.TL}{BoxChars.H * left_border}"
            f"{color}{title_decorated}{Colors.DARK_GRAY}"
            f"{BoxChars.H * right_border}{BoxChars.TR}{Colors.RESET}"
        )

        # Content
        inner_width = max(0, width - 4)
        for raw in content:
            truncated = _ansi_safe_truncate(raw, inner_width)
            pad_len = max(0, inner_width - _visible_width(truncated))
            lines.append(
                f"{Colors.DARK_GRAY}{BoxChars.V}{Colors.RESET} {truncated}{' ' * pad_len} {Colors.DARK_GRAY}{BoxChars.V}{Colors.RESET}"
            )

        # Bottom border
        lines.append(f"{Colors.DARK_GRAY}{BoxChars.BL}{BoxChars.H * (width - 2)}{BoxChars.BR}{Colors.RESET}")
        return lines

    def _metric_row(self, label: str, value: str, trend: Optional[str] = None, width: int = 30) -> str:
        trend_str = f" {trend}" if trend else ""
        value_with_trend = f"{value}{trend_str}"
        dot_space = max(1, width - _visible_width(label) - _visible_width(value_with_trend) - 2)
        dots = "·" * dot_space
        return f"{Colors.GRAY}{label} {Colors.DARK_GRAY}{dots}{Colors.RESET} {value_with_trend}"

    # ─── Main Sections ───

    def _render_header(self, training_data: Dict[str, Any]) -> None:
        mode = training_data.get('mode', 'TRAINING')
        timestep = int(training_data.get('timestep', 0))
        total_timesteps = int(training_data.get('total_timesteps', 100000))
        episode = int(training_data.get('episode', 0))

        title = "QUANTUM TRADING SYSTEM"
        gradient_title = Colors.gradient_text(title, (0, 122, 255), (175, 82, 222))
        live_indicator = (
            f"{Colors.BUY_GREEN}{Icons.ACTIVE} LIVE{Colors.RESET}"
            if mode == "LIVE" else f"{Colors.ORANGE}{Icons.WARNING} TRAINING{Colors.RESET}"
        )
        progress_pct = (timestep / total_timesteps * 100.0) if total_timesteps > 0 else 0.0

        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{gradient_title}  {live_indicator}"))
        self._add(
            self._center(
                f"{Colors.GRAY}Episode {Colors.GOLD}{episode:,}{Colors.GRAY} │ Step {Colors.CYAN}{timestep:,}"
                f"{Colors.GRAY}/{total_timesteps:,}{Colors.RESET}"
            )
        )
        self._add(self._center(MiniChart.progress_bar(progress_pct, width=max(20, min(60, self.terminal_width - 10)), gradient=True)))
        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")
        self._add("")

    def _render_performance_metrics(self, data: Dict[str, Any]) -> None:
        balance = float(data.get('balance', 3000))
        initial = float(data.get('initial_balance', 3000))
        pnl = balance - initial
        pnl_pct = ((balance / initial) - 1) * 100 if initial > 0 else 0.0

        self.pnl_history.append(pnl_pct)

        balance_str = f"{Colors.WHITE}${balance:,.2f}{Colors.RESET}"
        if pnl >= 0:
            pnl_str = f"{Colors.BUY_GREEN}+${pnl:,.2f} ({pnl_pct:+.2f}%){Colors.RESET}"
            trend = f"{Colors.BUY_GREEN}{Icons.UP}{Colors.RESET}"
        else:
            pnl_str = f"{Colors.SELL_RED}-${abs(pnl):,.2f} ({pnl_pct:.2f}%){Colors.RESET}"
            trend = f"{Colors.SELL_RED}{Icons.DOWN}{Colors.RESET}"

        # Responsive columns
        cols = 3
        gutter = 2
        col_width = max(30, (self.terminal_width - gutter * (cols - 1)) // cols)

        # Balance card
        balance_content = [
            self._metric_row("Current", balance_str, trend, col_width - 4),
            self._metric_row("P&L", pnl_str, None, col_width - 4),
            "",
            f"{Colors.GRAY}24h: {Colors.RESET}{MiniChart.sparkline(list(self.pnl_history)[-min(24, col_width-6):], width=max(10, col_width-6))}",
        ]

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

        current_reward = float(data.get('current_reward', 0))
        best_reward = float(data.get('best_reward', 0))
        avg_reward = float(data.get('avg_reward', 0))

        self.reward_history.append(current_reward)

        reward_content = [
            self._metric_row("Current", f"{Colors.WHITE}{current_reward:.3f}{Colors.RESET}", None, col_width - 4),
            self._metric_row("Best", f"{Colors.GOLD}{best_reward:.3f}{Colors.RESET}", None, col_width - 4),
            self._metric_row("Average", f"{Colors.GRAY}{avg_reward:.3f}{Colors.RESET}", None, col_width - 4),
            f"{Colors.GRAY}Trend: {Colors.RESET}{MiniChart.sparkline(list(self.reward_history)[-min(20, col_width-8):], width=max(10, col_width-8))}",
        ]

        balance_card = self._card("PORTFOLIO", balance_content, col_width, Colors.ELECTRIC_BLUE)
        risk_card = self._card("RISK METRICS", risk_content, col_width, Colors.PURPLE)
        reward_card = self._card("REWARDS", reward_content, col_width, Colors.GOLD)

        for i in range(max(len(balance_card), len(risk_card), len(reward_card))):
            line = ""
            line += balance_card[i] if i < len(balance_card) else " " * col_width
            line += " " * gutter
            line += risk_card[i] if i < len(risk_card) else " " * col_width
            line += " " * gutter
            line += reward_card[i] if i < len(reward_card) else " " * col_width
            self._add(line)

        self._add("")

    def _render_trading_signals(self, decision_data: Dict[str, Any], voting_data: Dict[str, Any]) -> None:
        action = str(decision_data.get('current_action', 'ANALYZING'))
        confidence = float(decision_data.get('confidence', 0))
        risk_level = str(decision_data.get('risk_level', 'MEDIUM'))

        if 'BUY' in action.upper():
            action_display = f"{Colors.BUY_GREEN}{Icons.UP} BUY SIGNAL{Colors.RESET}"
            self.action_history.append('BUY')
        elif 'SELL' in action.upper():
            action_display = f"{Colors.SELL_RED}{Icons.DOWN} SELL SIGNAL{Colors.RESET}"
            self.action_history.append('SELL')
        else:
            action_display = f"{Colors.GOLD}{Icons.NEUTRAL} HOLD{Colors.RESET}"
            self.action_history.append('HOLD')

        risk_colors = {'LOW': Colors.BUY_GREEN, 'MEDIUM': Colors.GOLD, 'HIGH': Colors.SELL_RED}
        risk_display = f"{risk_colors.get(risk_level.upper(), Colors.GRAY)}{risk_level}{Colors.RESET}"

        votes = voting_data.get('votes', [])
        vote_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for v in votes:
            decision = str(v.get('decision', v.get('action', 'HOLD'))).upper()
            vote_counts[decision] = vote_counts.get(decision, 0) + 1
        total_votes = max(1, sum(vote_counts.values()))

        gutter = 2
        signal_width = max(40, (self.terminal_width - gutter) // 2)

        signal_content = [
            "",
            self._center(action_display, signal_width),
            "",
            self._center(f"{Colors.GRAY}Confidence{Colors.RESET}", signal_width),
            self._center(MiniChart.progress_bar(confidence * 100, width=max(20, signal_width - 20), gradient=True), signal_width),
            "",
            self._center(f"{Colors.GRAY}Risk Level: {risk_display}{Colors.RESET}", signal_width),
            "",
        ]

        voting_content = [
            "",
            self._center(f"{Colors.GRAY}Committee Consensus{Colors.RESET}", signal_width),
            "",
        ]

        for action_type in ['BUY', 'SELL', 'HOLD']:
            count = vote_counts[action_type]
            pct = (count / total_votes) * 100.0
            color = Colors.BUY_GREEN if action_type == 'BUY' else Colors.SELL_RED if action_type == 'SELL' else Colors.GOLD
            bar = MiniChart.progress_bar(pct, width=max(10, signal_width - 20), show_percentage=False, gradient=False)
            voting_content.append(f"  {color}{action_type:5}{Colors.RESET} {bar} {Colors.GRAY}{count}/{total_votes}{Colors.RESET}")

        voting_content.extend(["", ""])

        signal_card = self._card("TRADING SIGNAL", signal_content, signal_width, Colors.CYAN)
        voting_card = self._card("VOTING ANALYSIS", voting_content, signal_width, Colors.PURPLE)

        for i in range(max(len(signal_card), len(voting_card))):
            line = ""
            line += signal_card[i] if i < len(signal_card) else " " * signal_width
            line += " " * gutter
            line += voting_card[i] if i < len(voting_card) else " " * signal_width
            self._add(line)

        self._add("")

    def _render_module_status(self, modules_data: Dict[str, Any]) -> None:
        modules = modules_data.get('modules', {})
        if not modules:
            modules = {
                'RiskController': {'status': 'healthy', 'health_score': 92},
                'StrategyEngine': {'status': 'healthy', 'health_score': 88},
                'MarketAnalyzer': {'status': 'healthy', 'health_score': 90},
                'VotingSystem': {'status': 'healthy', 'health_score': 87},
                'ExecutionEngine': {'status': 'healthy', 'health_score': 95},
                'MemoryCore': {'status': 'healthy', 'health_score': 91},
            }

        self._add(f"{Colors.GRAY}{'─' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{Colors.ELECTRIC_BLUE}SYSTEM HEALTH MONITOR{Colors.RESET}"))
        self._add("")

        items_per_row = max(2, min(4, self.terminal_width // 28))
        col_width = max(24, self.terminal_width // items_per_row)
        names = list(modules.keys())

        for i in range(0, len(names), items_per_row):
            row = ""
            for j in range(items_per_row):
                if i + j >= len(names):
                    continue
                name = names[i + j]
                info = modules[name]
                health = float(info.get('health_score', 0))
                status = str(info.get('status', 'unknown'))

                if status.lower() in ['healthy', 'active', 'online']:
                    indicator = f"{Colors.BUY_GREEN}{Icons.SUCCESS}{Colors.RESET}"
                elif status.lower() in ['warning', 'degraded']:
                    indicator = f"{Colors.ORANGE}{Icons.WARNING}{Colors.RESET}"
                else:
                    indicator = f"{Colors.SELL_RED}{Icons.ERROR}{Colors.RESET}"

                bar = MiniChart.progress_bar(health, width=10, show_percentage=False, gradient=True)
                display_name = name if _visible_width(name) <= 15 else name[:15]
                module_str = f"{indicator} {Colors.WHITE}{display_name:<15}{Colors.RESET} {bar}"
                row += _ansi_safe_truncate(module_str.ljust(col_width), col_width)
            self._add(row)

        self._add("")

    def _render_market_analysis(self, market_data: Dict[str, Any]) -> None:
        regime = str(market_data.get('regime', 'trending')).upper()
        volatility = str(market_data.get('volatility', 'medium')).upper()
        sentiment = str(market_data.get('sentiment', 'neutral')).upper()

        # Generate / update sample price data if needed
        if not self.price_history:
            base_price = 100.0
            for _ in range(50):
                base_price += (random.random() - 0.5) * 2.0
                self.price_history.append(base_price)
        else:
            last_price = self.price_history[-1]
            self.price_history.append(last_price + (random.random() - 0.5) * 2.0)

        regime_colors = {'TRENDING': Colors.BUY_GREEN, 'RANGING': Colors.GOLD, 'VOLATILE': Colors.ORANGE}

        self._add(f"{Colors.GRAY}{'─' * self.terminal_width}{Colors.RESET}")
        self._add(self._center(f"{Colors.PURPLE}MARKET ANALYSIS{Colors.RESET}"))
        self._add("")

        stats_line = (
            f"  {Colors.GRAY}Regime:{Colors.RESET} {regime_colors.get(regime, Colors.WHITE)}{regime}{Colors.RESET}  "
            f"{Colors.GRAY}│{Colors.RESET}  "
            f"{Colors.GRAY}Volatility:{Colors.RESET} {self._volatility_color(volatility)}{volatility}{Colors.RESET}  "
            f"{Colors.GRAY}│{Colors.RESET}  "
            f"{Colors.GRAY}Sentiment:{Colors.RESET} {self._sentiment_color(sentiment)}{sentiment}{Colors.RESET}"
        )
        self._add(_ansi_safe_truncate(stats_line, self.terminal_width))

        self._add("")
        self._add(f"  {Colors.GRAY}Price Action (50 periods):{Colors.RESET}")
        self._add(f"  {MiniChart.sparkline(list(self.price_history), width=max(10, self.terminal_width - 4))}")
        self._add("")

    def _render_footer(self, stats: Dict[str, Any]) -> None:
        elapsed = (datetime.now() - self.session_start).total_seconds()

        now = time.time()
        if self.last_update > 0:
            fps = 1.0 / max(0.001, now - self.last_update)
            self.fps_counter.append(fps)
        self.last_update = now
        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0.0

        steps_per_sec = float(stats.get('steps_per_second', 0))
        latency_p50 = float(stats.get('latency_p50', 0))
        latency_p95 = float(stats.get('latency_p95', 0))

        self._add(f"{Colors.DARK_GRAY}{'═' * self.terminal_width}{Colors.RESET}")

        runtime = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s"

        perf_line = (
            f"{Colors.GRAY}Runtime: {Colors.WHITE}{runtime}{Colors.RESET}  "
            f"{Colors.GRAY}│  FPS: {Colors.CYAN}{avg_fps:.1f}{Colors.RESET}  "
            f"{Colors.GRAY}│  Steps/s: {Colors.CYAN}{steps_per_sec:.1f}{Colors.RESET}  "
            f"{Colors.GRAY}│  Latency: {Colors.WHITE}P50={latency_p50:.1f}ms P95={latency_p95:.1f}ms{Colors.RESET}"
        )

        self._add(self._center(_ansi_safe_truncate(perf_line, self.terminal_width)))
        self._add(self._center(f"{Colors.DARK_GRAY}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}"))

    # ─── Helper Methods ───

    def _volatility_color(self, level: str) -> str:
        level = level.upper()
        if level in ['LOW', 'CALM']:
            return Colors.BUY_GREEN
        elif level in ['MEDIUM', 'MODERATE']:
            return Colors.GOLD
        else:
            return Colors.ORANGE

    def _sentiment_color(self, sentiment: str) -> str:
        sentiment = sentiment.upper()
        if sentiment in ['BULLISH', 'POSITIVE']:
            return Colors.BUY_GREEN
        elif sentiment in ['BEARISH', 'NEGATIVE']:
            return Colors.SELL_RED
        else:
            return Colors.GOLD

    def _safe_bus_get(self, smart_bus: Any, key: str, default: Any) -> Any:
        """Handle SmartInfoBus or dict-like sources gracefully."""
        try:
            # Your SmartInfoBus signature: get(key, module='Visualizer', default=...)
            return smart_bus.get(key, module='Visualizer', default=default)  # type: ignore[attr-defined]
        except Exception:
            try:
                return smart_bus.get(key, default)  # dict-like
            except Exception:
                return default

    # ─── Main Render Method ───

    def render_complete_display(self, smart_bus: Any, training_metrics: Dict[str, Any]) -> None:
        """Render the complete professional trading dashboard"""
        # FPS limiter
        now = time.time()
        if now - self._last_frame_ts < self._min_frame_interval:
            return
        self._last_frame_ts = now

        # Sync terminal size from UI (responsive)
        self.terminal_width = self._ui.width
        self.terminal_height = self._ui.height

        self._frame = []

        # Gather data from bus (safe)
        market_overview = self._safe_bus_get(smart_bus, 'market_overview', {}) or {}
        account_state = self._safe_bus_get(smart_bus, 'account_state', {}) or {}
        positions = self._safe_bus_get(smart_bus, 'positions', []) or []
        votes = self._safe_bus_get(smart_bus, 'votes', []) or []
        committee_votes = self._safe_bus_get(smart_bus, 'committee_votes', []) or []
        trading_performance = self._safe_bus_get(smart_bus, 'trading_performance', {}) or {}

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

        # Ensure we don't overflow available height: trim frame safely
        if len(self._frame) > self.terminal_height:
            self._frame = self._frame[:self.terminal_height]

        # Render to terminal (full repaint)
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
        modules: Dict[str, Any] = {}
        patterns = [
            'DynamicRiskController', 'MetaAgent', 'EnhancedAnomalyDetector',
            'PositionManager', 'ModuleOrchestrator', 'HealthMonitor',
        ]

        for name in patterns:
            status = self._safe_bus_get(smart_bus, f'{name}_status', None)
            health = self._safe_bus_get(smart_bus, f'{name}_health', None)
            if status is not None or health is not None:
                score = 85.0
                try:
                    if isinstance(health, dict) and 'score' in health:
                        score = float(health['score'])
                    elif isinstance(health, (int, float)):
                        score = float(health)
                except Exception:
                    pass
                modules[name] = {
                    'status': status or 'healthy',
                    'health_score': min(100.0, max(0.0, score))
                }
        return modules

# Export
__all__ = ['BeautifulTrainingVisualizer', 'Colors', 'Icons', 'BoxChars', 'TerminalUI', 'MiniChart']
