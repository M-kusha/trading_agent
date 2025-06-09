#!/bin/bash
gnome-terminal -- bash -c "cd ~/Documents/trading_agent && source venv/bin/activate && python3 train/train_sac.py; exec bash"
