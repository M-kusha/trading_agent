# Agent Ops

- Config: Centralized in `config/app_config.yaml` with typed models in `config/models.py`. Load via `config.load_app_config` / `config.build_trading_config` for training and live.
- Logging: Call `config.setup_logging` once and use `config.get_logger` everywhere; toggle verbosity with `logging.debug` flag in config.
- Training entrypoint: `python train/train_ppo_hybrid.py` (uses central config + shared observation schema).
- Live entrypoint: `python start_live_trading.py` (pulls instruments, risk, and MT5 credentials from central config).
- Smoke tests: `pytest -q tests/smoke/test_smoke_system.py` (env build/steps + live shell dry-run).
