import os

# Define base folder
base_dir = "modules/market_1"

# Folder and file structure
structure = {
    "__init__.py": "",
    "market_module.py": "# Main orchestrator\n",
    "components": {
        "__init__.py": "",
        "fractal_regime.py": "# Fractal analysis component\n",
        "liquidity_heatmap.py": "# Liquidity neural network component\n",
        "theme_detector.py": "# ML clustering component\n",
        "regime_matrix.py": "# Performance matrix component\n",
        "time_risk.py": "# Time-aware risk component\n",
    },
    "shared": {
        "__init__.py": "",
        "base_component.py": "# Base class for all components\n",
        "data_extractors.py": "# Unified data extraction logic\n",
        "state_manager.py": "# Centralized state management\n",
        "metrics_tracker.py": "# Unified metrics/performance tracking\n",
        "circuit_breaker.py": "# Shared circuit breaker logic\n",
    },
    "debug": {
        "__init__.py": "",
        "trace_logger.py": "# Advanced trace logging\n",
        "diagnostics.py": "# Real-time diagnostics\n",
        "visualizer.py": "# Debug visualization tools\n",
    },
}

def create_structure(base, struct):
    os.makedirs(base, exist_ok=True)
    for name, content in struct.items():
        path = os.path.join(base, name)
        if isinstance(content, dict):
            create_structure(path, content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# Run the creation
create_structure(base_dir, structure)

print(f"✅ Created folder structure under {base_dir}")
