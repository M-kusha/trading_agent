#!/usr/bin/env python3
"""
Script to populate the module_registry.yaml with discovered modules from IntegrationValidator.
"""

from modules.monitoring.integration_validator import IntegrationValidator
import yaml
from pathlib import Path

def main():
    print("Populating module registry...")
    
    # Create the validator instance
    validator = IntegrationValidator()
    
    # Discover modules
    validator._discover_modules_recursive()
    
    # Build module registry data
    modules_registry = {}
    for class_name, info in validator.discovered_modules.items():
        # Infer category from file path or class name
        category = validator._infer_category(class_name, str(info['file_path']))
        
        modules_registry[class_name] = {
            'category': category,
            'provides': [],  # Placeholder - should be filled based on @module decorator or analysis
            'requires': [],  # Placeholder - should be filled based on @module decorator or analysis
            'file_path': str(info['file_path']),
            'module_path': info['module_path']
        }
    
    # Load existing registry or create new one
    registry_path = "config/module_registry.yaml"
    if Path(registry_path).exists():
        with open(registry_path, 'r', encoding='utf-8') as f:
            existing_registry = yaml.safe_load(f) or {}
    else:
        existing_registry = {}
    
    # Update the modules section
    existing_registry = {
        'modules': modules_registry,
        'version': '1.0.0',
        'last_updated': '2025-08-28'
    }
    
    # Save updated registry
    with open(registry_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing_registry, f, default_flow_style=False)
    
    print(f"Module registry updated with {len(modules_registry)} modules.")
    print(f"Saved to: {registry_path}")

if __name__ == "__main__":
    main()
