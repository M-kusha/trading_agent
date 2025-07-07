# ─────────────────────────────────────────────────────────────
# File: modules/core/state_manager.py
# 🚀 State Manager for hot-reload and persistence
# ─────────────────────────────────────────────────────────────

import pickle
import importlib
import json
import inspect
from typing import Dict, Any, List, Optional, Tuple, Type
from pathlib import Path
import logging
import datetime
import traceback

from modules.utils.audit_utils import format_operator_message, RotatingLogger
from modules.core.core import BaseModule


class StateManager:
    """
    Manages module state for hot-reload, persistence, and recovery.
    Enables zero-downtime module updates.
    """
    
    def __init__(self, state_dir: str = "states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
        # State versioning
        self.state_versions = {}
        
        # Setup logging
        self.logger = RotatingLogger(
            name="StateManager",
            log_path="logs/state/state_manager.log",
            max_lines=2000,
            operator_mode=True
        )
    
    @staticmethod
    def save_module_state(module: BaseModule) -> bytes:
        """Serialize module state for persistence"""
        try:
            # Use module's get_state method if available
            if hasattr(module, 'get_state'):
                state = module.get_state()
            else:
                # Extract all non-method attributes
                state = {
                    k: v for k, v in module.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            
            # Add metadata
            state_with_meta = {
                'module_class': module.__class__.__name__,
                'module_path': module.__class__.__module__,
                'timestamp': datetime.datetime.now().isoformat(),
                'version': getattr(module, 'version', '1.0.0'),
                'state': state
            }
            
            return pickle.dumps(state_with_meta)
            
        except Exception as e:
            logging.error(f"Failed to save state for {module.__class__.__name__}: {e}")
            raise
    
    @staticmethod
    def reload_module(module_name: str, orchestrator: Any) -> bool:
        """Hot-reload a module preserving state"""
        logger = logging.getLogger("StateManager")
        
        try:
            # Get current module
            current_module = orchestrator.modules.get(module_name)
            if not current_module:
                logger.error(f"Module {module_name} not found")
                return False
            
            # Save current state
            state_data = StateManager.save_module_state(current_module)
            state_dict = pickle.loads(state_data)
            
            # Get module path
            module_path = current_module.__class__.__module__
            
            # Reload module code
            module_ref = importlib.import_module(module_path)
            importlib.reload(module_ref)
            
            # Get new class
            new_class = getattr(module_ref, module_name)
            
            # Verify it has required metadata
            if not hasattr(new_class, '__module_metadata__'):
                logger.error(f"Reloaded {module_name} missing @module decorator")
                return False
            
            # Create new instance
            new_instance = new_class()
            
            # Restore state
            if hasattr(new_instance, 'set_state'):
                new_instance.set_state(state_dict['state'])
            else:
                # Manual restoration
                for key, value in state_dict['state'].items():
                    if hasattr(new_instance, key):
                        try:
                            setattr(new_instance, key, value)
                        except:
                            pass
            
            # Replace in orchestrator
            orchestrator.modules[module_name] = new_instance
            
            # Update metadata if changed
            orchestrator.metadata[module_name] = new_class.__module_metadata__
            
            logger.info(
                format_operator_message(
                    "🔄", "HOT-RELOAD SUCCESS",
                    instrument=module_name,
                    details=f"State preserved",
                    context="hot_reload"
                )
            )
            
            return True
            
        except Exception as e:
            logger.error(
                format_operator_message(
                    "❌", "HOT-RELOAD FAILED",
                    instrument=module_name,
                    details=str(e),
                    context="hot_reload"
                )
            )
            logger.error(traceback.format_exc())
            return False
    
    def save_all_states(self, orchestrator: Any, checkpoint_name: str = "latest"):
        """Save all module states to disk"""
        checkpoint_dir = self.state_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        failed_modules = []
        
        for module_name, module in orchestrator.modules.items():
            try:
                # Save state
                state_data = self.save_module_state(module)
                
                # Write to file
                state_file = checkpoint_dir / f"{module_name}.state"
                with open(state_file, 'wb') as f:
                    f.write(state_data)
                
                saved_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to save {module_name}: {e}")
                failed_modules.append(module_name)
        
        # Save checkpoint metadata
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'saved_modules': saved_count,
            'failed_modules': failed_modules,
            'orchestrator_state': {
                'execution_order': orchestrator.execution_order,
                'voting_members': orchestrator.voting_members
            }
        }
        
        with open(checkpoint_dir / 'checkpoint.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(
            format_operator_message(
                "💾", "CHECKPOINT SAVED",
                details=f"{saved_count} modules",
                result=f"Failed: {len(failed_modules)}",
                context="persistence"
            )
        )
    
    def load_all_states(self, orchestrator: Any, checkpoint_name: str = "latest"):
        """Load all module states from disk"""
        checkpoint_dir = self.state_dir / checkpoint_name
        
        if not checkpoint_dir.exists():
            self.logger.error(f"Checkpoint {checkpoint_name} not found")
            return
        
        # Load metadata
        with open(checkpoint_dir / 'checkpoint.json', 'r') as f:
            metadata = json.load(f)
        
        loaded_count = 0
        failed_modules = []
        
        # Load each module state
        for state_file in checkpoint_dir.glob("*.state"):
            module_name = state_file.stem
            
            try:
                # Read state
                with open(state_file, 'rb') as f:
                    state_data = f.read()
                
                state_dict = pickle.loads(state_data)
                
                # Find module in orchestrator
                if module_name in orchestrator.modules:
                    module = orchestrator.modules[module_name]
                    
                    # Restore state
                    if hasattr(module, 'set_state'):
                        module.set_state(state_dict['state'])
                        loaded_count += 1
                    else:
                        self.logger.warning(
                            f"Module {module_name} doesn't support set_state"
                        )
                else:
                    self.logger.warning(f"Module {module_name} not found in orchestrator")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {module_name}: {e}")
                failed_modules.append(module_name)
        
        self.logger.info(
            format_operator_message(
                "📥", "CHECKPOINT LOADED",
                details=f"{loaded_count} modules",
                result=f"Failed: {len(failed_modules)}",
                context="persistence"
            )
        )
    
    def create_module_snapshot(self, module: BaseModule) -> Dict[str, Any]:
        """Create a snapshot of module state for debugging"""
        snapshot = {
            'timestamp': datetime.datetime.now().isoformat(),
            'module_info': {
                'class': module.__class__.__name__,
                'version': getattr(module, 'version', 'unknown'),
                'health_status': getattr(module, '_health_status', 'unknown'),
                'step_count': getattr(module, '_step_count', 0)
            }
        }
        
        # Get state if available
        if hasattr(module, 'get_state'):
            try:
                snapshot['state'] = module.get_state()
            except:
                snapshot['state'] = {'error': 'Failed to get state'}
        
        # Get performance metrics
        if hasattr(module, 'get_health_status'):
            try:
                snapshot['health'] = module.get_health_status()
            except:
                snapshot['health'] = {'error': 'Failed to get health status'}
        
        return snapshot
    
    def diff_module_states(self, module_name: str, state1: Dict[str, Any], 
                          state2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two module states to identify changes"""
        diff = {
            'module': module_name,
            'changes': {},
            'added_keys': [],
            'removed_keys': [],
            'modified_values': {}
        }
        
        state1_data = state1.get('state', {})
        state2_data = state2.get('state', {})
        
        # Find added/removed keys
        keys1 = set(state1_data.keys())
        keys2 = set(state2_data.keys())
        
        diff['added_keys'] = list(keys2 - keys1)
        diff['removed_keys'] = list(keys1 - keys2)
        
        # Find modified values
        for key in keys1.intersection(keys2):
            val1 = state1_data[key]
            val2 = state2_data[key]
            
            # Simple comparison (could be enhanced)
            try:
                if val1 != val2:
                    diff['modified_values'][key] = {
                        'old': str(val1)[:100],  # Truncate for display
                        'new': str(val2)[:100]
                    }
            except:
                # Some values might not be comparable
                diff['modified_values'][key] = {
                    'old': f"<{type(val1).__name__}>",
                    'new': f"<{type(val2).__name__}>"
                }
        
        return diff
    
    def validate_state_compatibility(self, module_class: Type[BaseModule], 
                                    state_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate if a saved state is compatible with current module version"""
        issues = []
        
        # Check version compatibility
        saved_version = state_data.get('version', '0.0.0')
        current_version = getattr(module_class, '__module_metadata__', {}).get('version', '1.0.0')
        
        if saved_version != current_version:
            issues.append(f"Version mismatch: saved={saved_version}, current={current_version}")
        
        # Check required state fields
        if hasattr(module_class, 'REQUIRED_STATE_FIELDS'):
            required = module_class.REQUIRED_STATE_FIELDS
            saved_keys = set(state_data.get('state', {}).keys())
            
            missing = set(required) - saved_keys
            if missing:
                issues.append(f"Missing required fields: {missing}")
        
        # Check state structure
        if hasattr(module_class, 'validate_state'):
            try:
                validation_result = module_class.validate_state(state_data['state'])
                if not validation_result:
                    issues.append("State validation failed")
            except Exception as e:
                issues.append(f"State validation error: {e}")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    def create_state_backup(self, orchestrator: Any):
        """Create automatic backup before major operations"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        
        self.save_all_states(orchestrator, backup_name)
        
        # Keep only last 10 backups
        self._cleanup_old_backups(keep_last=10)
    
    def _cleanup_old_backups(self, keep_last: int = 10):
        """Remove old backup directories"""
        backup_dirs = sorted([
            d for d in self.state_dir.iterdir() 
            if d.is_dir() and d.name.startswith("backup_")
        ])
        
        if len(backup_dirs) > keep_last:
            for old_dir in backup_dirs[:-keep_last]:
                try:
                    import shutil
                    shutil.rmtree(old_dir)
                    self.logger.debug(f"Removed old backup: {old_dir.name}")
                except Exception as e:
                    self.logger.error(f"Failed to remove {old_dir}: {e}")