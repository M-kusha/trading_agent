# ─────────────────────────────────────────────────────────────
# File: modules/core/state_manager.py
# 🚀 State management for hot-reload support in SmartInfoBus
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import pickle
import importlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import traceback

from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_orchestrator import ModuleOrchestrator
    from modules.core.core import BaseModule


class StateManager:
    """
    Manages module state for hot-reload functionality.
    Preserves state across module reloads and system restarts.
    """
    
    def __init__(self, state_dir: str = "state/modules"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory state cache
        self.state_cache: Dict[str, Any] = {}
        
        # State versioning
        self.state_versions: Dict[str, int] = {}
        
        # Setup logging
        self.logger = RotatingLogger(
            name="StateManager",
            log_path="logs/state/state_manager.log",
            max_lines=2000,
            operator_mode=True
        )
    
    def save_module_state(self, module: BaseModule) -> bytes:
        """
        Save module state to bytes for hot-reload.
        
        Args:
            module: Module instance to save state from
            
        Returns:
            Serialized state as bytes
        """
        module_name = module.__class__.__name__
        
        try:
            # Get state from module
            if hasattr(module, 'get_state'):
                state = module.get_state()
            else:
                # Extract all non-method attributes
                state = {
                    k: v for k, v in module.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            
            # Add metadata
            state_with_metadata = {
                'module_name': module_name,
                'timestamp': datetime.now().isoformat(),
                'version': self.state_versions.get(module_name, 0) + 1,
                'state': state
            }
            
            # Update version
            self.state_versions[module_name] = state_with_metadata['version']
            
            # Cache in memory
            self.state_cache[module_name] = state_with_metadata
            
            # Serialize
            serialized = pickle.dumps(state_with_metadata)
            
            # Also save to disk for persistence
            self._save_to_disk(module_name, state_with_metadata)
            
            self.logger.info(
                format_operator_message(
                    "💾", "STATE SAVED",
                    instrument=module_name,
                    details=f"Version {state_with_metadata['version']}",
                    context="state_management"
                )
            )
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Failed to save state for {module_name}: {e}")
            raise
    
    def reload_module(self, module_name: str, 
                     orchestrator: ModuleOrchestrator) -> bool:
        """
        Hot-reload a module preserving its state.
        
        Args:
            module_name: Name of module to reload
            orchestrator: Orchestrator instance managing modules
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current module
            current_module = orchestrator.modules.get(module_name)
            if not current_module:
                self.logger.error(f"Module {module_name} not found")
                return False
            
            # Save current state
            state_bytes = self.save_module_state(current_module)
            
            # Get module path
            module_path = current_module.__module__
            
            # Reload module code
            self.logger.info(f"Reloading module code for {module_name}")
            module_ref = importlib.import_module(module_path)
            importlib.reload(module_ref)
            
            # Get new class
            new_class = getattr(module_ref, module_name, None)
            if not new_class:
                self.logger.error(f"Class {module_name} not found after reload")
                return False
            
            # Verify it has module metadata
            if not hasattr(new_class, '__module_metadata__'):
                self.logger.error(f"Reloaded class missing @module decorator")
                return False
            
            # Create new instance
            new_instance = new_class()
            
            # Restore state
            state_data = pickle.loads(state_bytes)
            if hasattr(new_instance, 'set_state'):
                new_instance.set_state(state_data['state'])
            else:
                # Manually restore attributes
                for key, value in state_data['state'].items():
                    if hasattr(new_instance, key):
                        setattr(new_instance, key, value)
            
            # Replace in orchestrator
            orchestrator.modules[module_name] = new_instance
            
            # Update metadata if changed
            orchestrator.metadata[module_name] = new_class.__module_metadata__
            
            self.logger.info(
                format_operator_message(
                    "✅", "MODULE RELOADED",
                    instrument=module_name,
                    details=f"State restored from v{state_data['version']}",
                    context="hot_reload"
                )
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to reload {module_name}: {e}\n"
                f"{traceback.format_exc()}"
            )
            return False
    
    def _save_to_disk(self, module_name: str, state_data: Dict[str, Any]):
        """Save state to disk for persistence"""
        try:
            # Use JSON for readability (with pickle fallback for complex objects)
            file_path = self.state_dir / f"{module_name}_state.json"
            
            # Try JSON first
            try:
                with open(file_path, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
            except (TypeError, ValueError):
                # Fall back to pickle for complex objects
                file_path = self.state_dir / f"{module_name}_state.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(state_data, f)
                    
        except Exception as e:
            self.logger.error(f"Failed to save state to disk: {e}")
    
    def load_from_disk(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Load state from disk"""
        # Try JSON first
        json_path = self.state_dir / f"{module_name}_state.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Try pickle
        pkl_path = self.state_dir / f"{module_name}_state.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return None
    
    def restore_all_states(self, orchestrator: ModuleOrchestrator) -> Dict[str, bool]:
        """
        Restore states for all modules from disk.
        Called on system startup.
        
        Returns:
            Dict mapping module names to success status
        """
        results = {}
        
        # Find all state files
        state_files = list(self.state_dir.glob("*_state.json")) + \
                     list(self.state_dir.glob("*_state.pkl"))
        
        for state_file in state_files:
            # Extract module name
            module_name = state_file.stem.replace('_state', '')
            
            # Skip if module not in orchestrator
            if module_name not in orchestrator.modules:
                continue
            
            # Load state
            state_data = self.load_from_disk(module_name)
            if not state_data:
                results[module_name] = False
                continue
            
            # Restore state
            try:
                module = orchestrator.modules[module_name]
                if hasattr(module, 'set_state'):
                    module.set_state(state_data['state'])
                    results[module_name] = True
                    
                    self.logger.info(
                        f"Restored state for {module_name} "
                        f"(v{state_data.get('version', 0)})"
                    )
                else:
                    results[module_name] = False
                    
            except Exception as e:
                self.logger.error(f"Failed to restore {module_name}: {e}")
                results[module_name] = False
        
        # Summary
        success_count = sum(1 for v in results.values() if v)
        self.logger.info(
            format_operator_message(
                "📂", "STATE RESTORATION",
                details=f"Restored {success_count}/{len(results)} modules",
                context="startup"
            )
        )
        
        return results
    
    def create_checkpoint(self, orchestrator: ModuleOrchestrator, 
                         checkpoint_name: str = "manual") -> bool:
        """
        Create a checkpoint of all module states.
        
        Args:
            orchestrator: Orchestrator with modules
            checkpoint_name: Name for the checkpoint
            
        Returns:
            True if successful
        """
        checkpoint_dir = self.state_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'modules': {}
        }
        
        success_count = 0
        
        for module_name, module in orchestrator.modules.items():
            try:
                # Save module state
                if hasattr(module, 'get_state'):
                    state = module.get_state()
                    checkpoint_data['modules'][module_name] = state
                    
                    # Save individual file
                    module_file = checkpoint_dir / f"{module_name}.json"
                    with open(module_file, 'w') as f:
                        json.dump(state, f, indent=2, default=str)
                    
                    success_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to checkpoint {module_name}: {e}")
        
        # Save checkpoint metadata
        metadata_file = checkpoint_dir / "checkpoint.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'name': checkpoint_name,
                'timestamp': checkpoint_data['timestamp'],
                'module_count': len(checkpoint_data['modules']),
                'modules': list(checkpoint_data['modules'].keys())
            }, f, indent=2)
        
        self.logger.info(
            format_operator_message(
                "📸", "CHECKPOINT CREATED",
                instrument=checkpoint_name,
                details=f"Saved {success_count} modules",
                context="state_management"
            )
        )
        
        return success_count > 0
    
    def restore_checkpoint(self, orchestrator: ModuleOrchestrator,
                          checkpoint_name: str) -> bool:
        """
        Restore all modules from a checkpoint.
        
        Args:
            orchestrator: Orchestrator with modules
            checkpoint_name: Name of checkpoint to restore
            
        Returns:
            True if successful
        """
        checkpoint_dir = self.state_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_dir.exists():
            self.logger.error(f"Checkpoint {checkpoint_name} not found")
            return False
        
        # Load checkpoint metadata
        metadata_file = checkpoint_dir / "checkpoint.json"
        if not metadata_file.exists():
            self.logger.error("Checkpoint metadata not found")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(
            f"Restoring checkpoint '{checkpoint_name}' "
            f"from {metadata['timestamp']}"
        )
        
        success_count = 0
        
        # Restore each module
        for module_name in metadata['modules']:
            module_file = checkpoint_dir / f"{module_name}.json"
            
            if not module_file.exists():
                continue
            
            if module_name not in orchestrator.modules:
                continue
            
            try:
                with open(module_file, 'r') as f:
                    state = json.load(f)
                
                module = orchestrator.modules[module_name]
                if hasattr(module, 'set_state'):
                    module.set_state(state)
                    success_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to restore {module_name}: {e}")
        
        self.logger.info(
            format_operator_message(
                "✅", "CHECKPOINT RESTORED",
                instrument=checkpoint_name,
                details=f"Restored {success_count}/{len(metadata['modules'])} modules",
                context="state_management"
            )
        )
        
        return success_count > 0
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        checkpoints_dir = self.state_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return []
        
        checkpoints = []
        
        for checkpoint_dir in checkpoints_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_file = checkpoint_dir / "checkpoint.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            checkpoints.append(metadata)
                    except:
                        pass
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
    
    def cleanup_old_states(self, days_to_keep: int = 7):
        """Clean up old state files"""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 3600)
        
        cleaned_count = 0
        
        for state_file in self.state_dir.glob("*_state.*"):
            if state_file.stat().st_mtime < cutoff_time:
                try:
                    state_file.unlink()
                    cleaned_count += 1
                except:
                    pass
        
        if cleaned_count > 0:
            self.logger.info(
                f"Cleaned up {cleaned_count} old state files"
            )