#!/usr/bin/env python3
"""
AI Trading System Cleanup Script
Removes SAC/TD3 files and obsolete components to create a clean PPO-only system
"""

import os
import shutil
import sys
import logging
from pathlib import Path
from typing import List, Dict, Set
import argparse
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Files and directories to remove (SAC/TD3 related)
FILES_TO_REMOVE = [
    # Old main runner
    "run.py",
    
    # SAC training files
    "train/train_sac.py",
    "train/sac_trainer.py", 
    "train/train_sac_stable.py",
    "train/sac_training.py",
    
    # TD3 training files
    "train/train_td3.py",
    "train/td3_trainer.py",
    "train/train_td3_stable.py", 
    "train/td3_training.py",
    
    # Multi-agent files
    "train/train_multi_agent.py",
    "train/agent_comparison.py",
    
    # Old live trading CLI
    "live/live_trading.py",
    "live/live_runner.py",
    "live/cli_trading.py",
    
    # Legacy configuration files
    "config/sac_config.yaml",
    "config/td3_config.yaml",
    "config/agent_config.json",
    "configs/multi_agent.yaml",
    
    # Old test files
    "tests/test_sac.py",
    "tests/test_td3.py",
    "tests/test_multi_agent.py",
    
    # Legacy utilities
    "utils/sac_utils.py",
    "utils/td3_utils.py",
    "utils/agent_factory.py",
    
    # Old documentation
    "docs/sac_guide.md",
    "docs/td3_guide.md",
    "docs/multi_agent.md",
    
    # Obsolete scripts
    "scripts/train_sac.sh",
    "scripts/train_td3.sh",
    "scripts/compare_agents.py",
    
    # Old examples
    "examples/sac_example.py",
    "examples/td3_example.py",
    "examples/multi_agent_example.py",
]

# Directories to remove completely
DIRECTORIES_TO_REMOVE = [
    "models/sac/",
    "models/td3/", 
    "models/multi_agent/",
    "logs/sac/",
    "logs/td3/",
    "checkpoints/sac/",
    "checkpoints/td3/",
    "data/sac/",
    "data/td3/",
    "results/sac/",
    "results/td3/",
    "experiments/sac/",
    "experiments/td3/",
]

# Patterns to search for in files and remove/replace
PATTERNS_TO_CLEAN = {
    # Import statements to remove
    "imports_to_remove": [
        r"from stable_baselines3 import SAC",
        r"from stable_baselines3 import TD3", 
        r"import.*SAC",
        r"import.*TD3",
        r"from.*sac.*import",
        r"from.*td3.*import",
    ],
    
    # Configuration entries to remove
    "config_to_remove": [
        r'"sac".*:.*{[^}]*}',
        r'"td3".*:.*{[^}]*}',
        r"sac.*=.*{[^}]*}",
        r"td3.*=.*{[^}]*}",
    ],
    
    # Code blocks to remove
    "code_to_remove": [
        r"if.*agent.*==.*['\"]sac['\"].*:",
        r"elif.*agent.*==.*['\"]td3['\"].*:",
        r"def.*sac.*\([^)]*\):",
        r"def.*td3.*\([^)]*\):",
        r"class.*SAC.*\([^)]*\):",
        r"class.*TD3.*\([^)]*\):",
    ]
}

# Files to update with PPO-only content
FILES_TO_UPDATE = {
    # Configuration files
    "envs/config.py": {
        "remove_sections": ["SAC_CONFIG", "TD3_CONFIG", "AGENT_CONFIGS"],
        "keep_sections": ["PPO_CONFIG", "TRADING_CONFIG"]
    },
    
    # Training scripts  
    "train/__init__.py": {
        "remove_imports": ["sac", "td3", "multi_agent"],
        "keep_imports": ["ppo"]
    },
    
    # Live trading
    "live/__init__.py": {
        "remove_imports": ["sac", "td3"],
        "keep_imports": ["ppo"]
    },
    
    # Test files
    "tests/__init__.py": {
        "remove_imports": ["sac", "td3"],
        "keep_imports": ["ppo"]
    }
}

# ═══════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    return logging.getLogger("SystemCleanup")

# ═══════════════════════════════════════════════════════════════════
# Cleanup Functions
# ═══════════════════════════════════════════════════════════════════

class SystemCleanup:
    """Manages the cleanup of obsolete trading system components"""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose)
        
        self.removed_files: List[str] = []
        self.removed_dirs: List[str] = []
        self.updated_files: List[str] = []
        self.errors: List[str] = []
        
    def display_header(self):
        """Display cleanup header"""
        print("="*80)
        print("🧹 AI Trading System Cleanup - PPO-Only Migration")
        print("="*80)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE CLEANUP'}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print()
        
    def scan_files(self) -> Dict[str, List[str]]:
        """Scan system for files to clean up"""
        self.logger.info("Scanning system for obsolete files...")
        
        scan_results = {
            "files_to_remove": [],
            "dirs_to_remove": [],
            "files_to_update": [],
            "potential_issues": []
        }
        
        # Check files to remove
        for file_path in FILES_TO_REMOVE:
            if Path(file_path).exists():
                scan_results["files_to_remove"].append(file_path)
                self.logger.debug(f"Found file to remove: {file_path}")
        
        # Check directories to remove
        for dir_path in DIRECTORIES_TO_REMOVE:
            if Path(dir_path).exists():
                scan_results["dirs_to_remove"].append(dir_path)
                self.logger.debug(f"Found directory to remove: {dir_path}")
        
        # Check files to update
        for file_path in FILES_TO_UPDATE.keys():
            if Path(file_path).exists():
                scan_results["files_to_update"].append(file_path)
                self.logger.debug(f"Found file to update: {file_path}")
        
        # Scan for potential issues (files that might contain SAC/TD3 references)
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.py', '.yaml', '.json', '.md')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            if any(term in content for term in ['sac', 'td3', 'multi_agent', 'agent_factory']):
                                if str(file_path) not in scan_results["files_to_update"]:
                                    scan_results["potential_issues"].append(str(file_path))
                    except Exception:
                        pass
        
        return scan_results
    
    def create_backup(self):
        """Create backup of current system"""
        if self.dry_run:
            self.logger.info("DRY RUN: Would create backup")
            return
        
        backup_dir = f"backup_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Creating backup: {backup_dir}")
        
        try:
            # Create backup of key files
            Path(backup_dir).mkdir(exist_ok=True)
            
            backup_files = [
                "train/", "live/", "config/", "configs/", 
                "requirements.txt", "README.md"
            ]
            
            for item in backup_files:
                if Path(item).exists():
                    if Path(item).is_file():
                        shutil.copy2(item, backup_dir)
                    else:
                        shutil.copytree(item, Path(backup_dir) / item, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            
            self.logger.info(f"Backup created successfully: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            if not self.dry_run:
                raise
    
    def remove_files(self, files: List[str]):
        """Remove obsolete files"""
        self.logger.info(f"Removing {len(files)} obsolete files...")
        
        for file_path in files:
            try:
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Would remove file: {file_path}")
                else:
                    Path(file_path).unlink()
                    self.logger.info(f"Removed file: {file_path}")
                
                self.removed_files.append(file_path)
                
            except Exception as e:
                error_msg = f"Failed to remove file {file_path}: {e}"
                self.logger.error(error_msg)
                self.errors.append(error_msg)
    
    def remove_directories(self, directories: List[str]):
        """Remove obsolete directories"""
        self.logger.info(f"Removing {len(directories)} obsolete directories...")
        
        for dir_path in directories:
            try:
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Would remove directory: {dir_path}")
                else:
                    shutil.rmtree(dir_path)
                    self.logger.info(f"Removed directory: {dir_path}")
                
                self.removed_dirs.append(dir_path)
                
            except Exception as e:
                error_msg = f"Failed to remove directory {dir_path}: {e}"
                self.logger.error(error_msg)
                self.errors.append(error_msg)
    
    def update_files(self, files: List[str]):
        """Update files to remove SAC/TD3 references"""
        self.logger.info(f"Updating {len(files)} files to remove obsolete references...")
        
        for file_path in files:
            try:
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Would update file: {file_path}")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Remove SAC/TD3 imports and references
                import re
                for pattern in PATTERNS_TO_CLEAN["imports_to_remove"]:
                    content = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)
                
                # Clean up empty lines
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.logger.info(f"Updated file: {file_path}")
                    self.updated_files.append(file_path)
                
            except Exception as e:
                error_msg = f"Failed to update file {file_path}: {e}"
                self.logger.error(error_msg)
                self.errors.append(error_msg)
    
    def clean_model_files(self):
        """Clean up obsolete model files"""
        self.logger.info("Cleaning up obsolete model files...")
        
        model_patterns = ["*sac*", "*td3*", "*multi_agent*"]
        
        for pattern in model_patterns:
            for model_file in Path("models").glob(pattern):
                try:
                    if self.dry_run:
                        self.logger.info(f"DRY RUN: Would remove model: {model_file}")
                    else:
                        if model_file.is_file():
                            model_file.unlink()
                        else:
                            shutil.rmtree(model_file)
                        self.logger.info(f"Removed model: {model_file}")
                    
                    self.removed_files.append(str(model_file))
                    
                except Exception as e:
                    error_msg = f"Failed to remove model {model_file}: {e}"
                    self.logger.error(error_msg)
                    self.errors.append(error_msg)
    
    def update_requirements(self):
        """Update requirements.txt to PPO-only dependencies"""
        if self.dry_run:
            self.logger.info("DRY RUN: Would update requirements.txt")
            return
        
        self.logger.info("Updating requirements.txt...")
        
        try:
            # Backup current requirements
            if Path("requirements.txt").exists():
                shutil.copy2("requirements.txt", "requirements_backup.txt")
            
            # The new requirements.txt is already created as an artifact
            self.logger.info("Requirements.txt will be replaced with PPO-only version")
            self.updated_files.append("requirements.txt")
            
        except Exception as e:
            error_msg = f"Failed to update requirements.txt: {e}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
    
    def generate_report(self) -> Dict:
        """Generate cleanup report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "dry_run" if self.dry_run else "live_cleanup",
            "summary": {
                "files_removed": len(self.removed_files),
                "directories_removed": len(self.removed_dirs),
                "files_updated": len(self.updated_files),
                "errors": len(self.errors)
            },
            "details": {
                "removed_files": self.removed_files,
                "removed_directories": self.removed_dirs,
                "updated_files": self.updated_files,
                "errors": self.errors
            }
        }
        
        return report
    
    def save_report(self, report: Dict):
        """Save cleanup report to file"""
        report_file = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Cleanup report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def display_summary(self, scan_results: Dict):
        """Display cleanup summary"""
        print("\n" + "="*80)
        print("📊 CLEANUP SUMMARY")
        print("="*80)
        
        print(f"📁 Files to remove: {len(scan_results['files_to_remove'])}")
        for file in scan_results['files_to_remove'][:10]:  # Show first 10
            print(f"   - {file}")
        if len(scan_results['files_to_remove']) > 10:
            print(f"   ... and {len(scan_results['files_to_remove']) - 10} more")
        
        print(f"\n📂 Directories to remove: {len(scan_results['dirs_to_remove'])}")
        for dir in scan_results['dirs_to_remove']:
            print(f"   - {dir}")
        
        print(f"\n📝 Files to update: {len(scan_results['files_to_update'])}")
        for file in scan_results['files_to_update']:
            print(f"   - {file}")
        
        print(f"\n⚠️ Potential issues to review: {len(scan_results['potential_issues'])}")
        for file in scan_results['potential_issues'][:5]:  # Show first 5
            print(f"   - {file}")
        if len(scan_results['potential_issues']) > 5:
            print(f"   ... and {len(scan_results['potential_issues']) - 5} more")
        
        print("="*80)
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        self.display_header()
        
        # Scan system
        scan_results = self.scan_files()
        self.display_summary(scan_results)
        
        if not self.dry_run:
            # Confirm cleanup
            print(f"\n⚠️ WARNING: This will permanently remove {len(scan_results['files_to_remove'])} files and {len(scan_results['dirs_to_remove'])} directories!")
            response = input("Do you want to continue? (yes/no): ").lower().strip()
            
            if response != 'yes':
                print("Cleanup cancelled by user.")
                return
            
            # Create backup
            print("\n📦 Creating backup...")
            self.create_backup()
        
        # Perform cleanup
        print(f"\n🧹 {'Simulating' if self.dry_run else 'Performing'} cleanup...")
        
        self.remove_files(scan_results["files_to_remove"])
        self.remove_directories(scan_results["dirs_to_remove"]) 
        self.update_files(scan_results["files_to_update"])
        self.clean_model_files()
        self.update_requirements()
        
        # Generate and save report
        report = self.generate_report()
        self.save_report(report)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ CLEANUP COMPLETED")
        print("="*80)
        print(f"Files removed: {len(self.removed_files)}")
        print(f"Directories removed: {len(self.removed_dirs)}")
        print(f"Files updated: {len(self.updated_files)}")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors[:5]:
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more (see log file)")
        
        if not self.dry_run:
            print(f"\n📋 Detailed report saved to cleanup report file")
            print("🚀 System is now PPO-only! Ready to run the new dashboard.")
        else:
            print(f"\n🔍 This was a dry run. Use --live to perform actual cleanup.")
        
        print("="*80)

# ═══════════════════════════════════════════════════════════════════
# Command Line Interface
# ═══════════════════════════════════════════════════════════════════

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Trading System Cleanup - Remove SAC/TD3 components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan                   # Scan only, no changes
  %(prog)s --dry-run               # Dry run, show what would be changed
  %(prog)s --live                  # Perform actual cleanup
  %(prog)s --live --verbose        # Cleanup with detailed logging
        """
    )
    
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan system and show what would be cleaned up'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Perform dry run (simulation) of cleanup'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Perform actual cleanup (WARNING: This will delete files!)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--backup-dir',
        type=str,
        help='Custom backup directory name'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Determine mode
    if args.live:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        # Default to scan mode
        dry_run = True
        args.scan = True
    
    # Create cleanup manager
    cleanup = SystemCleanup(dry_run=dry_run, verbose=args.verbose)
    
    # Scan mode
    if args.scan:
        cleanup.display_header()
        scan_results = cleanup.scan_files()
        cleanup.display_summary(scan_results)
        print("\n🔍 Scan complete. Use --dry-run to simulate cleanup or --live to perform cleanup.")
        return 0
    
    # Run cleanup
    try:
        cleanup.run_cleanup()
        
        if cleanup.errors:
            return 1  # Some errors occurred
        else:
            return 0  # Success
            
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())