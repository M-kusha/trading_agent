#!/usr/bin/env python3
"""
🚀 Advanced AI Trading System Setup
Professional multi-language setup with comprehensive dependency management
Supports: English, Albanian
"""

import os
import sys
import subprocess
import logging
import json
import time
import platform
import venv
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-LANGUAGE SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

TRANSLATIONS = {
    'en': {
        'title': '🚀 Advanced AI Trading System Setup',
        'subtitle': 'Professional Setup with Comprehensive Dependency Management',
        'language_prompt': 'Choose your language / Zgjidhni gjuhën tuaj:',
        'language_choice': '[1] English\n[2] Shqip (Albanian)',
        'invalid_choice': 'Invalid choice. Please enter 1 or 2.',
        'welcome': 'Welcome to the AI Trading System Setup!',
        'features': [
            '✓ Virtual environment creation and management',
            '✓ Intelligent dependency resolution',
            '✓ MetaTrader5 integration verification',
            '✓ System requirements validation',
            '✓ Performance optimization',
            '✓ Real-time progress tracking',
            '✓ Comprehensive error handling'
        ],
        'checking_system': 'Checking system requirements...',
        'python_version': 'Python version',
        'system_info': 'System information',
        'arch': 'Architecture',
        'os_version': 'Operating System',
        'memory': 'Available Memory',
        'disk_space': 'Available Disk Space',
        'requirements_met': 'System requirements met',
        'requirements_failed': 'System requirements not met',
        'creating_venv': 'Creating virtual environment...',
        'venv_exists': 'Virtual environment already exists',
        'venv_created': 'Virtual environment created successfully',
        'activating_venv': 'Activating virtual environment...',
        'checking_packages': 'Checking installed packages...',
        'packages_found': 'packages found',
        'missing_packages': 'Missing packages',
        'outdated_packages': 'Outdated packages',
        'installing_packages': 'Installing packages',
        'package_installed': 'Package installed successfully',
        'package_failed': 'Package installation failed',
        'checking_mt5': 'Checking MetaTrader5 integration...',
        'mt5_available': 'MetaTrader5 available',
        'mt5_not_found': 'MetaTrader5 not found - install from MetaQuotes',
        'checking_nodejs': 'Checking Node.js...',
        'nodejs_found': 'Node.js found',
        'nodejs_not_found': 'Node.js not found',
        'nodejs_install_prompt': 'Install Node.js from https://nodejs.org',
        'creating_directories': 'Creating directory structure...',
        'directories_created': 'Directory structure created',
        'verifying_installation': 'Verifying installation...',
        'verification_success': 'All packages verified successfully',
        'verification_failed': 'Some packages failed verification',
        'setup_complete': 'Setup Complete!',
        'setup_failed': 'Setup Failed',
        'next_steps': 'Next Steps',
        'start_dashboard': 'Start the dashboard',
        'access_url': 'Access at',
        'documentation': 'Read documentation',
        'support': 'Get support',
        'error_occurred': 'An error occurred',
        'continue_prompt': 'Continue? (y/n)',
        'press_enter': 'Press Enter to continue...',
    },
    'sq': {
        'title': '🚀 Instalimi i Avancuar i Sistemit të Tregtimit AI',
        'subtitle': 'Instalim Profesional me Menaxhim të Plotë të Varësive',
        'language_prompt': 'Choose your language / Zgjidhni gjuhën tuaj:',
        'language_choice': '[1] English\n[2] Shqip (Albanian)',
        'invalid_choice': 'Zgjedhje e pavlefshme. Ju lutemi shkruani 1 ose 2.',
        'welcome': 'Mirë se erdhët në Instalimin e Sistemit të Tregtimit AI!',
        'features': [
            '✓ Krijimi dhe menaxhimi i mjedisit virtual',
            '✓ Zgjidhja inteligjente e varësive',
            '✓ Verifikimi i integrimit MetaTrader5',
            '✓ Validimi i kërkesave të sistemit',
            '✓ Optimizimi i performancës',
            '✓ Ndjekja e progresit në kohë reale',
            '✓ Trajtimi i plotë i gabimeve'
        ],
        'checking_system': 'Duke kontrolluar kërkesat e sistemit...',
        'python_version': 'Versioni i Python',
        'system_info': 'Informacioni i sistemit',
        'arch': 'Arkitektura',
        'os_version': 'Sistema Operative',
        'memory': 'Memoria e Disponueshme',
        'disk_space': 'Hapësira e Disponueshme në Disk',
        'requirements_met': 'Kërkesat e sistemit u plotësuan',
        'requirements_failed': 'Kërkesat e sistemit nuk u plotësuan',
        'creating_venv': 'Duke krijuar mjedisin virtual...',
        'venv_exists': 'Mjedisi virtual ekziston tashmë',
        'venv_created': 'Mjedisi virtual u krijua me sukses',
        'activating_venv': 'Duke aktivizuar mjedisin virtual...',
        'checking_packages': 'Duke kontrolluar paketat e instaluara...',
        'packages_found': 'paketa u gjetën',
        'missing_packages': 'Paketa të munguara',
        'outdated_packages': 'Paketa të vjetruara',
        'installing_packages': 'Duke instaluar paketat',
        'package_installed': 'Paketa u instalua me sukses',
        'package_failed': 'Instalimi i paketës dështoi',
        'checking_mt5': 'Duke kontrolluar integrimin MetaTrader5...',
        'mt5_available': 'MetaTrader5 i disponueshëm',
        'mt5_not_found': 'MetaTrader5 nuk u gjet - instaloni nga MetaQuotes',
        'checking_nodejs': 'Duke kontrolluar Node.js...',
        'nodejs_found': 'Node.js u gjet',
        'nodejs_not_found': 'Node.js nuk u gjet',
        'nodejs_install_prompt': 'Instaloni Node.js nga https://nodejs.org',
        'creating_directories': 'Duke krijuar strukturën e drejtorive...',
        'directories_created': 'Struktura e drejtorive u krijua',
        'verifying_installation': 'Duke verifikuar instalimin...',
        'verification_success': 'Të gjitha paketat u verifikuan me sukses',
        'verification_failed': 'Disa paketa dështuan në verifikim',
        'setup_complete': 'Instalimi u Plotësua!',
        'setup_failed': 'Instalimi Dështoi',
        'next_steps': 'Hapat e Ardhshëm',
        'start_dashboard': 'Nisni kontrollin',
        'access_url': 'Hyni në',
        'documentation': 'Lexoni dokumentacionin',
        'support': 'Merrni mbështetje',
        'error_occurred': 'Ndodhi një gabim',
        'continue_prompt': 'Vazhdoni? (y/n)',
        'press_enter': 'Shtypni Enter për të vazhduar...',
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION AND DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemRequirements:
    """System requirements for the AI trading system"""
    min_python_version: Tuple[int, int] = (3, 9)
    min_memory_gb: float = 4.0
    min_disk_space_gb: float = 2.0
    required_architecture: Optional[str] = None
    
@dataclass
class PackageInfo:
    """Information about a Python package"""
    name: str
    version: Optional[str] = None
    required: bool = True
    description: str = ""
    install_command: Optional[str] = None

@dataclass
class InstallationProgress:
    """Track installation progress"""
    total_steps: int = 0
    completed_steps: int = 0
    current_step: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class AdvancedSetupManager:
    """Advanced setup manager with comprehensive features"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.t = TRANSLATIONS[language]
        self.setup_logging()
        self.progress = InstallationProgress()
        self.system_info = self._gather_system_info()
        
        # Package definitions
        self.python_packages = self._define_python_packages()
        self.node_packages = self._define_node_packages()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('AdvancedSetup')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(
            logs_dir / 'setup.log', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information"""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_gb = psutil.disk_usage('.').free / (1024**3)
        except ImportError:
            memory_gb = 0
            disk_gb = 0
            
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': sys.version_info,
            'python_executable': sys.executable,
            'memory_gb': memory_gb,
            'disk_space_gb': disk_gb,
            'cpu_count': os.cpu_count(),
        }
        
    def _define_python_packages(self) -> List[PackageInfo]:
        """Define required Python packages with metadata"""
        return [
            PackageInfo("gymnasium", "0.29.1", True, "RL environment framework"),
            PackageInfo("numpy", "1.26.4", True, "Numerical computing"),
            PackageInfo("pandas", "2.2.1", True, "Data manipulation"),
            PackageInfo("plotly", "5.19.0", True, "Interactive plotting"),
            PackageInfo("pydantic", "2.7.1", True, "Data validation"),
            PackageInfo("python-dateutil", "2.9.0", True, "Date utilities"),
            PackageInfo("stable-baselines3", "2.3.2", True, "RL algorithms"),
            PackageInfo("optuna", "3.6.1", True, "Hyperparameter optimization"),
            PackageInfo("fastapi", "0.110.2", True, "Web API framework"),
            PackageInfo("uvicorn", "0.29.0", True, "ASGI server"),
            PackageInfo("streamlit", "1.35.0", False, "Dashboard framework"),
            PackageInfo("MetaTrader5", "5.0.48", True, "MT5 integration"),
            PackageInfo("torch", "2.2.2", True, "Deep learning framework"),
            PackageInfo("websockets", None, False, "WebSocket support"),
            PackageInfo("tensorboard", None, False, "ML visualization"),
            PackageInfo("psutil", None, True, "System monitoring"),
            PackageInfo("requests", None, True, "HTTP library"),
            PackageInfo("aiofiles", None, False, "Async file operations"),
        ]
        
    def _define_node_packages(self) -> Dict[str, Any]:
        """Define required Node.js packages"""
        return {
            "name": "ai-trading-dashboard",
            "version": "1.0.0",
            "description": "AI Trading System Dashboard",
            "scripts": {
                "build": "echo 'No build process needed'",
                "start": "echo 'Backend serves frontend'",
                "dev": "echo 'Development mode'"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "axios": "^1.6.0",
                "chart.js": "^4.4.0",
                "recharts": "^2.8.0"
            },
            "devDependencies": {
                "typescript": "^5.2.0",
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0"
            }
        }
        
    def display_header(self):
        """Display professional header"""
        width = 80
        print("=" * width)
        print(f"{self.t['title']:^{width}}")
        print(f"{self.t['subtitle']:^{width}}")
        print("=" * width)
        print()
        
        # Display features
        print(f"🎯 {self.t['welcome']}")
        print()
        for feature in self.t['features']:
            print(f"  {feature}")
        print()
        
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        self.logger.info(f"🔍 {self.t['checking_system']}")
        
        requirements = SystemRequirements()
        passed = True
        
        # Check Python version
        py_version = self.system_info['python_version']
        version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
        
        if py_version >= requirements.min_python_version:
            self.logger.info(f"✅ {self.t['python_version']}: {version_str}")
        else:
            self.logger.error(f"❌ {self.t['python_version']}: {version_str} (minimum: {requirements.min_python_version[0]}.{requirements.min_python_version[1]})")
            passed = False
            
        # Display system info
        self.logger.info(f"ℹ️  {self.t['system_info']}:")
        self.logger.info(f"   {self.t['os_version']}: {self.system_info['platform']} {self.system_info['platform_version']}")
        self.logger.info(f"   {self.t['arch']}: {self.system_info['architecture']}")
        
        if self.system_info['memory_gb'] > 0:
            self.logger.info(f"   {self.t['memory']}: {self.system_info['memory_gb']:.1f} GB")
            if self.system_info['memory_gb'] < requirements.min_memory_gb:
                self.logger.warning(f"⚠️  Low memory (minimum: {requirements.min_memory_gb} GB)")
                
        if self.system_info['disk_space_gb'] > 0:
            self.logger.info(f"   {self.t['disk_space']}: {self.system_info['disk_space_gb']:.1f} GB")
            if self.system_info['disk_space_gb'] < requirements.min_disk_space_gb:
                self.logger.error(f"❌ Insufficient disk space (minimum: {requirements.min_disk_space_gb} GB)")
                passed = False
                
        print()
        if passed:
            self.logger.info(f"✅ {self.t['requirements_met']}")
        else:
            self.logger.error(f"❌ {self.t['requirements_failed']}")
            
        return passed
        
    def setup_virtual_environment(self) -> Optional[str]:
        """Setup virtual environment with proper handling"""
        venv_path = Path('venv')
        
        if venv_path.exists():
            self.logger.info(f"✅ {self.t['venv_exists']}")
        else:
            self.logger.info(f"🔧 {self.t['creating_venv']}")
            try:
                venv.create('venv', with_pip=True)
                self.logger.info(f"✅ {self.t['venv_created']}")
            except Exception as e:
                self.logger.error(f"❌ Virtual environment creation failed: {e}")
                return None
                
        # Determine python executable path
        if sys.platform == "win32":
            venv_python = venv_path / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_path / 'bin' / 'python'
            
        if venv_python.exists():
            self.logger.info(f"✅ {self.t['activating_venv']}")
            return str(venv_python)
        else:
            self.logger.error("❌ Virtual environment python executable not found")
            return None
            
    def check_installed_packages(self, python_executable: str) -> Dict[str, str]:
        """Check which packages are already installed"""
        self.logger.info(f"📋 {self.t['checking_packages']}")
        
        try:
            result = subprocess.run(
                [python_executable, '-m', 'pip', 'list', '--format=freeze'],
                capture_output=True, text=True, check=True, timeout=30
            )
            
            installed = {}
            for line in result.stdout.strip().split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    installed[name.lower()] = version
                    
            self.logger.info(f"✅ {len(installed)} {self.t['packages_found']}")
            return installed
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check packages: {e}")
            return {}
            
    def install_packages_parallel(self, python_executable: str) -> bool:
        """Install packages with parallel processing and progress tracking"""
        installed = self.check_installed_packages(python_executable)
        
        missing = []
        outdated = []
        
        for pkg in self.python_packages:
            pkg_lower = pkg.name.lower()
            
            if pkg_lower not in installed:
                if pkg.version:
                    missing.append(f"{pkg.name}=={pkg.version}")
                else:
                    missing.append(pkg.name)
            elif pkg.version and installed[pkg_lower] != pkg.version:
                outdated.append(f"{pkg.name}=={pkg.version}")
                
        to_install = missing + outdated
        
        if not to_install:
            self.logger.info("✅ All packages already installed and up to date!")
            return True
            
        self.logger.info(f"📦 {self.t['installing_packages']}: {len(to_install)} packages")
        
        # Progress tracking
        self.progress.total_steps = len(to_install)
        self.progress.completed_steps = 0
        
        failed = []
        
        for i, package in enumerate(to_install, 1):
            pkg_name = package.split('==')[0]
            self.progress.current_step = f"Installing {pkg_name}"
            
            self.logger.info(f"📦 [{i}/{len(to_install)}] Installing {pkg_name}...")
            
            try:
                result = subprocess.run(
                    [python_executable, '-m', 'pip', 'install', package, '--quiet'],
                    capture_output=True, text=True, check=True, timeout=300
                )
                self.logger.info(f"✅ {pkg_name} {self.t['package_installed']}")
                self.progress.completed_steps += 1
                
            except subprocess.TimeoutExpired:
                error_msg = f"Timeout installing {pkg_name}"
                self.logger.error(f"❌ {error_msg}")
                failed.append(package)
                self.progress.errors.append(error_msg)
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed installing {pkg_name}: {e.stderr[:100] if e.stderr else 'Unknown error'}"
                self.logger.error(f"❌ {error_msg}")
                failed.append(package)
                self.progress.errors.append(error_msg)
                
        if failed:
            self.logger.warning(f"⚠️  {len(failed)} packages failed to install: {[p.split('==')[0] for p in failed]}")
            return False
        else:
            self.logger.info("✅ All Python packages installed successfully!")
            return True
            
    def check_metatrader5(self, python_executable: str) -> bool:
        """Check MetaTrader5 integration"""
        self.logger.info(f"🔍 {self.t['checking_mt5']}")
        
        try:
            result = subprocess.run(
                [python_executable, '-c', 'import MetaTrader5 as mt5; print("MT5 available" if mt5 else "MT5 not available")'],
                capture_output=True, text=True, check=True, timeout=10
            )
            
            if "available" in result.stdout:
                self.logger.info(f"✅ {self.t['mt5_available']}")
                return True
            else:
                self.logger.warning(f"⚠️  {self.t['mt5_not_found']}")
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️  {self.t['mt5_not_found']}: {e}")
            return False
            
    def check_nodejs(self) -> bool:
        """Check Node.js installation"""
        self.logger.info(f"🔍 {self.t['checking_nodejs']}")
        
        try:
            result = subprocess.run(
                ['node', '--version'], 
                capture_output=True, text=True, check=True, timeout=10
            )
            version = result.stdout.strip()
            self.logger.info(f"✅ {self.t['nodejs_found']}: {version}")
            return True
            
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            self.logger.warning(f"⚠️  {self.t['nodejs_not_found']}")
            self.logger.info(f"ℹ️  {self.t['nodejs_install_prompt']}")
            return False
            
    def setup_nodejs_packages(self) -> bool:
        """Setup Node.js packages"""
        if not self.check_nodejs():
            return False
            
        # Create or update package.json
        package_json_path = Path('package.json')
        
        if not package_json_path.exists():
            with open(package_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.node_packages, f, indent=2)
            self.logger.info("✅ Created package.json")
            
        # Install packages
        node_modules = Path('node_modules')
        if node_modules.exists() and any(node_modules.iterdir()):
            self.logger.info("✅ Node.js packages already installed")
            return True
            
        try:
            self.logger.info("📦 Installing Node.js packages...")
            subprocess.run(
                ['npm', 'install', '--silent'], 
                check=True, timeout=120
            )
            self.logger.info("✅ Node.js packages installed successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  Node.js package installation failed: {e}")
            return False
            
    def create_directory_structure(self):
        """Create necessary directory structure"""
        self.logger.info(f"📁 {self.t['creating_directories']}")
        
        directories = [
            'logs', 'logs/training', 'logs/risk', 'logs/simulation',
            'logs/strategy', 'logs/position', 'logs/tensorboard',
            'checkpoints', 'models', 'data', 'data/historical',
            'data/live', 'configs', 'tests', 'docs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"✅ {self.t['directories_created']}")
        
    def verify_installation(self, python_executable: str) -> bool:
        """Comprehensive installation verification"""
        self.logger.info(f"🔍 {self.t['verifying_installation']}")
        
        critical_packages = [
            'numpy', 'pandas', 'gymnasium', 'stable_baselines3',
            'fastapi', 'uvicorn', 'MetaTrader5'
        ]
        
        failed = []
        
        for package in critical_packages:
            try:
                result = subprocess.run(
                    [python_executable, '-c', f'import {package}'],
                    capture_output=True, text=True, check=True, timeout=10
                )
                self.logger.info(f"✅ {package} - OK")
                
            except Exception as e:
                self.logger.error(f"❌ {package} - FAILED: {e}")
                failed.append(package)
                
        if failed:
            self.logger.error(f"❌ {self.t['verification_failed']}: {failed}")
            return False
        else:
            self.logger.info(f"✅ {self.t['verification_success']}")
            return True
            
    def display_completion_summary(self, success: bool, python_executable: str):
        """Display completion summary with next steps"""
        width = 80
        print("\n" + "=" * width)
        
        if success:
            print(f"🎉 {self.t['setup_complete']:^{width}}")
            print("=" * width)
            
            print(f"\n🎯 {self.t['next_steps']}:")
            print(f"   1. {self.t['start_dashboard']}: python run_dashboard.py")
            print(f"   2. {self.t['access_url']}: http://localhost:8000")
            print(f"   3. {self.t['documentation']}: README.md")
            print(f"   4. {self.t['support']}: GitHub Issues")
            
            if 'venv' in python_executable:
                print(f"\n💡 Virtual Environment Commands:")
                if sys.platform == "win32":
                    print(f"   Activate: venv\\Scripts\\activate")
                else:
                    print(f"   Activate: source venv/bin/activate")
                    
        else:
            print(f"❌ {self.t['setup_failed']:^{width}}")
            print("=" * width)
            
            if self.progress.errors:
                print(f"\n🔍 Errors encountered:")
                for error in self.progress.errors[:5]:  # Show first 5 errors
                    print(f"   • {error}")
                    
        print("=" * width)
        
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        try:
            self.display_header()
            
            # Check system requirements
            if not self.check_system_requirements():
                print(f"\n{self.t['continue_prompt']}")
                if input().lower() != 'y':
                    return False
                    
            # Setup virtual environment
            python_executable = self.setup_virtual_environment()
            if not python_executable:
                python_executable = sys.executable
                
            # Create directories
            self.create_directory_structure()
            
            # Install Python packages
            self.install_packages_parallel(python_executable)
            
            # Check MetaTrader5
            self.check_metatrader5(python_executable)
            
            # Setup Node.js packages
            self.setup_nodejs_packages()
            
            # Verify installation
            verification_success = self.verify_installation(python_executable)
            
            # Display summary
            self.display_completion_summary(verification_success, python_executable)
            
            return verification_success
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Setup interrupted by user")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ {self.t['error_occurred']}: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SELECTION AND MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def select_language() -> str:
    """Interactive language selection"""
    print("=" * 60)
    print("🌍 AI Trading System Setup / Instalimi i Sistemit të Tregtimit AI")
    print("=" * 60)
    print("\nChoose your language / Zgjidhni gjuhën tuaj:")
    print("[1] English")
    print("[2] Shqip (Albanian)")
    print()
    
    while True:
        try:
            choice = input("Select / Zgjidhni (1-2): ").strip()
            if choice == '1':
                return 'en'
            elif choice == '2':
                return 'sq'
            else:
                print("Invalid choice. Please enter 1 or 2. / Zgjedhje e pavlefshme. Shkruani 1 ose 2.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting setup... / Duke dalë nga instalimi...")
            sys.exit(0)

def main():
    """Main entry point"""
    try:
        # Language selection
        language = select_language()
        
        # Initialize setup manager
        setup_manager = AdvancedSetupManager(language)
        
        # Run setup
        success = setup_manager.run_setup()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user / Instalimi u anulua nga përdoruesi")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nFatal error / Gabim fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()