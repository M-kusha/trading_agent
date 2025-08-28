import sys
sys.path.insert(0, 'C:\\Users\\Kushtrimi\\Desktop\\AI')

from modules.monitoring.dependency_inspector import DependencyInspector

def main():
    inspector = DependencyInspector()
    report = inspector.scan_once("Dependency Audit - System Health Check")
    
    with open('dependency_report.txt', 'w', encoding='utf-8') as f:
        f.write("DEPENDENCY AUDIT REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Title: {report.get('title', 'Unknown')}\n")
        f.write(f"Generated at: {report.get('generated_at', 'Unknown')}\n")
        f.write(f"Elapsed time: {report.get('elapsed_ms', 0)}ms\n\n")
        
        # Summary
        summary = report.get('summary', {})
        f.write("SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total providers: {summary.get('providers_total', 0)}\n")
        f.write(f"Total consumers: {summary.get('consumers_total', 0)}\n")
        f.write(f"Active keys: {summary.get('active_keys', 0)}\n")
        f.write(f"Cache hit rate: {summary.get('cache_hit_rate', 0.0) * 100:.1f}%\n")
        f.write(f"Disabled modules: {summary.get('disabled_modules', [])}\n\n")
        
        # Orphans
        orphans = report.get('orphans', [])
        if orphans:
            f.write("ORPHANED KEYS (consumed but not provided):\n")
            f.write("-" * 50 + "\n")
            for key, consumers in orphans:
                f.write(f"{key} ← consumers: {consumers}\n")
            f.write("\n")
        
        # Duplicate providers
        dups = report.get('duplicate_providers', [])
        if dups:
            f.write("DUPLICATE PROVIDERS:\n")
            f.write("-" * 30 + "\n")
            for key, providers in dups:
                f.write(f"{key} ← providers: {providers}\n")
            f.write("\n")
        
        # Critical keys
        critical = report.get('critical_keys', {})
        if critical:
            f.write("CRITICAL KEYS STATUS:\n")
            f.write("-" * 30 + "\n")
            for key, info in critical.items():
                f.write(f"{key}:\n")
                f.write(f"  Providers: {info.get('providers', [])}\n")
                f.write(f"  Consumers: {info.get('consumers', [])}\n\n")
        
        # Sample freshness
        sample = report.get('sample_fresh', [])
        if sample:
            f.write("SAMPLE FRESHNESS:\n")
            f.write("-" * 30 + "\n")
            for key, meta in sample:
                f.write(f"{key}: v{meta.get('version', '?')} from {meta.get('source', '?')} ")
                f.write(f"age={meta.get('age_seconds', 0):.1f}s conf={meta.get('confidence', 0):.2f}\n")
        
    print("Dependency report saved to dependency_report.txt")

if __name__ == "__main__":
    main()
