#!/usr/bin/env python3
"""
Script to run the IntegrationValidator and save the report to a file.
"""

from modules.monitoring.integration_validator import IntegrationValidator
import json
from pathlib import Path

def main():
    print("Starting SmartInfoBus integration validation...")
    
    # Create the validator instance
    validator = IntegrationValidator()
    
    # Run the validation
    report = validator.validate_system()
    
    # Print the plain English report
    print("\n" + "="*80)
    print(report.to_plain_english())
    print("="*80)
    
    # Save detailed report to JSON file
    output_path = "logs/validation_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'report': {
                'score': report.integration_score,
                'total_modules': report.total_modules,
                'validated_modules': report.validated_modules,
                'issues': [vars(i) for i in report.issues],
                'missing_decorators': report.missing_decorators,
                'missing_thesis': report.missing_thesis,
                'legacy_modules': report.legacy_modules,
                'config_issues': report.config_issues,
                'bad_categories': report.bad_categories
            },
            'plain_english': report.to_plain_english()
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {output_path}")
    
    # Also save the plain English version
    txt_path = "logs/validation_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report.to_plain_english())
    
    print(f"Plain English report saved to: {txt_path}")

if __name__ == "__main__":
    main()
