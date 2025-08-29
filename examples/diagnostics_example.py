#!/usr/bin/env python3
"""
Example usage of Diagnostics infrastructure.

This script demonstrates how to use the SMIEDDiagnostics class for analyzing
semantic pathfinding issues in the system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smied.Diagnostics import SMIEDDiagnostics
import json


def main():
    """Demonstrate SMIEDDiagnostics functionality."""
    print("Diagnostics Example")
    print("=" * 50)
    
    # Initialize diagnostics
    print("Initializing diagnostics...")
    diagnostics = SMIEDDiagnostics(verbosity=1)
    
    # 1. Test connectivity analysis
    print("\n1. Testing connectivity analysis:")
    print("   Analyzing path from 'cat.n.01' to 'chase.v.01'...")
    
    connectivity_result = diagnostics.analyze_synset_connectivity('cat.n.01', 'chase.v.01')
    
    if connectivity_result.get('error'):
        print(f"   Error: {connectivity_result['error']}")
    else:
        print(f"   Direct connection: {connectivity_result['direct_connection']}")
        print(f"   Paths found: {len(connectivity_result['paths'])}")
        if connectivity_result['paths']:
            print(f"   Shortest path length: {connectivity_result['shortest_path_length']}")
            # Show first path
            first_path = connectivity_result['paths'][0]
            print(f"   Example path: {' -> '.join(first_path['path'])}")
        else:
            print("   No paths found - this indicates a connectivity issue!")
    
    # 2. Audit WordNet relations
    print("\n2. Auditing WordNet relations:")
    audit_result = diagnostics.audit_wordnet_relations()
    
    print(f"   Implemented relations: {len(audit_result['implemented_relations'])}")
    print(f"   Missing relations: {len(audit_result['missing_relations'])}")
    print("   Missing critical relations:")
    for relation in audit_result['missing_relations']:
        print(f"     - {relation}")
    
    print("   Recommendations:")
    for rec in audit_result['recommendations']:
        print(f"     Priority {rec['priority']}: {rec['description']}")
    
    # 3. Test parameter sensitivity
    print("\n3. Testing parameter sensitivity:")
    test_cases = [
        ('cat.n.01', 'feline.n.01'),
        ('dog.n.01', 'canine.n.01')
    ]
    
    param_result = diagnostics.test_parameter_sensitivity(test_cases)
    
    best_config = None
    best_rate = 0
    for config, rate in param_result['success_rates'].items():
        print(f"   {config}: {rate:.1f}% success rate")
        if rate > best_rate:
            best_rate = rate
            best_config = config
    
    print(f"   Best configuration: {best_config} ({best_rate:.1f}% success)")
    
    # 4. Analyze graph topology
    print("\n4. Analyzing graph topology:")
    topology_result = diagnostics.analyze_graph_topology()
    
    stats = topology_result['basic_statistics']
    print(f"   Nodes: {stats['num_nodes']:,}")
    print(f"   Edges: {stats['num_edges']:,}")
    print(f"   Density: {stats['density']:.6f}")
    print(f"   Connected components: {topology_result['connectivity_analysis'].get('num_weak_components', 'N/A')}")
    
    # 5. Cross-POS connectivity analysis
    print("\n5. Analyzing cross-POS connectivity:")
    cross_pos_result = diagnostics.analyze_cross_pos_connectivity()
    
    cross_pos_stats = cross_pos_result['cross_pos_statistics']
    print(f"   Total edges: {cross_pos_stats['total_edges']:,}")
    print(f"   Cross-POS edges: {cross_pos_stats['cross_pos_edges']:,}")
    print(f"   Cross-POS percentage: {cross_pos_stats['cross_pos_percentage']:.2f}%")
    
    if cross_pos_result['recommendations']:
        print("   Recommendations:")
        for rec in cross_pos_result['recommendations']:
            print(f"     {rec['priority']}: {rec['description']}")
    
    # 6. Export comprehensive analysis
    print("\n6. Running comprehensive analysis and exporting results:")
    comprehensive_result = diagnostics.run_comprehensive_analysis()
    
    output_file = "smied_diagnostics_report.json"
    diagnostics.export_results(comprehensive_result, output_file)
    print(f"   Comprehensive analysis exported to: {output_file}")
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 50)
    
    # Key findings
    issues_found = []
    
    # Check for missing critical relations
    if 'derivationally_related_forms' in audit_result['missing_relations']:
        issues_found.append("Missing derivational relations (critical for cross-POS connections)")
    
    # Check cross-POS connectivity
    if cross_pos_stats['cross_pos_percentage'] < 10:
        issues_found.append(f"Low cross-POS connectivity ({cross_pos_stats['cross_pos_percentage']:.1f}%)")
    
    # Check connectivity failures
    if connectivity_result.get('paths') is not None and len(connectivity_result['paths']) == 0:
        issues_found.append("Critical pathfinding failure detected")
    
    if issues_found:
        print("🚨 Issues identified:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No critical issues detected")
    
    print(f"\nGraph statistics: {stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges")
    print(f"Best parameter configuration: {best_config} ({best_rate:.1f}% success rate)")
    print(f"Detailed report saved to: {output_file}")
    
    print("\nDiagnostics completed successfully!")


if __name__ == "__main__":
    main()