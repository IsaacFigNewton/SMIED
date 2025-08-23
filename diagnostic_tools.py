#!/usr/bin/env python3
"""
SMIED Semantic Pathfinding Diagnostic Tools

This module provides comprehensive analysis tools for debugging semantic pathfinding
issues in the SMIED system, particularly focusing on WordNet graph connectivity
and PairwiseBidirectionalAStar algorithm performance.

Usage:
    python diagnostic_tools.py --analyze-connectivity cat.n.01 chase.v.01 mouse.n.01
    python diagnostic_tools.py --audit-relations
    python diagnostic_tools.py --test-parameters
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
from nltk.corpus import wordnet as wn

# Import SMIED components
sys.path.append('src')
from smied.SemanticDecomposer import SemanticDecomposer
from smied.PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
from smied.BeamBuilder import BeamBuilder
from smied.EmbeddingHelper import EmbeddingHelper


class SMIEDConnectivityAnalyzer:
    """Comprehensive diagnostic analyzer for SMIED semantic pathfinding."""
    
    def __init__(self, verbosity: int = 1):
        """Initialize analyzer with SMIED components."""
        self.verbosity = verbosity
        self.semantic_decomposer = SemanticDecomposer(verbosity=verbosity)
        self.embedding_helper = EmbeddingHelper()
        self.beam_builder = BeamBuilder()
        
        # Build the WordNet graph
        self.graph = None
        self._build_wordnet_graph()
        
    def _build_wordnet_graph(self):
        """Build WordNet graph using SMIED's SemanticDecomposer."""
        print("[SMIEDConnectivityAnalyzer] Building WordNet graph...")
        
        # Get sample synsets to build graph (using common synsets for testing)
        test_synsets = [
            'cat.n.01', 'mouse.n.01', 'chase.v.01', 'hunt.v.01', 'predator.n.01',
            'prey.n.01', 'animal.n.01', 'mammal.n.01', 'feline.n.01', 'rodent.n.01',
            'catch.v.01', 'pursue.v.01', 'run.v.01', 'escape.v.01'
        ]
        
        synsets = []
        for synset_name in test_synsets:
            try:
                synset = wn.synset(synset_name)
                synsets.append(synset)
            except:
                print(f"Warning: Could not find synset {synset_name}")
        
        self.graph = self.semantic_decomposer.build_synset_graph(synsets)
        print(f"[SMIEDConnectivityAnalyzer] Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def analyze_synset_connectivity(self, synset1_name: str, synset2_name: str, max_hops: int = 6) -> Dict[str, Any]:
        """
        Analyze all possible paths between two synsets in the WordNet graph.
        
        Args:
            synset1_name: Source synset (e.g., 'cat.n.01')
            synset2_name: Target synset (e.g., 'chase.v.01')
            max_hops: Maximum number of hops to search
            
        Returns:
            Dict containing connectivity analysis results
        """
        print(f"\n=== Analyzing connectivity: {synset1_name} → {synset2_name} ===")
        
        if synset1_name not in self.graph:
            return {"error": f"Source synset {synset1_name} not found in graph"}
        if synset2_name not in self.graph:
            return {"error": f"Target synset {synset2_name} not found in graph"}
        
        results = {
            "source": synset1_name,
            "target": synset2_name,
            "max_hops": max_hops,
            "direct_connection": False,
            "paths": [],
            "relation_types_used": set(),
            "shortest_path_length": None,
            "connectivity_gaps": []
        }
        
        # Check direct connection
        if self.graph.has_edge(synset1_name, synset2_name):
            results["direct_connection"] = True
            edge_data = self.graph[synset1_name][synset2_name]
            results["direct_relation"] = edge_data.get("relation", "unknown")
            
        # Find all paths up to max_hops using BFS with path tracking
        paths_found = self._find_all_paths_bfs(synset1_name, synset2_name, max_hops)
        
        for path, relations in paths_found:
            results["paths"].append({
                "path": path,
                "length": len(path) - 1,
                "relations": relations
            })
            results["relation_types_used"].update(relations)
            
        results["relation_types_used"] = list(results["relation_types_used"])
        
        if results["paths"]:
            results["shortest_path_length"] = min(p["length"] for p in results["paths"])
        
        # Analyze connectivity gaps
        results["connectivity_gaps"] = self._analyze_connectivity_gaps(synset1_name, synset2_name)
        
        return results
    
    def _find_all_paths_bfs(self, source: str, target: str, max_hops: int) -> List[Tuple[List[str], List[str]]]:
        """Find all paths between source and target using BFS with path tracking."""
        paths_found = []
        queue = deque([(source, [source], [])])  # (current_node, path, relations)
        visited_paths = set()
        
        while queue:
            current, path, relations = queue.popleft()
            
            if len(path) - 1 >= max_hops:
                continue
                
            if current == target and len(path) > 1:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    paths_found.append((path, relations))
                continue
                
            for neighbor in self.graph.neighbors(current):
                if neighbor not in path:  # Avoid cycles
                    edge_data = self.graph[current][neighbor]
                    relation = edge_data.get("relation", "unknown")
                    new_path = path + [neighbor]
                    new_relations = relations + [relation]
                    queue.append((neighbor, new_path, new_relations))
        
        return paths_found
    
    def _analyze_connectivity_gaps(self, source: str, target: str) -> List[Dict[str, Any]]:
        """Analyze potential connectivity gaps between source and target."""
        gaps = []
        
        # Get synset objects for analysis
        try:
            source_synset = wn.synset(source)
            target_synset = wn.synset(target)
        except:
            return [{"error": "Could not analyze synsets"}]
        
        # Check for missing derivational relations (cross-POS connections)
        if source_synset.pos() != target_synset.pos():
            gaps.append({
                "type": "cross_pos_gap",
                "description": f"Cross-POS connection from {source_synset.pos()} to {target_synset.pos()}",
                "recommendation": "Check derivationally_related_forms() connections"
            })
        
        # Check outgoing relation density
        source_neighbors = list(self.graph.neighbors(source))
        target_neighbors = list(self.graph.neighbors(target))
        
        if len(source_neighbors) < 5:
            gaps.append({
                "type": "sparse_source",
                "description": f"Source synset has only {len(source_neighbors)} outgoing connections",
                "neighbors": source_neighbors
            })
            
        if len(target_neighbors) < 5:
            gaps.append({
                "type": "sparse_target", 
                "description": f"Target synset has only {len(target_neighbors)} incoming connections",
                "neighbors": target_neighbors
            })
        
        return gaps
    
    def audit_wordnet_relations(self) -> Dict[str, Any]:
        """
        Audit WordNet relation coverage in the graph construction.
        
        Returns comprehensive report on which relations are included/missing.
        """
        print("\n=== Auditing WordNet Relation Coverage ===")
        
        # Define all possible WordNet relations by POS
        all_relations = {
            'noun': [
                'hypernyms', 'hyponyms', 'member_holonyms', 'part_holonyms', 
                'substance_holonyms', 'member_meronyms', 'part_meronyms', 
                'substance_meronyms', 'similar_tos', 'attributes', 'also_sees'
            ],
            'verb': [
                'hypernyms', 'hyponyms', 'entailments', 'causes', 'similar_tos',
                'also_sees', 'verb_groups'
            ],
            'adjective': [
                'similar_tos', 'also_sees', 'attributes'
            ],
            'adverb': [
                'similar_tos', 'also_sees'
            ],
            'cross_pos': [
                'derivationally_related_forms', 'antonyms', 'pertainyms'
            ]
        }
        
        # Check which relations are implemented in SemanticDecomposer
        implemented_relations = {
            'hypernyms', 'hyponyms', 'member_holonyms', 'part_holonyms',
            'substance_holonyms', 'member_meronyms', 'part_meronyms',
            'substance_meronyms', 'similar_tos', 'also_sees', 'verb_groups',
            'entailments', 'causes'
        }
        
        audit_results = {
            "implemented_relations": list(implemented_relations),
            "missing_relations": [],
            "relation_coverage_by_pos": {},
            "edge_statistics": {},
            "recommendations": []
        }
        
        # Analyze coverage by POS
        for pos, relations in all_relations.items():
            if pos == 'cross_pos':
                missing = [r for r in relations if r not in implemented_relations]
                audit_results["missing_relations"].extend(missing)
            else:
                coverage = len([r for r in relations if r in implemented_relations]) / len(relations)
                audit_results["relation_coverage_by_pos"][pos] = {
                    "coverage_percentage": coverage * 100,
                    "implemented": [r for r in relations if r in implemented_relations],
                    "missing": [r for r in relations if r not in implemented_relations]
                }
        
        # Analyze edge statistics in current graph
        edge_relations = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            edge_relations[relation] += 1
            
        audit_results["edge_statistics"] = dict(edge_relations)
        
        # Generate recommendations
        if 'derivationally_related_forms' in audit_results["missing_relations"]:
            audit_results["recommendations"].append({
                "priority": "HIGH",
                "type": "missing_cross_pos_relations",
                "description": "derivationally_related_forms() missing - critical for noun↔verb connections",
                "impact": "Severely limits cross-POS pathfinding (e.g., cat.n.01 → chase.v.01)"
            })
        
        missing_attrs = [r for r in audit_results["missing_relations"] if 'attributes' in r]
        if missing_attrs:
            audit_results["recommendations"].append({
                "priority": "MEDIUM",
                "type": "missing_attribute_relations",
                "description": "attributes() relation missing - important for adjective connections",
                "impact": "Limits noun-adjective semantic bridging"
            })
        
        return audit_results
    
    def test_pathfinding_parameters(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Test PairwiseBidirectionalAStar with different parameter combinations.
        
        Args:
            test_cases: List of (source, target) synset pairs to test
            
        Returns:
            Results of parameter sensitivity analysis
        """
        print("\n=== Testing Pathfinding Parameters ===")
        
        parameter_sets = [
            {"beam_width": 3, "max_depth": 6, "relax_beam": False},
            {"beam_width": 5, "max_depth": 8, "relax_beam": False},
            {"beam_width": 7, "max_depth": 10, "relax_beam": False},
            {"beam_width": 3, "max_depth": 6, "relax_beam": True},   # No beam constraints
            {"beam_width": 10, "max_depth": 12, "relax_beam": False},
        ]
        
        results = {
            "test_cases": test_cases,
            "parameter_results": {},
            "success_rates": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        for i, params in enumerate(parameter_sets):
            param_key = f"config_{i+1}_{params}"
            results["parameter_results"][param_key] = {}
            successful_paths = 0
            total_time = 0
            
            print(f"\nTesting parameter set {i+1}: {params}")
            
            for source, target in test_cases:
                try:
                    # Create pathfinder with current parameters
                    pathfinder = PairwiseBidirectionalAStar(
                        g=self.graph,
                        src=source,
                        tgt=target,
                        get_new_beams_fn=self._get_beams_function,
                        **params
                    )
                    
                    # Find paths
                    import time
                    start_time = time.time()
                    paths = pathfinder.find_paths(max_results=3, len_tolerance=2)
                    end_time = time.time()
                    
                    total_time += (end_time - start_time)
                    
                    if paths:
                        successful_paths += 1
                        results["parameter_results"][param_key][f"{source}->{target}"] = {
                            "success": True,
                            "num_paths": len(paths),
                            "shortest_length": min(len(p[0])-1 for p in paths),
                            "best_cost": min(p[1] for p in paths),
                            "time_seconds": end_time - start_time
                        }
                    else:
                        results["parameter_results"][param_key][f"{source}->{target}"] = {
                            "success": False,
                            "time_seconds": end_time - start_time
                        }
                        
                except Exception as e:
                    results["parameter_results"][param_key][f"{source}->{target}"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            success_rate = successful_paths / len(test_cases) * 100
            avg_time = total_time / len(test_cases)
            
            results["success_rates"][param_key] = success_rate
            results["performance_metrics"][param_key] = {
                "average_time_seconds": avg_time,
                "success_rate_percent": success_rate
            }
        
        # Generate recommendations based on results
        best_config = max(results["success_rates"], key=results["success_rates"].get)
        best_rate = results["success_rates"][best_config]
        
        results["recommendations"].append({
            "type": "best_configuration",
            "config": best_config,
            "success_rate": best_rate,
            "description": f"Configuration {best_config} achieved highest success rate of {best_rate:.1f}%"
        })
        
        # Check if relaxed beam helps
        relaxed_configs = [k for k in results["success_rates"] if "relax_beam': True" in k]
        if relaxed_configs:
            relaxed_rate = max(results["success_rates"][k] for k in relaxed_configs)
            constrained_rate = max(results["success_rates"][k] for k in results["success_rates"] if k not in relaxed_configs)
            
            if relaxed_rate > constrained_rate + 10:  # 10% improvement
                results["recommendations"].append({
                    "type": "beam_constraint_issue",
                    "description": f"Relaxed beam constraints improve success rate by {relaxed_rate - constrained_rate:.1f}%",
                    "recommendation": "Consider less restrictive beam filtering or improved beam generation"
                })
        
        return results
    
    def _get_beams_function(self, g: nx.DiGraph, src: str, tgt: str):
        """Wrapper function to get embedding-based beams."""
        try:
            return self.beam_builder.get_new_beams(g, src, tgt)
        except Exception as e:
            print(f"Warning: Beam generation failed: {e}")
            return []
    
    def validate_heuristic_function(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Validate the effectiveness of embedding-based heuristics.
        
        Args:
            test_cases: List of (source, target) synset pairs to analyze
            
        Returns:
            Analysis of heuristic function performance
        """
        print("\n=== Validating Heuristic Function ===")
        
        results = {
            "test_cases": test_cases,
            "heuristic_analysis": {},
            "embedding_vs_wordnet": {},
            "recommendations": []
        }
        
        for source, target in test_cases:
            case_key = f"{source}->{target}"
            
            try:
                # Get embedding similarity
                try:
                    src_embedding = self.embedding_helper.get_embedding(source)
                    tgt_embedding = self.embedding_helper.get_embedding(target)
                    embedding_sim = self.embedding_helper.compute_similarity(src_embedding, tgt_embedding)
                    embedding_heuristic = max(0.0, 1.0 - float(embedding_sim))
                except Exception as e:
                    embedding_sim = 0.0
                    embedding_heuristic = 1.0
                
                # Check actual WordNet path existence
                wordnet_paths = self._find_all_paths_bfs(source, target, max_hops=6)
                has_wordnet_path = len(wordnet_paths) > 0
                shortest_wordnet_distance = min(len(p[0])-1 for p, _ in wordnet_paths) if wordnet_paths else float('inf')
                
                results["heuristic_analysis"][case_key] = {
                    "embedding_similarity": float(embedding_sim),
                    "embedding_heuristic": float(embedding_heuristic),
                    "has_wordnet_path": has_wordnet_path,
                    "shortest_wordnet_distance": shortest_wordnet_distance if shortest_wordnet_distance != float('inf') else None
                }
                
                # Analyze correlation
                if has_wordnet_path:
                    # High embedding similarity should correlate with short WordNet paths
                    if embedding_sim > 0.7 and shortest_wordnet_distance > 4:
                        results["embedding_vs_wordnet"][case_key] = "embedding_overestimates"
                    elif embedding_sim < 0.3 and shortest_wordnet_distance <= 3:
                        results["embedding_vs_wordnet"][case_key] = "embedding_underestimates"
                    else:
                        results["embedding_vs_wordnet"][case_key] = "consistent"
                else:
                    if embedding_sim > 0.5:
                        results["embedding_vs_wordnet"][case_key] = "false_positive"
                    else:
                        results["embedding_vs_wordnet"][case_key] = "true_negative"
                        
            except Exception as e:
                results["heuristic_analysis"][case_key] = {"error": str(e)}
        
        # Generate recommendations
        false_positives = len([v for v in results["embedding_vs_wordnet"].values() if v == "false_positive"])
        overestimates = len([v for v in results["embedding_vs_wordnet"].values() if v == "embedding_overestimates"])
        
        if false_positives > len(test_cases) * 0.3:  # More than 30% false positives
            results["recommendations"].append({
                "type": "embedding_false_positives",
                "description": f"{false_positives} out of {len(test_cases)} cases show high embedding similarity but no WordNet paths",
                "recommendation": "Consider WordNet-aware embedding training or fallback heuristics"
            })
        
        if overestimates > len(test_cases) * 0.3:
            results["recommendations"].append({
                "type": "embedding_overestimates",
                "description": f"{overestimates} cases show high embedding similarity but long WordNet paths",
                "recommendation": "Recalibrate embedding heuristic or add distance-based penalty"
            })
        
        return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="SMIED Semantic Pathfinding Diagnostics")
    parser.add_argument("--analyze-connectivity", nargs=2, metavar=("SOURCE", "TARGET"),
                       help="Analyze connectivity between two synsets")
    parser.add_argument("--audit-relations", action="store_true",
                       help="Audit WordNet relation coverage")
    parser.add_argument("--test-parameters", action="store_true", 
                       help="Test pathfinding with different parameters")
    parser.add_argument("--validate-heuristics", action="store_true",
                       help="Validate embedding-based heuristic function")
    parser.add_argument("--verbosity", type=int, default=1,
                       help="Verbosity level (0-2)")
    parser.add_argument("--output", type=str,
                       help="Output results to JSON file")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SMIEDConnectivityAnalyzer(verbosity=args.verbosity)
    
    results = {}
    
    # Standard test cases for comprehensive analysis
    standard_test_cases = [
        ("cat.n.01", "chase.v.01"),
        ("chase.v.01", "mouse.n.01"), 
        ("cat.n.01", "mouse.n.01"),
        ("predator.n.01", "hunt.v.01"),
        ("hunt.v.01", "prey.n.01")
    ]
    
    if args.analyze_connectivity:
        source, target = args.analyze_connectivity
        results["connectivity_analysis"] = analyzer.analyze_synset_connectivity(source, target)
        
    if args.audit_relations:
        results["relation_audit"] = analyzer.audit_wordnet_relations()
        
    if args.test_parameters:
        results["parameter_analysis"] = analyzer.test_pathfinding_parameters(standard_test_cases)
        
    if args.validate_heuristics:
        results["heuristic_validation"] = analyzer.validate_heuristic_function(standard_test_cases)
    
    # If no specific analysis requested, run comprehensive analysis
    if not any([args.analyze_connectivity, args.audit_relations, args.test_parameters, args.validate_heuristics]):
        print("Running comprehensive analysis...")
        results["relation_audit"] = analyzer.audit_wordnet_relations()
        
        for source, target in standard_test_cases:
            key = f"connectivity_{source}_to_{target}".replace(".", "_")
            results[key] = analyzer.analyze_synset_connectivity(source, target)
        
        results["parameter_analysis"] = analyzer.test_pathfinding_parameters(standard_test_cases)
        results["heuristic_validation"] = analyzer.validate_heuristic_function(standard_test_cases)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("DIAGNOSTIC RESULTS SUMMARY")
        print("="*80)
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()