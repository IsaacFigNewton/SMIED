#!/usr/bin/env python3
"""
SMIED Diagnostics Module

Comprehensive diagnostic toolkit for SMIED semantic pathfinding analysis.
This module consolidates all diagnostic tools previously scattered across
multiple files into a unified, well-structured API.

Classes:
    SMIEDDiagnostics: Main diagnostic class with comprehensive analysis methods

Example:
    >>> diagnostics = SMIEDDiagnostics(verbosity=1)
    >>> result = diagnostics.analyze_synset_connectivity('cat.n.01', 'chase.v.01')
    >>> audit = diagnostics.audit_wordnet_relations()
"""

import json
import sys
import time
import argparse
import itertools
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import networkx as nx
from nltk.corpus import wordnet as wn

# Import SMIED components
try:
    from .SemanticDecomposer import SemanticDecomposer
    from .PairwiseBidirectionalAStar import PairwiseBidirectionalAStar
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from smied.SemanticDecomposer import SemanticDecomposer
    from smied.PairwiseBidirectionalAStar import PairwiseBidirectionalAStar


class SMIEDDiagnostics:
    """
    Comprehensive diagnostic toolkit for SMIED semantic pathfinding analysis.
    
    This class consolidates diagnostic functionality for debugging and analyzing
    SMIED's semantic pathfinding capabilities, particularly focusing on WordNet
    graph connectivity and algorithm performance.
    
    Attributes:
        verbosity (int): Verbosity level (0=quiet, 1=normal, 2=verbose)
        semantic_decomposer (SemanticDecomposer): SMIED semantic decomposer
        graph (nx.DiGraph): Current WordNet graph instance
    """
    
    def __init__(self, verbosity: int = 1, nlp_func=None, embedding_model=None):
        """
        Initialize diagnostics with SMIED components.
        
        Args:
            verbosity (int): Verbosity level for output (0-2)
            nlp_func: Optional spaCy NLP function override
            embedding_model: Optional embedding model override
        """
        self.verbosity = verbosity
        
        # Initialize SMIED components
        try:
            if nlp_func is None:
                import spacy
                nlp_func = spacy.load('en_core_web_sm')
        except (ImportError, OSError):
            if self.verbosity >= 1:
                print("Warning: spaCy model not available, using mock NLP function")
            nlp_func = self._create_mock_nlp()
        
        self.semantic_decomposer = SemanticDecomposer(
            wn_module=wn, 
            nlp_func=nlp_func, 
            embedding_model=embedding_model,
            verbosity=verbosity
        )
        
        # Build default graph
        self.graph = None
        self._build_default_graph()
    
    def _create_mock_nlp(self):
        """Create mock NLP function for testing without spaCy."""
        def mock_nlp(text):
            class MockToken:
                def __init__(self, text):
                    self.text = text
                    self.lemma_ = text.lower()
                    self.pos_ = 'NOUN'
            class MockDoc:
                def __init__(self, text):
                    self.tokens = [MockToken(t) for t in text.split()]
                def __iter__(self):
                    return iter(self.tokens)
            return MockDoc(text)
        return mock_nlp
    
    def _build_default_graph(self):
        """Build default WordNet graph for testing."""
        if self.verbosity >= 1:
            print("[SMIEDDiagnostics] Building default WordNet graph...")
        
        # Use common test synsets for efficient graph building
        test_synset_names = [
            'cat.n.01', 'mouse.n.01', 'chase.v.01', 'hunt.v.01', 'predator.n.01',
            'prey.n.01', 'animal.n.01', 'mammal.n.01', 'feline.n.01', 'rodent.n.01',
            'catch.v.01', 'pursue.v.01', 'run.v.01', 'escape.v.01', 'hunter.n.01',
            'dog.n.01', 'bird.n.01', 'fish.n.01', 'carnivore.n.01'
        ]
        
        synsets = []
        for name in test_synset_names:
            try:
                synset = wn.synset(name)
                synsets.append(synset)
            except:
                if self.verbosity >= 2:
                    print(f"Warning: Could not find synset {name}")
        
        self.graph = self.semantic_decomposer.build_synset_graph()
        
        if self.verbosity >= 1:
            print(f"[SMIEDDiagnostics] Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    # ========================================
    # Core Analysis Methods
    # ========================================
    
    def analyze_synset_connectivity(self, synset1_name: str, synset2_name: str, 
                                  max_hops: int = 6) -> Dict[str, Any]:
        """
        Analyze all possible paths between two synsets in the WordNet graph.
        
        Args:
            synset1_name: Source synset (e.g., 'cat.n.01')
            synset2_name: Target synset (e.g., 'chase.v.01')
            max_hops: Maximum number of hops to search
            
        Returns:
            Dict containing comprehensive connectivity analysis results
        """
        if self.verbosity >= 1:
            print(f"\n=== Analyzing connectivity: {synset1_name} -> {synset2_name} ===")
        
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
        
        # Find all paths up to max_hops
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
    
    def audit_wordnet_relations(self) -> Dict[str, Any]:
        """
        Audit WordNet relation coverage in the graph construction.
        
        Returns comprehensive report on which relations are included/missing.
        """
        if self.verbosity >= 1:
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
                "description": "derivationally_related_forms() missing - critical for noun<->verb connections",
                "impact": "Severely limits cross-POS pathfinding (e.g., cat.n.01 -> chase.v.01)"
            })
        
        if 'attributes' in audit_results["missing_relations"]:
            audit_results["recommendations"].append({
                "priority": "MEDIUM",
                "type": "missing_attribute_relations",
                "description": "attributes() relation missing - important for adjective connections",
                "impact": "Limits noun-adjective semantic bridging"
            })
        
        return audit_results
    
    def test_parameter_sensitivity(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Test PairwiseBidirectionalAStar with different parameter combinations.
        
        Args:
            test_cases: List of (source, target) synset pairs to test
            
        Returns:
            Results of parameter sensitivity analysis
        """
        if self.verbosity >= 1:
            print("\n=== Testing Pathfinding Parameters ===")
        
        parameter_sets = [
            {"beam_width": 3, "max_depth": 6, "relax_beam": False},   # Current defaults
            {"beam_width": 5, "max_depth": 8, "relax_beam": False},   # Moderate relaxation
            {"beam_width": 7, "max_depth": 10, "relax_beam": False},  # More relaxed
            {"beam_width": 3, "max_depth": 6, "relax_beam": True},    # No beam constraints
            {"beam_width": 10, "max_depth": 12, "relax_beam": False}, # Very relaxed
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
            
            if self.verbosity >= 1:
                print(f"\nTesting parameter set {i+1}: {params}")
            
            for source, target in test_cases:
                if source not in self.graph or target not in self.graph:
                    results["parameter_results"][param_key][f"{source}->{target}"] = {
                        "success": False,
                        "error": "Node not in graph"
                    }
                    continue
                
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
            avg_time = total_time / len(test_cases) if len(test_cases) > 0 else 0
            
            results["success_rates"][param_key] = success_rate
            results["performance_metrics"][param_key] = {
                "average_time_seconds": avg_time,
                "success_rate_percent": success_rate
            }
        
        # Generate recommendations
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
            constrained_configs = [k for k in results["success_rates"] if k not in relaxed_configs]
            if constrained_configs:
                constrained_rate = max(results["success_rates"][k] for k in constrained_configs)
                
                if relaxed_rate > constrained_rate + 10:  # 10% improvement
                    results["recommendations"].append({
                        "type": "beam_constraint_issue",
                        "description": f"Relaxed beam constraints improve success rate by {relaxed_rate - constrained_rate:.1f}%",
                        "recommendation": "Consider less restrictive beam filtering or improved beam generation"
                    })
        
        return results
    
    def validate_pathfinding_fixes(self) -> Dict[str, Any]:
        """
        Validate automated pathfinding fixes and improvements.
        
        This method tests critical pathfinding cases and validates that 
        proposed fixes (like adding derivational relations) would work.
        
        Returns:
            Validation results for pathfinding fixes
        """
        if self.verbosity >= 1:
            print("\n=== Validating Pathfinding Fixes ===")
        
        # Critical test cases that currently fail
        critical_cases = [
            ("cat.n.01", "chase.v.01"),
            ("chase.v.01", "mouse.n.01"),
            ("hunt.v.01", "prey.n.01"),
            ("predator.n.01", "hunt.v.01")
        ]
        
        results = {
            "critical_test_cases": critical_cases,
            "current_failures": {},
            "derivational_analysis": {},
            "fix_simulations": {},
            "recommendations": []
        }
        
        # Test current failures
        current_failures = 0
        for src, tgt in critical_cases:
            if src in self.graph and tgt in self.graph:
                try:
                    path = nx.shortest_path(self.graph, src, tgt)
                    results["current_failures"][f"{src}->{tgt}"] = {
                        "has_path": True,
                        "path_length": len(path) - 1,
                        "path": path
                    }
                except nx.NetworkXNoPath:
                    results["current_failures"][f"{src}->{tgt}"] = {
                        "has_path": False,
                        "error": "No path found"
                    }
                    current_failures += 1
            else:
                missing = [x for x in [src, tgt] if x not in self.graph]
                results["current_failures"][f"{src}->{tgt}"] = {
                    "has_path": False,
                    "error": f"Missing nodes: {missing}"
                }
                current_failures += 1
        
        # Analyze derivational connections that could fix failures
        for src, tgt in critical_cases:
            try:
                src_synset = wn.synset(src)
                tgt_synset = wn.synset(tgt)
                
                derivational_bridges = []
                for src_lemma in src_synset.lemmas():
                    for derived in src_lemma.derivationally_related_forms():
                        der_synset = derived.synset()
                        if der_synset.pos() == tgt_synset.pos():
                            derivational_bridges.append({
                                "source_lemma": src_lemma.name(),
                                "derived_lemma": derived.name(),
                                "derived_synset": der_synset.name(),
                                "bridge_type": f"{src_synset.pos()}_to_{der_synset.pos()}"
                            })
                
                results["derivational_analysis"][f"{src}->{tgt}"] = {
                    "cross_pos_connection": src_synset.pos() != tgt_synset.pos(),
                    "derivational_bridges_found": len(derivational_bridges),
                    "bridges": derivational_bridges[:5]  # Top 5
                }
            except Exception as e:
                results["derivational_analysis"][f"{src}->{tgt}"] = {
                    "error": str(e)
                }
        
        # Generate recommendations
        failure_rate = current_failures / len(critical_cases) * 100
        
        results["recommendations"].append({
            "type": "current_system_performance",
            "failure_rate": failure_rate,
            "description": f"{current_failures} out of {len(critical_cases)} critical cases fail"
        })
        
        if failure_rate > 50:
            results["recommendations"].append({
                "type": "critical_improvement_needed",
                "priority": "HIGH",
                "description": "More than 50% of critical pathfinding cases fail",
                "suggested_fix": "Implement derivationally_related_forms() in graph construction"
            })
        
        return results
    
    # ========================================
    # Graph Analysis Methods
    # ========================================
    
    def analyze_graph_topology(self) -> Dict[str, Any]:
        """
        Analyze the topology and structure of the current WordNet graph.
        
        Returns:
            Comprehensive graph topology analysis
        """
        if self.verbosity >= 1:
            print("\n=== Analyzing Graph Topology ===")
        
        results = {
            "basic_statistics": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_directed": self.graph.is_directed()
            },
            "connectivity_analysis": {},
            "degree_analysis": {},
            "relation_distribution": {},
            "pos_distribution": {}
        }
        
        # Connectivity analysis
        if self.graph.is_directed():
            weakly_connected = nx.is_weakly_connected(self.graph)
            num_weak_components = nx.number_weakly_connected_components(self.graph)
            results["connectivity_analysis"] = {
                "weakly_connected": weakly_connected,
                "num_weak_components": num_weak_components,
                "largest_component_size": len(max(nx.weakly_connected_components(self.graph), key=len))
            }
        else:
            connected = nx.is_connected(self.graph)
            num_components = nx.number_connected_components(self.graph)
            results["connectivity_analysis"] = {
                "connected": connected,
                "num_components": num_components,
                "largest_component_size": len(max(nx.connected_components(self.graph), key=len))
            }
        
        # Degree analysis
        degrees = dict(self.graph.degree())
        if degrees:
            results["degree_analysis"] = {
                "average_degree": sum(degrees.values()) / len(degrees),
                "max_degree": max(degrees.values()),
                "min_degree": min(degrees.values()),
                "degree_distribution": {i: count for i, count in enumerate(nx.degree_histogram(self.graph))}
            }
        
        # Relation distribution
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            relation_counts[relation] += 1
        results["relation_distribution"] = dict(relation_counts)
        
        # POS distribution
        pos_counts = defaultdict(int)
        for node in self.graph.nodes():
            try:
                synset = wn.synset(node)
                pos_counts[synset.pos()] += 1
            except:
                pos_counts['unknown'] += 1
        results["pos_distribution"] = dict(pos_counts)
        
        return results
    
    def analyze_cross_pos_connectivity(self) -> Dict[str, Any]:
        """
        Analyze cross-POS (part-of-speech) connectivity in the graph.
        
        Returns:
            Analysis of connections between different parts of speech
        """
        if self.verbosity >= 1:
            print("\n=== Analyzing Cross-POS Connectivity ===")
        
        results = {
            "pos_pairs": {},
            "cross_pos_edges": [],
            "missing_connections": [],
            "recommendations": []
        }
        
        # Analyze all edges for cross-POS connections
        cross_pos_count = 0
        total_edges = 0
        
        for u, v, data in self.graph.edges(data=True):
            total_edges += 1
            try:
                u_synset = wn.synset(u)
                v_synset = wn.synset(v)
                u_pos = u_synset.pos()
                v_pos = v_synset.pos()
                
                pos_pair = f"{u_pos}->{v_pos}"
                if pos_pair not in results["pos_pairs"]:
                    results["pos_pairs"][pos_pair] = 0
                results["pos_pairs"][pos_pair] += 1
                
                if u_pos != v_pos:
                    cross_pos_count += 1
                    results["cross_pos_edges"].append({
                        "source": u,
                        "target": v,
                        "source_pos": u_pos,
                        "target_pos": v_pos,
                        "relation": data.get('relation', 'unknown')
                    })
            except:
                continue
        
        results["cross_pos_statistics"] = {
            "total_edges": total_edges,
            "cross_pos_edges": cross_pos_count,
            "cross_pos_percentage": (cross_pos_count / total_edges * 100) if total_edges > 0 else 0
        }
        
        # Check for missing critical connections
        critical_pos_pairs = ["n->v", "v->n", "n->a", "a->n"]
        for pos_pair in critical_pos_pairs:
            if pos_pair not in results["pos_pairs"]:
                results["missing_connections"].append(pos_pair)
        
        # Generate recommendations
        if cross_pos_count == 0:
            results["recommendations"].append({
                "priority": "CRITICAL",
                "type": "no_cross_pos_connections",
                "description": "No cross-POS connections found in graph",
                "recommendation": "Add derivationally_related_forms() to enable noun-verb connections"
            })
        elif cross_pos_count < total_edges * 0.1:  # Less than 10%
            results["recommendations"].append({
                "priority": "HIGH",
                "type": "low_cross_pos_connectivity",
                "description": f"Only {cross_pos_count} ({cross_pos_count/total_edges*100:.1f}%) cross-POS connections",
                "recommendation": "Increase cross-POS relation coverage for better semantic bridging"
            })
        
        return results
    
    def analyze_relation_density(self, synsets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze relation density for specific synsets or the entire graph.
        
        Args:
            synsets: Optional list of synset names to analyze (defaults to all)
            
        Returns:
            Relation density analysis results
        """
        if self.verbosity >= 1:
            print("\n=== Analyzing Relation Density ===")
        
        if synsets is None:
            synsets = list(self.graph.nodes())
        
        results = {
            "analyzed_synsets": synsets[:10],  # Show first 10
            "total_analyzed": len(synsets),
            "density_statistics": {},
            "sparse_nodes": [],
            "well_connected_nodes": [],
            "relation_patterns": {}
        }
        
        degrees = []
        sparse_threshold = 3
        well_connected_threshold = 10
        
        for synset_name in synsets:
            if synset_name not in self.graph:
                continue
                
            degree = self.graph.degree(synset_name)
            degrees.append(degree)
            
            if degree < sparse_threshold:
                results["sparse_nodes"].append({
                    "synset": synset_name,
                    "degree": degree,
                    "neighbors": list(self.graph.neighbors(synset_name))
                })
            elif degree >= well_connected_threshold:
                results["well_connected_nodes"].append({
                    "synset": synset_name,
                    "degree": degree
                })
        
        if degrees:
            results["density_statistics"] = {
                "average_degree": sum(degrees) / len(degrees),
                "median_degree": sorted(degrees)[len(degrees)//2],
                "max_degree": max(degrees),
                "min_degree": min(degrees),
                "sparse_nodes_count": len(results["sparse_nodes"]),
                "well_connected_count": len(results["well_connected_nodes"])
            }
        
        # Analyze relation patterns
        relation_usage = defaultdict(int)
        for synset_name in synsets:
            if synset_name not in self.graph:
                continue
            for neighbor in self.graph.neighbors(synset_name):
                edge_data = self.graph[synset_name][neighbor]
                relation = edge_data.get('relation', 'unknown')
                relation_usage[relation] += 1
        
        results["relation_patterns"] = dict(relation_usage)
        
        return results
    
    # ========================================
    # Algorithm Analysis Methods
    # ========================================
    
    def analyze_beam_filtering(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze beam filtering effectiveness in pathfinding algorithm.
        
        Args:
            test_cases: List of (source, target) synset pairs to test
            
        Returns:
            Beam filtering analysis results
        """
        if self.verbosity >= 1:
            print("\n=== Analyzing Beam Filtering ===")
        
        results = {
            "test_cases": test_cases,
            "beam_analysis": {},
            "filtering_impact": {},
            "recommendations": []
        }
        
        for source, target in test_cases:
            if source not in self.graph or target not in self.graph:
                continue
                
            case_key = f"{source}->{target}"
            
            try:
                # Test without beam constraints (relax_beam=True)
                pathfinder_relaxed = PairwiseBidirectionalAStar(
                    g=self.graph,
                    src=source,
                    tgt=target,
                    get_new_beams_fn=self._get_beams_function,
                    beam_width=10,
                    max_depth=10,
                    relax_beam=True
                )
                
                paths_relaxed = pathfinder_relaxed.find_paths(max_results=5, len_tolerance=3)
                
                # Test with strict beam constraints
                pathfinder_strict = PairwiseBidirectionalAStar(
                    g=self.graph,
                    src=source,
                    tgt=target,
                    get_new_beams_fn=self._get_beams_function,
                    beam_width=3,
                    max_depth=6,
                    relax_beam=False
                )
                
                paths_strict = pathfinder_strict.find_paths(max_results=5, len_tolerance=3)
                
                results["beam_analysis"][case_key] = {
                    "relaxed_paths": len(paths_relaxed),
                    "strict_paths": len(paths_strict),
                    "beam_filtering_blocks": len(paths_relaxed) - len(paths_strict)
                }
                
                if len(paths_relaxed) > len(paths_strict):
                    results["filtering_impact"][case_key] = {
                        "type": "beam_blocks_paths",
                        "blocked_count": len(paths_relaxed) - len(paths_strict),
                        "impact": "negative"
                    }
                elif len(paths_strict) > 0:
                    results["filtering_impact"][case_key] = {
                        "type": "beam_effective",
                        "impact": "positive"
                    }
                else:
                    results["filtering_impact"][case_key] = {
                        "type": "no_paths_found",
                        "impact": "unknown"
                    }
            
            except Exception as e:
                results["beam_analysis"][case_key] = {
                    "error": str(e)
                }
        
        # Generate recommendations
        blocked_cases = len([v for v in results["filtering_impact"].values() 
                           if v.get("type") == "beam_blocks_paths"])
        
        if blocked_cases > len(test_cases) * 0.5:
            results["recommendations"].append({
                "type": "beam_too_restrictive",
                "description": f"Beam filtering blocks paths in {blocked_cases} out of {len(test_cases)} cases",
                "recommendation": "Consider increasing beam_width or using relax_beam=True"
            })
        
        return results
    
    def analyze_heuristic_effectiveness(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze effectiveness of embedding-based heuristics.
        
        Args:
            test_cases: List of (source, target) synset pairs to analyze
            
        Returns:
            Heuristic effectiveness analysis
        """
        if self.verbosity >= 1:
            print("\n=== Analyzing Heuristic Effectiveness ===")
        
        results = {
            "test_cases": test_cases,
            "heuristic_analysis": {},
            "embedding_vs_wordnet": {},
            "recommendations": []
        }
        
        if self.embedding_helper is None:
            results["error"] = "Embedding helper not available"
            return results
        
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
        
        if false_positives > len(test_cases) * 0.3:
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
    
    def profile_search_performance(self, test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Profile search performance across different configurations.
        
        Args:
            test_cases: List of (source, target) synset pairs to test
            
        Returns:
            Performance profiling results
        """
        if self.verbosity >= 1:
            print("\n=== Profiling Search Performance ===")
        
        configurations = [
            {"name": "default", "beam_width": 3, "max_depth": 6, "relax_beam": False},
            {"name": "relaxed", "beam_width": 7, "max_depth": 10, "relax_beam": False},
            {"name": "no_beam", "beam_width": 10, "max_depth": 12, "relax_beam": True}
        ]
        
        results = {
            "configurations": configurations,
            "performance_data": {},
            "summary_statistics": {},
            "recommendations": []
        }
        
        for config in configurations:
            config_name = config["name"]
            config_params = {k: v for k, v in config.items() if k != "name"}
            
            times = []
            success_count = 0
            
            for source, target in test_cases:
                if source not in self.graph or target not in self.graph:
                    continue
                
                try:
                    pathfinder = PairwiseBidirectionalAStar(
                        g=self.graph,
                        src=source,
                        tgt=target,
                        get_new_beams_fn=self._get_beams_function,
                        **config_params
                    )
                    
                    start_time = time.time()
                    paths = pathfinder.find_paths(max_results=3, len_tolerance=2)
                    end_time = time.time()
                    
                    elapsed = end_time - start_time
                    times.append(elapsed)
                    
                    if paths:
                        success_count += 1
                
                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"Error in {config_name} for {source}->{target}: {e}")
            
            if times:
                results["performance_data"][config_name] = {
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "success_rate": success_count / len(test_cases) * 100,
                    "total_tests": len(test_cases)
                }
            else:
                results["performance_data"][config_name] = {
                    "error": "No successful tests"
                }
        
        # Generate summary and recommendations
        if results["performance_data"]:
            best_performance = min(results["performance_data"].items(), 
                                 key=lambda x: x[1].get("average_time", float('inf')))
            best_success = max(results["performance_data"].items(), 
                             key=lambda x: x[1].get("success_rate", 0))
            
            results["summary_statistics"] = {
                "fastest_config": best_performance[0],
                "fastest_time": best_performance[1].get("average_time", 0),
                "highest_success_config": best_success[0],
                "highest_success_rate": best_success[1].get("success_rate", 0)
            }
            
            if best_performance[0] != best_success[0]:
                results["recommendations"].append({
                    "type": "performance_tradeoff",
                    "description": f"Fastest config ({best_performance[0]}) differs from most successful ({best_success[0]})",
                    "recommendation": "Consider balancing speed vs success rate based on use case"
                })
        
        return results
    
    # ========================================
    # Helper Methods
    # ========================================
    
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
    
    def _get_beams_function(self, g: nx.DiGraph, src: str, tgt: str):
        """Wrapper function to get embedding-based beams."""
        if self.beam_builder is None:
            return []
        
        try:
            return self.beam_builder.get_new_beams(g, src, tgt)
        except Exception as e:
            if self.verbosity >= 2:
                print(f"Warning: Beam generation failed: {e}")
            return []
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def get_standard_test_cases(self) -> List[Tuple[str, str]]:
        """Get standard test cases for pathfinding analysis."""
        return [
            ("cat.n.01", "chase.v.01"),
            ("chase.v.01", "mouse.n.01"),
            ("cat.n.01", "mouse.n.01"),
            ("predator.n.01", "hunt.v.01"),
            ("hunt.v.01", "prey.n.01"),
            ("dog.n.01", "run.v.01"),
            ("bird.n.01", "fly.v.01")
        ]
    
    def export_results(self, results: Dict[str, Any], filepath: str, format: str = "json"):
        """
        Export analysis results to file.
        
        Args:
            results: Results dictionary to export
            filepath: Output file path
            format: Export format ("json", "txt")
        """
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format.lower() == "txt":
            with open(filepath, 'w') as f:
                f.write("SMIED Diagnostics Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(json.dumps(results, indent=2, default=str))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbosity >= 1:
            print(f"Results exported to {filepath}")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis across all diagnostic methods.
        
        Returns:
            Complete diagnostic analysis results
        """
        if self.verbosity >= 1:
            print("Running comprehensive SMIED diagnostics analysis...")
        
        test_cases = self.get_standard_test_cases()
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "graph_info": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            },
            "relation_audit": self.audit_wordnet_relations(),
            "graph_topology": self.analyze_graph_topology(),
            "cross_pos_connectivity": self.analyze_cross_pos_connectivity(),
            "pathfinding_validation": self.validate_pathfinding_fixes(),
            "parameter_sensitivity": self.test_parameter_sensitivity(test_cases),
            "heuristic_effectiveness": self.analyze_heuristic_effectiveness(test_cases),
            "performance_profile": self.profile_search_performance(test_cases),
            "connectivity_analysis": {}
        }
        
        # Analyze connectivity for key test cases
        for source, target in test_cases[:5]:  # Limit to first 5 for comprehensive analysis
            key = f"{source}_to_{target}".replace(".", "_")
            results["connectivity_analysis"][key] = self.analyze_synset_connectivity(source, target)
        
        if self.verbosity >= 1:
            print("Comprehensive analysis completed.")
        
        return results


def main():
    """Main function with command-line interface for diagnostics."""
    parser = argparse.ArgumentParser(description="SMIED Comprehensive Diagnostics")
    parser.add_argument("--analyze-connectivity", nargs=2, metavar=("SOURCE", "TARGET"),
                       help="Analyze connectivity between two synsets")
    parser.add_argument("--audit-relations", action="store_true",
                       help="Audit WordNet relation coverage")
    parser.add_argument("--test-parameters", action="store_true",
                       help="Test pathfinding with different parameters")
    parser.add_argument("--validate-fixes", action="store_true",
                       help="Validate pathfinding fixes")
    parser.add_argument("--analyze-topology", action="store_true",
                       help="Analyze graph topology")
    parser.add_argument("--cross-pos-analysis", action="store_true",
                       help="Analyze cross-POS connectivity")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive analysis")
    parser.add_argument("--verbosity", type=int, default=1,
                       help="Verbosity level (0-2)")
    parser.add_argument("--output", type=str,
                       help="Output results to JSON file")
    parser.add_argument("--format", type=str, default="json", choices=["json", "txt"],
                       help="Output format")
    
    args = parser.parse_args()
    
    # Create diagnostics instance
    diagnostics = SMIEDDiagnostics(verbosity=args.verbosity)
    
    results = {}
    
    # Run requested analyses
    if args.analyze_connectivity:
        source, target = args.analyze_connectivity
        results["connectivity_analysis"] = diagnostics.analyze_synset_connectivity(source, target)
    
    if args.audit_relations:
        results["relation_audit"] = diagnostics.audit_wordnet_relations()
    
    if args.test_parameters:
        test_cases = diagnostics.get_standard_test_cases()
        results["parameter_analysis"] = diagnostics.test_parameter_sensitivity(test_cases)
    
    if args.validate_fixes:
        results["pathfinding_validation"] = diagnostics.validate_pathfinding_fixes()
    
    if args.analyze_topology:
        results["graph_topology"] = diagnostics.analyze_graph_topology()
    
    if args.cross_pos_analysis:
        results["cross_pos_connectivity"] = diagnostics.analyze_cross_pos_connectivity()
    
    if args.comprehensive:
        results = diagnostics.run_comprehensive_analysis()
    
    # If no specific analysis requested, run basic analysis
    if not any([args.analyze_connectivity, args.audit_relations, args.test_parameters,
                args.validate_fixes, args.analyze_topology, args.cross_pos_analysis, args.comprehensive]):
        print("Running basic diagnostic analysis...")
        results["relation_audit"] = diagnostics.audit_wordnet_relations()
        results["pathfinding_validation"] = diagnostics.validate_pathfinding_fixes()
    
    # Output results
    if args.output:
        diagnostics.export_results(results, args.output, args.format)
    else:
        print("\n" + "="*80)
        print("SMIED DIAGNOSTIC RESULTS")
        print("="*80)
        if args.format == "json":
            print(json.dumps(results, indent=2, default=str))
        else:
            for key, value in results.items():
                print(f"\n{key.upper()}:")
                print("-" * len(key))
                print(json.dumps(value, indent=2, default=str))


if __name__ == "__main__":
    main()