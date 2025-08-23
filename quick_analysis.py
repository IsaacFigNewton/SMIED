#!/usr/bin/env python3
"""
Quick analysis of SMIED semantic pathfinding issues
Focused on Phase 1 and Phase 2 priority items from TODO
"""

import sys
sys.path.append('src')

from collections import defaultdict, deque
from nltk.corpus import wordnet as wn
import networkx as nx
from smied.SemanticDecomposer import SemanticDecomposer

def analyze_wordnet_relation_coverage():
    """Audit what relations are currently included vs available in WordNet."""
    print("=== PHASE 1.1: WordNet Relation Coverage Audit ===")
    
    # Relations currently implemented in SemanticDecomposer (from lines 620-634)
    implemented_relations = {
        'hypernyms', 'hyponyms', 'member_holonyms', 'part_holonyms',
        'substance_holonyms', 'member_meronyms', 'part_meronyms',
        'substance_meronyms', 'similar_tos', 'also_sees', 'verb_groups',
        'entailments', 'causes'
    }
    
    # Available WordNet relations by POS
    wordnet_relations = {
        'noun': ['hypernyms', 'hyponyms', 'member_holonyms', 'part_holonyms', 'substance_holonyms',
                 'member_meronyms', 'part_meronyms', 'substance_meronyms', 'similar_tos', 'also_sees', 'attributes'],
        'verb': ['hypernyms', 'hyponyms', 'entailments', 'causes', 'similar_tos', 'also_sees', 'verb_groups'],
        'adj': ['similar_tos', 'also_sees', 'attributes'],
        'cross_pos_lemma': ['derivationally_related_forms', 'antonyms', 'pertainyms']
    }
    
    print(f"[+] Implemented relations: {len(implemented_relations)}")
    for rel in sorted(implemented_relations):
        print(f"   - {rel}")
    
    # Find missing relations
    all_available = set()
    for pos_rels in wordnet_relations.values():
        all_available.update(pos_rels)
    
    missing_relations = all_available - implemented_relations
    print(f"\n[-] Missing relations: {len(missing_relations)}")
    for rel in sorted(missing_relations):
        print(f"   - {rel}")
    
    # Critical analysis for cross-POS connections
    print(f"\n[!] CRITICAL FINDING:")
    if 'derivationally_related_forms' in missing_relations:
        print(f"   [-] derivationally_related_forms() is MISSING")
        print(f"   [*] This is essential for noun<->verb connections like cat.n.01 -> chase.v.01")
        print(f"   [*] This likely explains why cat->chase paths are failing")
    
    if 'attributes' in missing_relations:
        print(f"   [-] attributes() is MISSING")
        print(f"   [*] Important for noun-adjective connections")
    
    return {
        'implemented': implemented_relations,
        'missing': missing_relations,
        'critical_missing': [r for r in missing_relations if r in ['derivationally_related_forms', 'attributes']]
    }

def test_specific_connectivity():
    """Phase 1.2: Test specific failing cases manually."""
    print("\n=== PHASE 1.2: Specific Path Connectivity Testing ===")
    
    test_cases = [
        ("cat.n.01", "chase.v.01"),
        ("chase.v.01", "mouse.n.01"),
        ("cat.n.01", "mouse.n.01")
    ]
    
    for source, target in test_cases:
        print(f"\n🔍 Analyzing: {source} → {target}")
        
        try:
            src_synset = wn.synset(source)
            tgt_synset = wn.synset(target)
            
            print(f"   Source: {src_synset.definition()}")
            print(f"   Target: {tgt_synset.definition()}")
            
            # Check for cross-POS connection (the likely issue)
            if src_synset.pos() != tgt_synset.pos():
                print(f"   [!] Cross-POS connection: {src_synset.pos()} -> {tgt_synset.pos()}")
                
                # Test derivational connections (not in current graph!)
                src_lemmas = src_synset.lemmas()
                potential_connections = []
                
                for lemma in src_lemmas:
                    derived = lemma.derivationally_related_forms()
                    for der_lemma in derived:
                        der_synset = der_lemma.synset()
                        if der_synset.pos() == tgt_synset.pos():
                            potential_connections.append((lemma.name(), der_lemma.name(), der_synset))
                
                if potential_connections:
                    print(f"   [*] Found potential derivational bridges:")
                    for src_lemma, tgt_lemma, bridge_synset in potential_connections[:3]:
                        print(f"      {src_lemma} -> {tgt_lemma} ({bridge_synset})")
                else:
                    print(f"   [-] No direct derivational connections found")
            
            # Check relation density
            src_relations = {
                'hypernyms': src_synset.hypernyms(),
                'hyponyms': src_synset.hyponyms(),
                'similar_tos': src_synset.similar_tos() if hasattr(src_synset, 'similar_tos') else [],
            }
            
            total_connections = sum(len(rels) for rels in src_relations.values())
            print(f"   📊 Source relation density: {total_connections} total connections")
            
            if total_connections < 3:
                print(f"   ⚠️  Low connectivity - may be isolated in graph")
                
        except Exception as e:
            print(f"   ❌ Error analyzing: {e}")

def build_test_graph_and_analyze():
    """Build a small test graph and analyze connectivity."""
    print("\n=== PHASE 1.3: Test Graph Analysis ===")
    
    # Create SemanticDecomposer
    decomposer = SemanticDecomposer(verbosity=1)
    
    # Get test synsets
    test_synset_names = [
        'cat.n.01', 'mouse.n.01', 'chase.v.01', 'hunt.v.01', 
        'predator.n.01', 'prey.n.01', 'animal.n.01', 'mammal.n.01',
        'feline.n.01', 'rodent.n.01'
    ]
    
    test_synsets = []
    for name in test_synset_names:
        try:
            synset = wn.synset(name)
            test_synsets.append(synset)
            print(f"   ✅ Added: {synset} - {synset.definition()[:50]}...")
        except:
            print(f"   ❌ Could not find: {name}")
    
    print(f"\n🔧 Building graph with {len(test_synsets)} synsets...")
    graph = decomposer.build_synset_graph(test_synsets)
    
    print(f"📊 Graph statistics:")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    # Analyze connectivity for our problem cases
    test_pairs = [("cat.n.01", "chase.v.01"), ("chase.v.01", "mouse.n.01"), ("cat.n.01", "mouse.n.01")]
    
    for source, target in test_pairs:
        if source in graph and target in graph:
            # Check direct connection
            has_direct = graph.has_edge(source, target)
            print(f"\n🔍 {source} → {target}:")
            print(f"   Direct connection: {'✅ Yes' if has_direct else '❌ No'}")
            
            if has_direct:
                edge_data = graph[source][target]
                print(f"   Relation: {edge_data.get('relation', 'unknown')}")
            
            # Check for paths using NetworkX
            try:
                if nx.has_path(graph, source, target):
                    shortest_path = nx.shortest_path(graph, source, target)
                    print(f"   Shortest path length: {len(shortest_path) - 1}")
                    print(f"   Path: {' → '.join(shortest_path)}")
                    
                    # Show relations used
                    path_relations = []
                    for i in range(len(shortest_path) - 1):
                        u, v = shortest_path[i], shortest_path[i + 1]
                        rel = graph[u][v].get('relation', 'unknown')
                        path_relations.append(rel)
                    print(f"   Relations: {' → '.join(path_relations)}")
                else:
                    print(f"   ❌ No path found in current graph")
                    
            except nx.NetworkXNoPath:
                print(f"   ❌ No path found in current graph")
        else:
            missing = []
            if source not in graph:
                missing.append(source)
            if target not in graph:
                missing.append(target)
            print(f"   ❌ Missing nodes: {missing}")
    
    # Analyze relation type distribution
    relation_counts = defaultdict(int)
    for u, v, data in graph.edges(data=True):
        rel = data.get('relation', 'unknown')
        relation_counts[rel] += 1
    
    print(f"\n📊 Edge relation distribution:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {rel}: {count}")
    
    return graph, relation_counts

def main():
    """Run the complete analysis."""
    print("SMIED Semantic Pathfinding Analysis")
    print("=" * 60)
    
    # Phase 1.1: Relation coverage audit
    relation_audit = analyze_wordnet_relation_coverage()
    
    # Phase 1.2: Specific connectivity testing  
    test_specific_connectivity()
    
    # Phase 1.3: Build test graph and analyze
    graph, relation_stats = build_test_graph_and_analyze()
    
    print(f"\n" + "=" * 60)
    print("🎯 KEY FINDINGS SUMMARY:")
    print("=" * 60)
    
    if 'derivationally_related_forms' in relation_audit['critical_missing']:
        print("🚨 CRITICAL ISSUE #1: Missing Cross-POS Relations")
        print("   - derivationally_related_forms() not implemented")
        print("   - This prevents noun↔verb connections (cat.n.01 ↔ chase.v.01)")
        print("   - PRIORITY: HIGH - Add to SemanticDecomposer.build_synset_graph()")
        print("   - FIX: Add lemma-level relation processing around line 649")
    
    print(f"\n📊 Graph Statistics:")
    print(f"   - Nodes: {graph.number_of_nodes()}")
    print(f"   - Edges: {graph.number_of_edges()}")
    print(f"   - Most common relations: {list(sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    if graph.number_of_edges() / graph.number_of_nodes() < 3:
        print("⚠️  Low edge density - graph may be poorly connected")
    
    print(f"\n💡 IMMEDIATE ACTION ITEMS:")
    print("1. Add derivationally_related_forms() to graph construction")
    print("2. Add attributes() relation for noun-adjective connections") 
    print("3. Test cross-POS pathfinding after relation additions")
    print("4. Consider relation-specific edge weights")

if __name__ == "__main__":
    main()