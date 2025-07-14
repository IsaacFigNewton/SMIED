from noske.PatternLoader import PatternLoader
from noske.PatternMatcher import *
from noske.utils import match_chain
import networkx as nx

# Example usage
def create_example_patterns_file():
    """Create an example patterns file for demonstration"""
    loader = PatternLoader()
    loader.save_patterns_to_file("semantic_patterns.json")
    print("Example patterns file created: semantic_patterns.json")

# Command-line interface for pattern management
def main():
    """Simple CLI for pattern management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Pattern Matcher")
    parser.add_argument("--create-example", action="store_true",
                       help="Create example patterns file")
    parser.add_argument("--patterns-file", type=str, default="semantic_patterns.json",
                       help="Path to patterns file")
    parser.add_argument("--list-categories", action="store_true",
                       help="List available pattern categories")
    parser.add_argument("--list-patterns", type=str,
                       help="List patterns in a specific category")
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_patterns_file()
        return
    
    try:
        loader = PatternLoader(args.patterns_file)
    except FileNotFoundError:
        print(f"Patterns file not found: {args.patterns_file}")
        print("Use --create-example to create a sample file")
        return
    
    if args.list_categories:
        print("Available categories:")
        for category in loader.list_categories():
            print(f"  - {category}")
    
    if args.list_patterns:
        patterns = loader.list_patterns_in_category(args.list_patterns)
        print(f"Patterns in '{args.list_patterns}':")
        for pattern in patterns:
            print(f"  - {pattern}")

if __name__ == "__main__":
    main()