"""
Comprehensive test runner for SMIED validation and performance analysis.
This script runs all testing suites and generates comprehensive reports.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List
import argparse

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# SMIED module has been removed
SMIED_AVAILABLE = False

# Import test modules
try:
    from test_semantic_pathfinding import run_semantic_pathfinding_tests
    from test_performance_analysis import run_performance_analysis
    from test_regression import run_comprehensive_regression_testing
    from test_comparative_analysis import run_comparative_analysis
    from test_optimization_strategies import run_optimization_analysis
    TEST_MODULES_AVAILABLE = True
except ImportError as e:
    TEST_MODULES_AVAILABLE = False
    print(f"WARNING: Some test modules not available: {e}")


class ComprehensiveTestRunner:
    """
    Runs all SMIED testing suites and generates comprehensive reports.
    """
    
    def __init__(self, verbosity: int = 1, save_results: bool = True):
        self.verbosity = verbosity
        self.save_results = save_results
        self.results = {}
        self.start_time = time.time()
        
        if self.verbosity >= 1:
            print("=" * 80)
            print("COMPREHENSIVE TESTING SUITE")
            print("=" * 80)
            print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            # SMIED module has been removed
            print(f"Test Modules Available: {TEST_MODULES_AVAILABLE}")
            print()
    
    def run_semantic_pathfinding_tests(self) -> Dict[str, Any]:
        """Run semantic pathfinding validation tests."""
        if not SMIED_AVAILABLE or not TEST_MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        if self.verbosity >= 1:
            print("1. RUNNING SEMANTIC PATHFINDING TESTS")
            print("-" * 50)
        
        try:
            results = run_semantic_pathfinding_tests(
                verbosity=self.verbosity,
                save_results=False  # We'll save everything together
            )
            
            if self.verbosity >= 1 and results:
                main_metrics = results.get('main_test_results', {}).get('overall_metrics', {})
                success_rate = main_metrics.get('success_rate', 0)
                avg_time = main_metrics.get('average_time', 0)
                print(f"✓ Semantic pathfinding tests completed: {success_rate:.1f}% success rate, {avg_time:.3f}s avg time")
                print()
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if self.verbosity >= 0:
                print(f"✗ Semantic pathfinding tests failed: {e}")
                print()
            return error_result
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance and scalability analysis."""
        if not SMIED_AVAILABLE or not TEST_MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        if self.verbosity >= 1:
            print("2. RUNNING PERFORMANCE ANALYSIS")
            print("-" * 50)
        
        try:
            results = run_performance_analysis(
                verbosity=self.verbosity,
                save_results=False
            )
            
            if self.verbosity >= 1 and results:
                graph_data = results.get('analysis_sections', {}).get('graph_construction_memory', {})
                if 'construction_time' in graph_data:
                    print(f"✓ Performance analysis completed: Graph construction {graph_data['construction_time']:.2f}s")
                else:
                    print("✓ Performance analysis completed")
                print()
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if self.verbosity >= 0:
                print(f"✗ Performance analysis failed: {e}")
                print()
            return error_result
    
    def run_regression_testing(self) -> Dict[str, Any]:
        """Run regression testing."""
        if not SMIED_AVAILABLE or not TEST_MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        if self.verbosity >= 1:
            print("3. RUNNING REGRESSION TESTING")
            print("-" * 50)
        
        try:
            results = run_comprehensive_regression_testing(
                verbosity=self.verbosity,
                establish_new_baseline=False,  # Don't overwrite existing baselines
                save_results=False
            )
            
            if self.verbosity >= 1 and results:
                regression_report = results.get('regression_report', {}).get('summary', {})
                pass_rate = regression_report.get('pass_rate', 0)
                failed = regression_report.get('failed', 0)
                print(f"✓ Regression testing completed: {pass_rate:.1f}% pass rate, {failed} failures")
                print()
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if self.verbosity >= 0:
                print(f"✗ Regression testing failed: {e}")
                print()
            return error_result
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run comparative analysis vs ConceptNet."""
        if not SMIED_AVAILABLE or not TEST_MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        if self.verbosity >= 1:
            print("4. RUNNING COMPARATIVE ANALYSIS")
            print("-" * 50)
        
        try:
            results = run_comparative_analysis(
                verbosity=self.verbosity,
                save_results=False,
                max_tests=15  # Limit for comprehensive run
            )
            
            if self.verbosity >= 1 and results:
                summary = results.get('comparative_report', {}).get('summary', {})
                smied_success = summary.get('smied_success_rate', 0)
                conceptnet_available = summary.get('conceptnet_available', False)
                
                if conceptnet_available:
                    conceptnet_success = summary.get('conceptnet_success_rate', 0)
                    print(f"✓ Comparative analysis completed: SMIED {smied_success:.1f}% vs ConceptNet {conceptnet_success:.1f}%")
                else:
                    print(f"✓ Comparative analysis completed: SMIED {smied_success:.1f}% (ConceptNet unavailable)")
                print()
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if self.verbosity >= 0:
                print(f"✗ Comparative analysis failed: {e}")
                print()
            return error_result
    
    def run_optimization_analysis(self) -> Dict[str, Any]:
        """Run optimization strategies analysis."""
        if not SMIED_AVAILABLE or not TEST_MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        if self.verbosity >= 1:
            print("5. RUNNING OPTIMIZATION ANALYSIS")
            print("-" * 50)
        
        try:
            results = run_optimization_analysis(
                verbosity=self.verbosity,
                save_results=False
            )
            
            if self.verbosity >= 1 and results:
                summary = results.get('summary', {})
                avg_speedup = summary.get('average_speedup', 0)
                hit_rate = summary.get('optimization_stats', {}).get('hit_rate_percent', 0)
                print(f"✓ Optimization analysis completed: {avg_speedup:.2f}x speedup, {hit_rate:.1f}% cache hit rate")
                print()
            
            return results
            
        except Exception as e:
            error_result = {'error': str(e)}
            if self.verbosity >= 0:
                print(f"✗ Optimization analysis failed: {e}")
                print()
            return error_result
    
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all testing suites."""
        if self.verbosity >= 1:
            print("Running comprehensive SMIED validation and testing...")
            print()
        
        # Run all test suites
        self.results['semantic_pathfinding'] = self.run_semantic_pathfinding_tests()
        self.results['performance_analysis'] = self.run_performance_analysis()
        self.results['regression_testing'] = self.run_regression_testing()
        self.results['comparative_analysis'] = self.run_comparative_analysis()
        self.results['optimization_analysis'] = self.run_optimization_analysis()
        
        # Add metadata
        self.results['metadata'] = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration': time.time() - self.start_time,
            'smied_available': SMIED_AVAILABLE,
            'test_modules_available': TEST_MODULES_AVAILABLE
        }
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save results if requested
        if self.save_results:
            self.save_comprehensive_results()
        
        return self.results
    
    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report."""
        if self.verbosity >= 1:
            print("=" * 80)
            print("COMPREHENSIVE TESTING SUMMARY")
            print("=" * 80)
        
        total_duration = self.results['metadata']['total_duration']
        
        if self.verbosity >= 1:
            print(f"\nTotal Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"Tests Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # Summary statistics
        test_results = []
        
        # Semantic pathfinding results
        semantic_results = self.results.get('semantic_pathfinding', {})
        if 'error' not in semantic_results:
            main_metrics = semantic_results.get('main_test_results', {}).get('overall_metrics', {})
            success_rate = main_metrics.get('success_rate', 0)
            test_results.append(f"Semantic Pathfinding: {success_rate:.1f}% success rate")
        
        # Performance analysis results
        perf_results = self.results.get('performance_analysis', {})
        if 'error' not in perf_results:
            test_results.append("Performance Analysis: Completed")
        
        # Regression testing results
        regression_results = self.results.get('regression_testing', {})
        if 'error' not in regression_results:
            regression_summary = regression_results.get('regression_report', {}).get('summary', {})
            pass_rate = regression_summary.get('pass_rate', 0)
            test_results.append(f"Regression Testing: {pass_rate:.1f}% pass rate")
        
        # Comparative analysis results
        comp_results = self.results.get('comparative_analysis', {})
        if 'error' not in comp_results:
            comp_summary = comp_results.get('comparative_report', {}).get('summary', {})
            smied_success = comp_summary.get('smied_success_rate', 0)
            test_results.append(f"Comparative Analysis: SMIED {smied_success:.1f}% success")
        
        # Optimization results
        opt_results = self.results.get('optimization_analysis', {})
        if 'error' not in opt_results:
            opt_summary = opt_results.get('summary', {})
            speedup = opt_summary.get('average_speedup', 0)
            test_results.append(f"Optimization Analysis: {speedup:.2f}x average speedup")
        
        
        if self.verbosity >= 1:
            print("TEST RESULTS:")
            for result in test_results:
                print(f"  ✓ {result}")
            
            # Error summary
            errors = []
            for test_name, test_result in self.results.items():
                if test_name != 'metadata' and isinstance(test_result, dict) and 'error' in test_result:
                    errors.append(f"{test_name}: {test_result['error']}")
            
            if errors:
                print(f"\nERRORS:")
                for error in errors:
                    print(f"  ✗ {error}")
            
            print(f"\nRECOMMENDations:")
            print("  1. Review test results for any failures or regressions")
            print("  2. Complete human evaluation study for qualitative assessment")
            print("  3. Investigate any performance bottlenecks identified")
            print("  4. Update optimization strategies based on analysis results")
            print("  5. Consider establishing new regression baselines if appropriate")
            
            print("=" * 80)
    
    def save_comprehensive_results(self) -> str:
        """Save all test results to a comprehensive report file."""
        timestamp = int(time.time())
        filename = f"smied_comprehensive_test_results_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Make results JSON-serializable
        json_results = self._make_json_serializable(self.results)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            if self.verbosity >= 1:
                print(f"\nComprehensive test results saved to: {filepath}")
            
            return filepath
            
        except Exception as e:
            if self.verbosity >= 0:
                print(f"Error saving results: {e}")
            return ""
    
    def _make_json_serializable(self, obj) -> Any:
        """Convert complex objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Run comprehensive SMIED testing suite')
    parser.add_argument('-v', '--verbosity', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level (0=silent, 1=normal, 2=detailed)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save test results to file')
    parser.add_argument('--test-suite', choices=[
        'semantic', 'performance', 'regression', 'comparative', 'optimization'
    ], help='Run only specific test suite')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(
        verbosity=args.verbosity,
        save_results=not args.no_save
    )
    
    if args.test_suite:
        # Run specific test suite
        if args.test_suite == 'semantic':
            results = {'semantic_pathfinding': runner.run_semantic_pathfinding_tests()}
        elif args.test_suite == 'performance':
            results = {'performance_analysis': runner.run_performance_analysis()}
        elif args.test_suite == 'regression':
            results = {'regression_testing': runner.run_regression_testing()}
        elif args.test_suite == 'comparative':
            results = {'comparative_analysis': runner.run_comparative_analysis()}
        elif args.test_suite == 'optimization':
            results = {'optimization_analysis': runner.run_optimization_analysis()}
        
        runner.results = results
        runner.results['metadata'] = {
            'start_time': runner.start_time,
            'end_time': time.time(),
            'total_duration': time.time() - runner.start_time,
            'smied_available': SMIED_AVAILABLE,
            'test_modules_available': TEST_MODULES_AVAILABLE
        }
        
        if args.verbosity >= 1:
            print(f"\nSpecific test suite '{args.test_suite}' completed")
    else:
        # Run all test suites
        results = runner.run_all_tests()
    
    return runner.results


if __name__ == '__main__':
    if not SMIED_AVAILABLE:
        print("ERROR: SMIED not available. Cannot run comprehensive tests.")
        sys.exit(1)
    
    if not TEST_MODULES_AVAILABLE:
        print("ERROR: Test modules not available. Check imports.")
        sys.exit(1)
    
    try:
        results = main()
        
        # Exit with appropriate code
        has_errors = any(
            isinstance(result, dict) and 'error' in result 
            for result in results.values() 
            if isinstance(result, dict)
        )
        
        sys.exit(1 if has_errors else 0)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: Comprehensive testing failed: {e}")
        sys.exit(1)