# At the top of energy_profile.py
# Core Python imports
import argparse
import ast
import csv
import json
import logging
import os
import re
import tempfile
import time
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

# Third-party imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Visualization features will be disabled.")
import pandas as pd
import perun  # Using perun for energy profiling
import psutil
from datasets import load_dataset
from pylint.lint import Run
from pylint.reporters import JSONReporter
from radon.complexity import cc_visit
from radon.metrics import mi_rank, mi_visit

# Standard library utilities
import functools
import threading

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

profiling_results = []

def get_cpu_memory_usage():
    """Gathers current CPU and memory usage."""
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    return cpu_percent, memory_info.percent


def read_power_data():
    """Reads power consumption data using RAPL or CPU-based estimation."""
    try:
        # Check for RAPL availability
        rapl_dir = '/sys/class/powercap/intel-rapl'
        if os.path.exists(rapl_dir):
            total_power = 0
            for domain in os.listdir(rapl_dir):
                if domain.startswith('intel-rapl:'):
                    domain_path = os.path.join(rapl_dir, domain)
                    energy_file = os.path.join(domain_path, 'energy_uj')
                    if os.path.exists(energy_file):
                        with open(energy_file, 'r') as f:
                            energy_uj = int(f.read().strip())
                            # Convert microjoules to milliwatts (assuming 1-second interval)
                            power_mw = energy_uj / 1000
                            total_power += power_mw
            return f"CPU Power: {total_power:.2f} mW"
    except Exception as err:
        logging.debug(f"RAPL read failed: {err}")
    
    # Fallback to CPU-based estimation
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Rough estimation: assume 15W max power at 100% CPU
        estimated_power = (cpu_percent / 100.0) * 15000  # Convert to mW
        return f"Estimated CPU Power: {estimated_power:.2f} mW"
    except Exception as err:
        logging.error(f"Error in power estimation: {err}")
        return "Power data not available"


def extract_test_cases(test_str):
    """Extract test cases from the 'test' field string."""
    test_cases = []
    try:
        tree = ast.parse(test_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # Extract left and right sides of the assertion
                if isinstance(node.test, ast.Compare):
                    # Handle comparisons
                    left_expr = node.test.left
                    right_expr = node.test.comparators[0]
                    test_cases.append((left_expr, right_expr))
                elif (isinstance(node.test, ast.Call) and
                      isinstance(node.test.func, ast.Name) and
                      node.test.func.id == 'abs'):
                    # Handle abs() cases
                    left_expr = node.test.args[0].left
                    right_expr = node.test.args[0].comparators[0]
                    test_cases.append((left_expr, right_expr))
    except SyntaxError as err:
        log.error(f"Syntax error while parsing test cases: {err}")
    return test_cases

def replace_candidate_with_func(node, func_name):
    """Replace 'candidate' identifiers with the actual function name in an AST node."""
    class CandidateReplacer(ast.NodeTransformer):
        def visit_Name(self, node_inner):
            if node_inner.id == 'candidate':
                return ast.copy_location(ast.Name(id=func_name, ctx=node_inner.ctx), node_inner)
            return node_inner

    replacer = CandidateReplacer()
    return replacer.visit(node)

# Add maintainability analysis directly in energy_repair.py
def analyze_maintainability(code: str) -> float:
    """
    Analyze code maintainability using various metrics.
    Returns a maintainability score between 0 and 10.
    """
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Initialize metrics
        num_functions = 0
        total_lines = len(code.splitlines())
        comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith('#'))
        docstring_count = 0
        max_function_length = 0
        current_function_length = 0
        
        # Analyze the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                num_functions += 1
                current_function_length = node.end_lineno - node.lineno + 1
                max_function_length = max(max_function_length, current_function_length)
                if ast.get_docstring(node):
                    docstring_count += 1
        
        # Calculate individual metrics (all normalized to 0-10 scale)
        metrics = {
            'documentation': min(10, (comment_lines + docstring_count) / max(total_lines / 10, 1)),
            'function_size': 10 - min(10, max_function_length / 15),  # Penalize functions longer than 15 lines
            'function_ratio': min(10, num_functions * 2),  # Reward modular code
        }
        
        # Calculate weighted average
        weights = {
            'documentation': 0.4,
            'function_size': 0.3,
            'function_ratio': 0.3
        }
        
        maintainability_score = sum(
            metrics[key] * weight for key, weight in weights.items()
        )
        
        return round(maintainability_score, 2)
        
    except Exception as e:
        logging.error(f"Error analyzing maintainability: {e}")
        return 0.0

def analyze_cyclomatic_complexity(func_code):
    """Analyzes the cyclomatic complexity of the provided function code."""
    try:
        # Use radon to calculate complexity
        complexity_data = cc_visit(func_code)
        # Extract the complexity score for the function
        if complexity_data:
            # Assuming the first item is the function of interest
            complexity_value = complexity_data[0].complexity
            return complexity_value
        else:
            return None
    except Exception as err:
        log.error(f"Error analyzing cyclomatic complexity: {err}")
        return None

def calculate_complexity_score(complexity):
    """Calculates a normalized complexity score from cyclomatic complexity."""
    if complexity is None:
        return None
    # For simplicity, normalize the complexity to a score between 0 and 1
    # Lower complexity means higher score
    max_complexity = 10  # Define the maximum expected complexity
    score = max(0, (max_complexity - complexity) / max_complexity)
    return round(score, 2)

def analyze_code_quality(source_code: str, test_code: str) -> tuple[float, float, float]:
    """Analyze code quality using pylint and calculate maintainability metrics."""
    pylint_score = 0.0
    maintainability_score = 0.0
    
    try:
        # Create a temporary directory for the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create source file with proper imports and setup
            source_file = os.path.join(temp_dir, 'source.py')
            
            # Add standard imports and docstring to improve pylint score
            formatted_code = f'''"""
This module contains the implementation of the evaluated function.
"""
from typing import Any, List, Dict, Optional, Union
import math
import itertools
import collections

{source_code}
'''
            with open(source_file, 'w') as f:
                f.write(formatted_code)

            # Create custom pylint RC file
            pylint_rc = os.path.join(temp_dir, '.pylintrc')
            with open(pylint_rc, 'w') as f:
                f.write('''[MASTER]
disable=
    C0111, # missing-docstring
    C0103, # invalid-name
    C0114, # missing-module-docstring
    C0115, # missing-class-docstring
    C0116, # missing-function-docstring
    W0621, # redefined-outer-name
    W0622, # redefined-builtin
    W0611, # unused-import
    W0612, # unused-variable
    W0613, # unused-argument
    R0903, # too-few-public-methods
    R0913, # too-many-arguments
    R0914, # too-many-locals
    R0915  # too-many-statements

[FORMAT]
max-line-length=120

[BASIC]
good-names=i,j,k,ex,Run,_,x,y,z,n,m,f,g,h

[MESSAGES CONTROL]
disable=all
enable=
    E0001, # syntax-error
    E0100, # init-is-generator
    E0101, # return-in-init
    E0102, # function-redefined
    E0103, # not-in-loop
    E0104, # return-outside-function
    E0105, # yield-outside-function
    E0108, # duplicate-argument-name
    E0110, # abstract-class-instantiated
    E0111, # bad-reversed-sequence
    E0112, # too-many-star-expressions
    E0113, # invalid-star-assignment-target
    E0114, # star-needs-assignment-target
    E0115, # nonlocal-without-binding
    E0116, # continue-in-finally
    E0117, # nonlocal-without-binding
    E0118, # used-prior-global-declaration
    C0121, # singleton-comparison
    C0123, # unidiomatic-typecheck
    C0200, # consider-using-enumerate
    C0201, # consider-iterating-dictionary
    C0325, # superfluous-parens
    C0326, # bad-whitespace
    C0330, # bad-continuation
    W0101, # unreachable
    W0102, # dangerous-default-value
    W0104, # pointless-statement
    W0105, # pointless-string-statement
    W0106, # expression-not-assigned
    W0107, # unnecessary-pass
    W0108, # unnecessary-lambda
    W0109, # duplicate-key
    W0110, # deprecated-lambda
    W0120, # useless-else-on-loop
    W0122, # exec-used
    W0123, # eval-used
    W0150, # lost-exception
    W0199, # assert-on-tuple
    W0301, # unnecessary-semicolon
    W0311, # bad-indentation
    W0312, # mixed-indentation
    W0401, # wildcard-import
    W0404, # reimported
    W0406, # import-self
    W0410, # misplaced-future
    R0123, # literal-comparison
    R0124, # comparison-with-itself
    R0133, # comparison-of-constants
    R0201, # no-self-use
    R0202, # no-classmethod-decorator
    R0203, # no-staticmethod-decorator
    R0205  # useless-object-inheritance
''')

            # Run pylint with JSON reporter and custom RC file
            output = StringIO()
            reporter = JSONReporter(output)
            
            # Run pylint on source file with custom RC
            Run([
                f'--rcfile={pylint_rc}',
                source_file
            ], reporter=reporter, exit=False)
            
            try:
                # Parse pylint results
                results = json.loads(output.getvalue())
                
                if results:
                    # Calculate score based on number of messages
                    num_messages = len([msg for msg in results if msg.get('type') in ('error', 'warning', 'convention', 'refactor')])
                    base_score = 10.0
                    deduction_per_message = 0.1
                    pylint_score = max(0.0, base_score - (num_messages * deduction_per_message))
                else:
                    # If no issues found, give a high score
                    pylint_score = 9.0
                
            except (json.JSONDecodeError, IndexError) as e:
                log.warning(f"Error parsing pylint output: {e}")
                pylint_score = 5.0  # Default score on error

            # Calculate maintainability index
            try:
                maintainability_raw = mi_visit(source_code, multi=True)
                if isinstance(maintainability_raw, (int, float)):
                    maintainability_score = maintainability_raw / 10.0
                maintainability_score = min(10.0, max(0.0, maintainability_score))
            except Exception as e:
                log.warning(f"Error calculating maintainability score: {e}")

    except Exception as e:
        log.error(f"Error in analyze_code_quality: {e}")

    # Calculate combined code quality score
    code_quality_score = (pylint_score + maintainability_score) / 2
    return pylint_score, maintainability_score, code_quality_score

def calculate_combined_score(complexity_score, pylint_score):
    """Calculates a combined score from complexity and pylint scores."""
    if complexity_score is None and pylint_score is None:
        return None
    elif complexity_score is None:
        return pylint_score / 10  # Normalize pylint score to 0-1 scale
    elif pylint_score is None:
        return complexity_score
    else:
        # Normalize pylint_score to 0-1 scale
        pylint_score_normalized = pylint_score / 10
        combined_score = (complexity_score + pylint_score_normalized) / 2
        return round(combined_score, 2)

def reformat_code(code):
    # Unescape escaped newlines
    code = code.replace('\\n', '\n')
    
    # Add newlines before keywords
    keywords = ['def ', 'if ', 'else:', 'elif ', 'for ', 'while ', 'return ', 'try:', 'except:', 'with ', 'class ', 'finally:']
    for kw in keywords:
        code = code.replace(kw, '\n' + kw)
    
    # Handle special case for except statements
    code = code.replace('except:', '\nexcept:')
    
    # Add newlines after colons if not followed by a newline
    code = re.sub(r':(?!\n)', ':\n', code)
    
    # Clean up any double newlines
    code = re.sub(r'\n\s*\n', '\n', code)
    
    return code


@perun.monitor()
def run_human_eval_function(func_str, test_cases):
    """Execute and profile a function from the HumanEval dataset."""
    try:
        # Create a restricted namespace for exec
        local_namespace = {}
        exec(func_str, {}, local_namespace)

        # Get the function name (assuming it's the first function defined in the string)
        func_name = func_str.split("def ")[1].split("(")[0]

        # Get the function object
        func = local_namespace.get(func_name)
        if not func:
            log.error(f"Function {func_name} not found in the provided code.")
            return

        # Run the function with test cases
        for left_expr, right_expr in test_cases:
            try:
                # Prepare the expressions
                # Replace 'candidate' with the actual function name in the AST
                left_code = ast.fix_missing_locations(replace_candidate_with_func(left_expr, func_name))
                right_code = ast.fix_missing_locations(replace_candidate_with_func(right_expr, func_name))

                # Compile the expressions
                left_func = compile(ast.Expression(body=left_code), filename="<ast>", mode="eval")
                right_func = compile(ast.Expression(body=right_code), filename="<ast>", mode="eval")

                # Evaluate expressions in the local namespace
                left_value = eval(left_func, {}, local_namespace)
                right_value = eval(right_func, {}, local_namespace)

                if left_value == right_value:
                    log.info(f"Test passed: {ast.unparse(left_expr)} == {ast.unparse(right_expr)}")
                else:
                    log.warning(f"Test failed: {ast.unparse(left_expr)} != {ast.unparse(right_expr)}")

            except Exception as err:
                log.error(f"Error executing test case: {err}")

    except Exception as err:
        log.error(f"Error in executing function code: {err}")

@perun.monitor()
def run_mbpp_function(func_code, test_list):
    """Executes the function with test cases."""
    try:
        # Add a small delay before starting to ensure power measurement is ready
        time.sleep(0.1)
        
        # Create a safe import function
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe version of __import__ that only allows specific modules"""
            allowed_modules = {
                'math', 'random', 'datetime', 'collections', 
                'itertools', 'functools', 'operator', 're',
                'string', 'copy', 'numbers', 'fractions', 'decimal',
                'statistics', 'bisect', 'heapq', 'array',
                'json', 'csv', 'typing', 'enum',
                # Add any other modules you want to allow
            }
            
            if name.split('.')[0] not in allowed_modules:
                raise ImportError(f"Import of {name} is not allowed")
            return __import__(name, globals, locals, fromlist, level)
        
        # Create a clean namespace with necessary built-ins
        safe_builtins = {
            # Basic types
            'tuple': tuple,
            'list': list,
            'dict': dict,
            'set': set,
            'frozenset': frozenset,
            'bool': bool,
            'int': int,
            'float': float,
            'complex': complex,
            'str': str,
            'bytes': bytes,
            'bytearray': bytearray,
            'bin': bin,
            'hex': hex,
            'oct': oct,
            'ord': ord,
            'chr': chr,
            
            # Built-in functions
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'iter': iter,
            'next': next,
            'slice': slice,
            'pow': pow,
            
            # Math operations
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'all': all,
            'any': any,
            
            # Type checking
            'isinstance': isinstance,
            'issubclass': issubclass,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'type': type,
            'callable': callable,
            
            # Import functionality
            '__import__': safe_import,
            
            # Other utilities
            'print': print,
            'repr': repr,
            'format': format,
            'dir': dir,
            'vars': vars,
            'id': id,
            'hash': hash,
            'help': help,
            'input': input,
            'open': open,
            'property': property,
            'staticmethod': staticmethod,
            'classmethod': classmethod,
            
            # Special attributes
            '__name__': '__main__',
            '__build_class__': __build_class__,
        }
        
        # Pre-import commonly used modules
        preloaded_modules = {
            'math': __import__('math'),
            're': __import__('re'),
            'itertools': __import__('itertools'),
            'collections': __import__('collections'),
            'functools': __import__('functools'),
            'operator': __import__('operator'),
            'random': __import__('random'),
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            'csv': __import__('csv'),
        }
        
        # Create the namespace with both builtins and preloaded modules
        namespace = {
            '__builtins__': safe_builtins,
            **preloaded_modules
        }
        
        # Execute the function code in the restricted namespace
        exec(func_code, namespace)
        
        # Run test cases
        for i, test_case in enumerate(test_list, 1):
            try:
                exec(test_case, namespace)
                log.info(f"Test case {i} passed.")
            except AssertionError:
                log.info(f"Test case {i} failed.")
            except Exception as e:
                log.info(f"Test case {i} raised an error: {str(e)}")
        
    except Exception as err:
        log.error(f"Error running function: {str(err)}")
    finally:
        # Add a small delay after execution
        time.sleep(0.1)

# Custom Energy Meter for macOS
class RAPLEnergyMeter:
    """Energy meter using Intel RAPL for Linux systems."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.energy_data = []
        self.method_energy_data = {}
        self._power_readings = []
        self._time_readings = []
        self._cpu_readings = []
        self._sampling_interval = 0.5
        self._keep_measuring = False
        self._thread = None
        
        # Check RAPL availability
        self._rapl_available = os.path.exists('/sys/class/powercap/intel-rapl')
        if not self._rapl_available:
            logging.info("RAPL not available, using CPU-based power estimation")
            
        # Find RAPL domains
        self._domains = []
        if self._rapl_available:
            try:
                rapl_dir = '/sys/class/powercap/intel-rapl'
                for domain in os.listdir(rapl_dir):
                    if domain.startswith('intel-rapl:'):
                        self._domains.append(os.path.join(rapl_dir, domain))
            except Exception as e:
                logging.error(f"Error finding RAPL domains: {e}")
                self._rapl_available = False

    def _read_rapl_energy(self):
        """Read energy values from RAPL."""
        total_energy = 0
        try:
            for domain in self._domains:
                energy_file = os.path.join(domain, 'energy_uj')
                if os.path.exists(energy_file):
                    with open(energy_file, 'r') as f:
                        # Convert microjoules to joules
                        energy_uj = int(f.read().strip())
                        total_energy += energy_uj / 1_000_000  # Convert to Joules
            return total_energy
        except Exception as e:
            logging.error(f"Error reading RAPL energy: {e}")
            return None

    def _measure_power(self):
        """Get power measurement using RAPL or fallback to CPU estimation."""
        if self._rapl_available:
            try:
                t1 = time.time()
                e1 = self._read_rapl_energy()
                time.sleep(self._sampling_interval)
                t2 = time.time()
                e2 = self._read_rapl_energy()
                
                if e1 is not None and e2 is not None:
                    power_w = (e2 - e1) / (t2 - t1)
                    return power_w * 1000  # Convert W to mW
            except Exception as e:
                logging.error(f"Error measuring RAPL power: {e}")
        
        # Fallback to CPU-based estimation
        cpu_percent = psutil.cpu_percent(interval=self._sampling_interval)
        power_watts = (cpu_percent / 100.0) * 15.0  # Assuming max 15W at 100% CPU
        return power_watts * 1000  # Convert W to mW

    def measure_energy(self, func):
        """Decorator to measure energy consumption of a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._keep_measuring:
                self.start()
                started_here = True
            else:
                started_here = False
            try:
                result = func(*args, **kwargs)
                if started_here:
                    energy = self.stop()
                    self.method_energy_data[func.__name__] = energy
                else:
                    energy = None
                return result
            except Exception as e:
                if started_here:
                    self.stop()
                raise e
            finally:
                if started_here:
                    self.reset()
        return wrapper


    def get_method_energy_data(self):
        """Returns the per-method energy consumption data."""
        return self.method_energy_data

    def start(self):
        """Start energy measurement."""
        self._power_readings = []
        self._time_readings = []
        self._cpu_readings = []  # Reset CPU readings
        self.start_time = time.time()
        
        psutil.cpu_percent(interval=self._sampling_interval)  # Initialize CPU percent measurement
        
        # Start background measurement thread
        self._keep_measuring = True
        self._thread = threading.Thread(target=self._continuous_measurement)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop measurement and calculate total energy."""
        self._keep_measuring = False
        if self._thread:
            self._thread.join(timeout=1)

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Calculate total energy using trapezoidal integration
        total_energy = 0.0
        if len(self._power_readings) > 1:
            for i in range(1, len(self._power_readings)):
                dt = self._time_readings[i] - self._time_readings[i-1]
                # Convert power from mW to W and calculate average power for the interval
                p1 = self._power_readings[i-1] / 1000.0  # Convert mW to W
                p2 = self._power_readings[i] / 1000.0    # Convert mW to W
                avg_power = (p1 + p2) / 2.0
                energy = avg_power * dt
                total_energy += energy
        
        # If no readings were collected or energy is 0, use fallback calculation
        if total_energy <= 0.0:
            # Get the last known power reading or measure new one
            if self._power_readings:
                last_power = self._power_readings[-1] / 1000.0  # Convert mW to W
            else:
                # Measure power directly
                power_mw = self._measure_power()
                last_power = power_mw / 1000.0  # Convert mW to W
            
            # Calculate energy using the power reading and duration
            total_energy = last_power * duration
            logging.debug(f"Using fallback energy calculation: {total_energy:.6f} Joules")

        # Ensure we never return 0 energy
        if total_energy <= 0.0:
            # Minimum energy estimation based on CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            min_power = (cpu_percent / 100.0) * 15.0  # Assuming max 15W at 100% CPU
            total_energy = min_power * duration
            logging.debug(f"Using minimum energy estimation: {total_energy:.6f} Joules")

        self.energy_data.append(total_energy)
        logging.debug(f"Final energy measurement: {total_energy:.6f} Joules")
        return total_energy

    def reset(self):
        """Reset all measurements."""
        self.start_time = None
        self.end_time = None
        self._power_readings = []
        self._time_readings = []
        self._cpu_readings = []
        self._keep_measuring = False
        self.energy_data = []

    def get_trace(self):
        """Get the energy consumption trace data."""
        return self.energy_data

    def _continuous_measurement(self):
        """Background thread for continuous power measurement."""
        try:
            last_time = time.time()
            while self._keep_measuring:
                current_time = time.time()
                power = self._measure_power()
                
                # Only record if we got a valid power reading
                if power > 0:
                    self._power_readings.append(power)
                    self._time_readings.append(current_time)
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self._cpu_readings.append(cpu_percent)
                    
                    # Add debug logging
                    logging.debug(f"Time: {current_time:.3f}, Power: {power:.2f} mW, CPU: {cpu_percent:.1f}%")
                    
                    last_time = current_time
                
                # Small sleep to prevent too frequent measurements
                time.sleep(self._sampling_interval)
                
        except Exception as e:
            logging.error(f"Exception in _continuous_measurement: {e}", exc_info=True)

def save_results_to_csv(filename='human_eval_profile.csv'):
    """Saves the profiling results to a CSV file."""
    if not profiling_results:
        log.warning("No profiling results to save.")
        return

    try:
        with open(filename, mode='w', newline='') as file:
            fieldnames = profiling_results[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in profiling_results:
                writer.writerow(result)
        log.info(f"Profiling results saved to {filename}")
    except Exception as err:
        log.error(f"Error saving results to CSV: {err}")

def generate_visualization(dataset_name):
    """Generates multiple visualizations for profiling results and saves them in a new folder."""
    if not profiling_results:
        log.warning("No data to visualize.")
        return
        
    if not PLOTTING_AVAILABLE:
        log.warning("Visualization skipped - matplotlib/seaborn not available")
        return

    try:
        # Create a new folder for visualizations
        folder_name = f'{dataset_name}_visualizations'
        os.makedirs(folder_name, exist_ok=True)

        df = pd.DataFrame(profiling_results)
        
        # Generate and save multiple visualizations
        
        # 1. Bar plots for main metrics
        metrics = ['execution_time', 'cpu_usage', 'memory_usage', 'cyclomatic_complexity', 'complexity_score', 'pylint_score', 'combined_score']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=metric, y='task_id', data=df)
            plt.title(f'{metric.replace("_", " ").title()} by Function')
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, f'{metric}_barplot.png'))
            plt.close()

        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'correlation_heatmap.png'))
        plt.close()

        # 3. Scatter plot: Execution Time vs Energy Consumed
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='execution_time', y='energy_consumed', data=df)
        plt.title('Execution Time vs Energy Consumed')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Energy Consumed (Joules)')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'execution_time_vs_energy.png'))
        plt.close()

        # 4. Box plot of main metrics
        plt.figure(figsize=(12, 6))
        df_melted = df[metrics].melt()
        sns.boxplot(x='variable', y='value', data=df_melted)
        plt.title('Distribution of Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'metrics_boxplot.png'))
        plt.close()

        # 5. Pairplot for key metrics
        sns.pairplot(df[['execution_time', 'cpu_usage', 'memory_usage', 'cyclomatic_complexity', 'combined_score']])
        plt.suptitle('Pairplot of Key Metrics', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'metrics_pairplot.png'))
        plt.close()

        log.info(f"Visualizations saved in folder: {folder_name}")
    except Exception as err:
        log.error(f"Error generating visualizations: {err}")

def profile_code(code_snippet: str) -> Optional[Dict[str, Any]]:
    """Profiles the given code snippet and returns performance metrics."""
    try:
        # Analyze code complexity
        complexity_value = analyze_cyclomatic_complexity(code_snippet)
        complexity_score = calculate_complexity_score(complexity_value)
        pylint_score, maintainability_score, code_quality_score = analyze_code_quality(code_snippet, '')
        combined_score = calculate_combined_score(complexity_score, pylint_score)

        start_time = time.time()
        energy_meter = RAPLEnergyMeter()
        energy_meter.start()

        # Create namespace with basic built-ins
        import builtins
        import importlib

        # Define unsafe built-ins
        unsafe_builtins = {
            'eval', 'exec', 'open', 'input', 'compile', 'exit', 'quit',
            '__import__', '__loader__', '__spec__', '__package__', '__file__', '__cached__'
        }
        
        # Include all built-ins except those explicitly marked as unsafe
        safe_builtins = {k: getattr(builtins, k) for k in dir(builtins)
                        if k not in unsafe_builtins}
        
        # Ensure __build_class__ is available for class definitions
        if '__build_class__' not in safe_builtins:
            safe_builtins['__build_class__'] = builtins.__build_class__

        # Create base namespace with safe built-ins
        namespace = {
            '__builtins__': safe_builtins,
        }

        # Try to import optional scientific computing libraries
        optional_modules = {
            'numpy': 'np',
            'pandas': 'pd',
            'scipy.stats': 'stats',
            'sklearn': 'sklearn',
            'matplotlib.pyplot': 'plt'
        }

        for module_name, alias in optional_modules.items():
            try:
                module = importlib.import_module(module_name)
                namespace[alias] = module
            except ImportError:
                logging.debug(f"Optional module {module_name} not available")

        # Add additional common modules that might be needed
        common_modules = {
            'math', 'random', 'datetime', 'collections', 
            'itertools', 'functools', 'operator', 're', 'numba'
        }

        for module_name in common_modules:
            try:
                module = importlib.import_module(module_name)
                namespace[module_name] = module
            except ImportError:
                logging.warning(f"Could not import {module_name}")

        # Parse and analyze imports from the code snippet
        def get_imported_modules(code_snippet):
            tree = ast.parse(code_snippet)
            modules = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        modules.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module is not None:
                        modules.add(node.module.split('.')[0])
            return modules

        # Get additional modules from the code
        modules_to_import = get_imported_modules(code_snippet)

        # Import additional modules that aren't already in namespace
        for module_name in modules_to_import:
            if (module_name not in namespace and 
                module_name not in {'os', 'sys', 'subprocess', 'socket', 'multiprocessing', 'threading'}):
                try:
                    module = importlib.import_module(module_name)
                    namespace[module_name] = module
                except ImportError:
                    logging.warning(f"Could not import {module_name}")

        # Remove import statements from the code snippet
        def remove_import_statements(code_snippet):
            tree = ast.parse(code_snippet)
            new_body = [node for node in tree.body 
                       if not isinstance(node, (ast.Import, ast.ImportFrom))]
            tree.body = new_body
            return ast.unparse(tree)

        code_snippet = remove_import_statements(code_snippet)

        # Execute the code
        try:
            exec(code_snippet, namespace)
        except ImportError as e:
            logging.warning(f"Import error while executing code: {e}")
            # Continue execution despite import errors
        except Exception as e:
            logging.error(f"Error executing code snippet: {e}", exc_info=True)
            return None

        energy_meter.stop()
        energy_data = energy_meter.get_trace()
        energy_consumed = energy_data[-1] if energy_data else "N/A"
        energy_meter.reset()

        end_time = time.time()
        execution_time = end_time - start_time
        cpu_percent, memory_percent = get_cpu_memory_usage()
        power_data = read_power_data()

        return {
            'execution_time': execution_time,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'power_data': power_data,
            'energy_consumed': energy_consumed,
            'cyclomatic_complexity': complexity_value,
            'complexity_score': complexity_score,
            'pylint_score': pylint_score,
            'maintainability_score': maintainability_score,
            'code_quality_score': code_quality_score,
            'combined_score': combined_score
        }

    except Exception as e:
        logging.error(f"Error profiling code: {e}", exc_info=True)
        return None

@perun.monitor()
def main():
    """Main function to orchestrate profiling for HumanEval or MBPP++ dataset."""
    parser = argparse.ArgumentParser(description='Method-level energy profiling script.')
    parser.add_argument('--dataset', choices=['humaneval', 'evalplus'], required=True,
                        help='Select the dataset to profile: humaneval or evalplus.')
    args = parser.parse_args()

    dataset_name = args.dataset

    try:
        # Initialize the custom energy meter
        energy_meter = RAPLEnergyMeter()

        if dataset_name == 'humaneval':
            # Load the HumanEval dataset
            dataset = load_dataset("openai_humaneval", split="test")
            log.info("Starting energy consumption test for HumanEval functions...")

            for idx, data_point in enumerate(dataset):
                log.info(f"\nTesting function from HumanEval/{idx}")

                # Extract function and test cases
                func_str = data_point['prompt'] + data_point['canonical_solution']
                test_cases_str = data_point['test']

                test_cases = extract_test_cases(test_cases_str)

                if not test_cases:
                    log.warning(f"No test cases extracted for HumanEval/{idx}. Skipping.")
                    continue

                # Analyze cyclomatic complexity
                complexity_value = analyze_cyclomatic_complexity(func_str)
                complexity_score = calculate_complexity_score(complexity_value)
                pylint_score, pyflakes_score, code_quality_score = analyze_code_quality(func_str, data_point['test'])
                combined_score = calculate_combined_score(complexity_score, code_quality_score)

                # Run and profile the function
                start_time = time.time()

                energy_meter.start()
                run_human_eval_function(func_str, test_cases)
                energy_meter.stop()

                energy_data = energy_meter.get_trace()
                energy_consumed = energy_data[-1] if energy_data else "N/A"
                energy_meter.reset()

                end_time = time.time()

                execution_time = end_time - start_time
                cpu_percent, memory_percent = get_cpu_memory_usage()
                power_data = read_power_data()

                profiling_results.append({
                    'task_id': data_point['task_id'],
                    'prompt': data_point['prompt'],
                    'canonical_solution': data_point['canonical_solution'],
                    'test': data_point['test'],
                    'entry_point': data_point['entry_point'],
                    'execution_time': execution_time,
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'power_data': power_data,
                    'energy_consumed': energy_consumed,
                    'cyclomatic_complexity': complexity_value,
                    'complexity_score': complexity_score,
                    'pylint_score': pylint_score,
                    'maintainability_score': pyflakes_score,  # Changed from pyflakes_score
                    'code_quality_score': code_quality_score,
                    'combined_score': combined_score
                })
                log.info("Execution time: {:.6f} seconds".format(execution_time))
                log.info(f"CPU Usage: {cpu_percent}%".format(cpu_percent))
                log.info(f"Memory Usage: {memory_percent}%")
                log.info(f"Power Data: {power_data}")
                log.info(f"Energy Consumed: {energy_consumed} Joules")
                log.info(f"Cyclomatic Complexity: {complexity_value}")
                log.info(f"Complexity Score: {complexity_score}")
                log.info(f"Pylint Score: {pylint_score:.2f}")
                log.info(f"Maintainability Score: {pyflakes_score:.2f}")  # Changed from Pyflakes Score
                log.info(f"Code Quality Score: {code_quality_score:.2f}")
                log.info(f"Combined Score: {combined_score:.2f}")

                time.sleep(1)  # Add a small delay between functions

            # Save results to CSV
            save_results_to_csv(filename='human_eval_profile.csv')
            generate_visualization('HumanEval')

        elif dataset_name == 'evalplus':
            # Load the MBPP++ (EvalPlus) dataset
            dataset = load_dataset("evalplus/mbppplus", split="test")
            log.info("Starting energy consumption test for EvalPlus functions...")

            for idx, data_point in enumerate(dataset):
                log.info(f"\nTesting function from EvalPlus/{idx}")

                func_code = data_point['code']
                prompt = data_point['prompt']
                source_file = data_point.get('source_file', '')
                test_imports = data_point.get('test_imports', [])
                test_code = data_point['test']
                test_list = data_point['test_list']
                
                # Analyze cyclomatic complexity
                complexity_value = analyze_cyclomatic_complexity(func_code)
                complexity_score = calculate_complexity_score(complexity_value)
                pylint_score, pyflakes_score, code_quality_score = analyze_code_quality(func_code, data_point['test'])
                combined_score = calculate_combined_score(complexity_score, code_quality_score)

                # Run and profile the function
                start_time = time.time()

                energy_meter.start()
                run_mbpp_function(func_code, test_list)
                energy_meter.stop()

                energy_data = energy_meter.get_trace()
                energy_consumed = energy_data[-1] if energy_data else "N/A"
                energy_meter.reset()

                end_time = time.time()

                execution_time = end_time - start_time
                cpu_percent, memory_percent = get_cpu_memory_usage()
                power_data = read_power_data()

                profiling_results.append({
                    'task_id': idx,
                    'code': func_code,
                    'prompt': prompt,
                    'source_file': source_file,
                    'test_imports': test_imports,
                    'test_code': test_code,
                    'execution_time': execution_time,
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'power_data': power_data,
                    'energy_consumed': energy_consumed,
                    'cyclomatic_complexity': complexity_value,
                    'complexity_score': complexity_score,
                    'pylint_score': pylint_score,
                    'maintainability_score': pyflakes_score,  # Changed from pyflakes_score
                    'code_quality_score': code_quality_score,
                    'combined_score': combined_score
                })

                log.info(f"Execution time: {execution_time:.6f} seconds")
                log.info(f"CPU Usage: {cpu_percent}%")
                log.info(f"Memory Usage: {memory_percent}%")
                log.info(f"Power Data: {power_data}")
                log.info(f"Energy Consumed: {energy_consumed} Joules")
                log.info(f"Cyclomatic Complexity: {complexity_value}")
                log.info(f"Complexity Score: {complexity_score}")
                log.info(f"Pylint Score: {pylint_score:.2f}")
                log.info(f"Maintainability Score: {pyflakes_score:.2f}")  # Changed from Pyflakes Score
                log.info(f"Code Quality Score: {code_quality_score:.2f}")
                log.info(f"Combined Score: {combined_score:.2f}")

                time.sleep(1)  # Add a small delay between functions

            # Save results to CSV
            save_results_to_csv(filename='evalplus_profile.csv')
            generate_visualization('EvalPlus')

        else:
            log.error(f"Unsupported dataset: {dataset_name}")

    except Exception as err:
        log.error(f"An error occurred in main: {err}")

if __name__ == "__main__":
    log.info("Starting the method-level energy profiling process...")
    main()
    log.info("Profiling completed.")

