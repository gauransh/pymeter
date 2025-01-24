import json
import logging
import requests
import os
import csv
from difflib import unified_diff
from typing import Optional, Dict, Any, List, Tuple
from string import Template  # Import Template for prompt formatting
from datetime import datetime
import time
import sys
import ast
import tempfile
import re
import math
import signal
from contextlib import contextmanager
import subprocess
import importlib

# Import all required functions from energy_profile
from energy_profile import (
    RAPLEnergyMeter,
    get_cpu_memory_usage,
    analyze_cyclomatic_complexity,
    calculate_complexity_score,
    analyze_code_quality,
    analyze_maintainability,
    profile_code
)

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/2)

# Update the logging configuration near the top of the file (around line 34)
def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create different log files for different levels
    log_files = {
        'debug': os.path.join(logs_dir, f'debug_{timestamp}.log'),
        'info': os.path.join(logs_dir, f'info_{timestamp}.log'),
        'error': os.path.join(logs_dir, f'error_{timestamp}.log'),
        'profile': os.path.join(logs_dir, f'profile_{timestamp}.log')
    }

    # Configure root logger
    logging.basicConfig(level=logging.DEBUG)
    root_logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )

    # Debug file handler (catches all levels)
    debug_handler = logging.FileHandler(log_files['debug'])
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_handler)

    # Info file handler
    info_handler = logging.FileHandler(log_files['info'])
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(simple_formatter)
    root_logger.addHandler(info_handler)

    # Error file handler
    error_handler = logging.FileHandler(log_files['error'])
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Profile file handler (for metrics and analysis)
    profile_handler = logging.FileHandler(log_files['profile'])
    profile_handler.setLevel(logging.INFO)
    profile_handler.setFormatter(simple_formatter)
    # Only log profiling-related messages
    profile_handler.addFilter(lambda record: 'profile' in record.getMessage().lower() 
                            or 'metrics' in record.getMessage().lower()
                            or 'analysis' in record.getMessage().lower())
    root_logger.addHandler(profile_handler)

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    return root_logger

# Initialize logging
logger = setup_logging()

PROMPT_TEMPLATE = Template("""
Optimize the following code for **energy efficiency** based on its energy profile and code quality metrics:

--- BEGIN CODE ---
$code
--- END CODE ---

**Pre-Execution Analysis:**
$pre_execution_analysis

**Energy Profile:**
- **Energy Consumed**: $energy_consumed Joules
- **Execution Time**: $execution_time seconds
- **CPU Usage**: $cpu_usage%
- **Memory Usage**: $memory_usage%

**Code Quality Metrics:**
- **Cyclomatic Complexity**: $complexity_value
- **Complexity Score**: $complexity_score
- **Pylint Score**: $pylint_score
- **Maintainability Score**: $maintainability_score
- **Code Quality Score**: $code_quality_score
- **Combined Score**: $combined_score

**Previous Optimizations:**
$optimization_history

**Best Practices from History:**
$best_practices

---

### **Thought Preference Optimization (TPO) Analysis**

As an AI language model utilizing **Thought Preference Optimization (TPO)**, generate multiple internal optimization strategies with the sole focus on **maximizing energy efficiency** of the code. Internally evaluate each strategy using **Tree of Thoughts (ToT)** analysis and **energy-focused heuristic evaluation**.

### **Internal Thought Processes**

**1. Strategy 1:**
- **Description**: [Describe the first optimization strategy aimed at reducing energy consumption]
- **ToT Analysis:**
  - **Branching Thoughts**:
    - **Idea 1**: [First idea within this strategy to improve energy efficiency]
    - **Idea 2**: [Second idea]
  - **Evaluation of Branches**: Assess each idea's potential impact on energy consumption reduction.
- **Energy-Focused Heuristic Evaluation:**
  - **H1 (Maximize Energy Efficiency)**: [Assess how significantly the strategy reduces energy consumption]
  - **H2 (Resource Utilization)**: [Evaluate the impact on CPU and memory usage as it relates to energy efficiency]
  - **H3 (Feasibility and Complexity)**: [Assess the ease of implementation considering energy gains]
- **Overall Evaluation**:
  - **Energy Efficiency Improvement**: [High/Medium/Low]
  - **Potential Drawbacks**: [List any drawbacks that might affect energy efficiency]

**2. Strategy 2:**
- **Description**: [Describe the second optimization strategy aimed at reducing energy consumption]
- **ToT Analysis:**
  - **Branching Thoughts**:
    - **Idea 1**: [First idea]
    - **Idea 2**: [Second idea]
  - **Evaluation of Branches**: Assess each idea's potential impact on energy consumption reduction.
- **Energy-Focused Heuristic Evaluation:**
  - **H1 (Maximize Energy Efficiency)**: [Assess the energy reduction potential]
  - **H2 (Resource Utilization)**: [Evaluate CPU/memory impacts related to energy]
  - **H3 (Feasibility and Complexity)**: [Assess implementation feasibility]
- **Overall Evaluation**:
  - **Energy Efficiency Improvement**: [High/Medium/Low]
  - **Potential Drawbacks**: [List any drawbacks]

**3. Strategy 3:**
- **Description**: [Describe the third optimization strategy aimed at reducing energy consumption]
- **ToT Analysis:**
  - **Branching Thoughts**:
    - **Idea 1**: [First idea]
    - **Idea 2**: [Second idea]
  - **Evaluation of Branches**: Assess each idea's potential impact on energy consumption reduction.
- **Energy-Focused Heuristic Evaluation:**
  - **H1 (Maximize Energy Efficiency)**: [Assess the energy reduction potential]
  - **H2 (Resource Utilization)**: [Evaluate CPU/memory impacts related to energy]
  - **H3 (Feasibility and Complexity)**: [Assess implementation feasibility]
- **Overall Evaluation**:
  - **Energy Efficiency Improvement**: [High/Medium/Low]
  - **Potential Drawbacks**: [List any drawbacks]

### **Selected Strategy**

Based on the internal evaluations, **select the strategy that offers the greatest reduction in energy consumption**, even if it may require trade-offs in performance or code complexity.

### **Optimized Code**

Provide the complete optimized code implementing the selected strategy. Ensure that the code is syntactically correct and includes a full function implementation:


```python
### $file_path
<<<<< SEARCH
[existing code]
======
[optimized code]
>>>>>> REPLACE
```


### **Performance Analysis Categories (Weighted by Impact)**

**1. Unnecessary Computations (30%)**
- Redundant calculations
- Loop inefficiencies
- Dead code
- Caching opportunities
- Built-in function usage

**2. Expensive Operations (25%)**
- Algorithm complexity
- Type conversions
- String operations
- Non-vectorized operations
- Library usage

**3. Data Structure Efficiency (20%)**
- Container selection
- Memory layout
- Object overhead
- Lookup mechanisms
- Memory contiguity

**4. Disk I/O Optimization (15%)**
- I/O batching
- File handling
- Serialization
- Buffer management
- Memory mapping

**5. Thread Synchronization (10%)**
- Lock contention
- Critical sections
- Thread pooling
- Synchronization primitives
- Inter-thread communication


### **Explanation**

Please provide your explanation in this section, structured as follows:

**1. Reasoning Behind Selection:**
- Explain why this strategy was chosen
- Reference specific patterns from pre-execution analysis
- Justify the selection based on energy efficiency potential

**2. Applied Energy-Efficient Techniques:**
- List and describe each technique applied
- Explain how each technique contributes to energy reduction
- Detail any specific optimizations made

**3. Expected Improvements:**
- **Energy Consumption**: [Quantify expected reduction]
- **Resource Utilization**: [Describe CPU/memory impact]
- **Overall Efficiency**: [Summarize total expected improvement]

**4. Trade-offs and Mitigation:**
- **Identified Trade-offs**: [List any compromises made]
- **Mitigation Strategies**: [Explain how downsides are handled]
- **Justification**: [Why the energy savings outweigh the trade-offs]

### **Energy-Efficient Techniques Applied**

Indicate which energy-efficient techniques were applied from the following list:

- Algorithmic optimizations
- Memory access patterns
- I/O efficiency
- Parallel processing
- Resource pooling
- Lazy evaluation
- Power-aware strategies

---

**Note**: The primary goal is to **minimize the energy consumption** of the code. Other factors such as performance and code quality should only be considered insofar as they impact energy efficiency. If necessary, acceptable trade-offs can be made in these areas to achieve significant energy savings. se the pre-execution analysis insights to target the most impactful optimizations.
**Note**: Your response **MUST** include the **Explanation** section with all four subsections clearly formatted as shown above with the optimized code.
**Note**: The code generated **MUST** be syntactically correct and include a **full function implementation**
**Note**: The code generated **MUST** be a valid Python function definition with the exact signature as the original code.
""")

# Add this near the top of the file, after imports
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float."""
    if value == '' or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

class OptimizationHistoryBuffer:
    def __init__(self):
        self.history = []
        self.best_practices = {
            'energy_efficiency': [
                'Minimize I/O operations',
                'Use efficient data structures',
                'Avoid unnecessary computations',
                'Optimize loops and iterations',
                'Reduce memory allocations'
            ],
            'code_quality': [
                'Follow PEP 8 style guide',
                'Use meaningful variable names',
                'Keep functions small and focused',
                'Add appropriate documentation',
                'Handle errors gracefully'
            ]
        }
        self.quality_metrics = {
            'pylint_score': 0.0,
            'maintainability_score': 0.0,
            'code_quality_score': 0.0,
            'complexity_score': 0.0
        }

    def add_optimization(self, original_code: str, optimized_code: str, 
                        original_metrics: Dict[str, float], 
                        optimized_metrics: Dict[str, float],
                        explanation: str, evaluation_score: float,
                        test_results: Optional[Dict[str, Any]] = None):
        """Add an optimization attempt to the history."""
        self.history.append({
            'original_code': original_code,
            'optimized_code': optimized_code,
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'explanation': explanation,
            'evaluation_score': evaluation_score,
            'test_results': test_results
        })

    def get_history_summary(self) -> str:
        """Generate a summary of the optimization history."""
        if not self.history:
            return "No optimization history available."

        latest = self.history[-1]
        orig = latest['original_metrics']
        opt = latest['optimized_metrics']

        # Calculate improvements
        energy_improvement = safe_float(orig.get('energy_consumed', 0)) - safe_float(opt.get('energy_consumed', 0))
        time_improvement = safe_float(orig.get('execution_time', 0)) - safe_float(opt.get('execution_time', 0))
        cpu_improvement = safe_float(orig.get('cpu_usage', 0)) - safe_float(opt.get('cpu_usage', 0))
        memory_improvement = safe_float(orig.get('memory_usage', 0)) - safe_float(opt.get('memory_usage', 0))

        summary = "- Optimization achieved:\n"
        summary += f"  * Energy: {energy_improvement:.3f}J\n"
        summary += f"  * Time: {time_improvement:.3f}s\n"
        summary += f"  * CPU: {cpu_improvement:.2f}%\n"
        summary += f"  * Memory: {memory_improvement:.2f}%\n"
        summary += f"  * Evaluation Score: {latest['evaluation_score']:.2f}\n"
        
        # Add test results if available
        if latest.get('test_results'):
            test_results = latest['test_results']
            summary += f"  * Pass Rate: {test_results.get('pass_rate', 0)}%\n"
            summary += f"  * Tests: {test_results.get('passed_tests', 0)}/{test_results.get('total_tests', 0)}\n"
        
        summary += f"  Summary: {latest['explanation']}....\n"
        
        return summary

    def get_best_practices(self) -> Dict[str, List[str]]:
        """Return the best practices for energy efficiency and code quality."""
        return self.best_practices

    def get_quality_metrics(self) -> Dict[str, float]:
        """Return the current quality metrics."""
        return self.quality_metrics

    def update_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """Update the quality metrics."""
        self.quality_metrics.update(metrics)

def calculate_combined_score(complexity_score: float, code_quality_score: float) -> float:
    """Calculate combined score from complexity and code quality scores."""
    # Weight factors (can be adjusted)
    complexity_weight = 0.4
    quality_weight = 0.6
    
    # Normalize scores to 0-1 range if needed
    normalized_complexity = min(max(complexity_score / 10.0, 0), 1)
    normalized_quality = min(max(code_quality_score / 10.0, 0), 1)
    
    # Calculate weighted average
    combined_score = (complexity_weight * normalized_complexity + 
                     quality_weight * normalized_quality)
    
    return round(combined_score * 10, 2)  # Return on 0-10 scale

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """Normalize a score to a 0-10 scale."""
    try:
        # Handle edge cases
        if score is None or math.isnan(score):
            return 0.0
        
        # Clamp the score between min and max values
        clamped = max(min(score, max_val), min_val)
        
        # Normalize to 0-10 scale
        normalized = ((clamped - min_val) / (max_val - min_val)) * 10
        return round(normalized, 2)
    except Exception as e:
        logger.error(f"Error normalizing score: {e}")
        return 0.0

def evaluate_repair(original_metrics: Dict[str, float], 
                   optimized_metrics: Dict[str, float],
                   test_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate repair effectiveness with improved error handling."""
    try:
        # Calculate improvements with safety checks
        improvements = {}
        for metric in ['energy_consumed', 'execution_time', 'cpu_usage', 'memory_usage']:
            orig_val = float(original_metrics.get(metric, 0))
            opt_val = float(optimized_metrics.get(metric, 0))
            
            # Calculate improvement (negative means better)
            improvement = orig_val - opt_val
            
            # Store raw improvement value
            improvements[f'{metric}_improvement'] = improvement
            
            # Calculate percentage improvement with safety check for zero
            if orig_val > 0:
                pct_improvement = (improvement / orig_val) * 100
            else:
                # If original value is 0, use absolute improvement
                pct_improvement = -improvement * 100 if improvement != 0 else 0
                
            improvements[f'{metric}_pct_improvement'] = pct_improvement

        # Calculate normalized scores (0-10 scale)
        normalized_scores = {}
        
        # Energy score (higher is better)
        energy_improvement = improvements['energy_consumed_improvement']
        if original_metrics.get('energy_consumed', 0) > 0:
            energy_score = min(10, max(0, (energy_improvement / original_metrics['energy_consumed']) * 10))
        else:
            energy_score = 10 if energy_improvement > 0 else 0
        normalized_scores['energy_score'] = energy_score

        # Performance score
        exec_improvement = improvements['execution_time_improvement']
        if original_metrics.get('execution_time', 0) > 0:
            perf_score = min(10, max(0, (exec_improvement / original_metrics['execution_time']) * 10))
        else:
            perf_score = 10 if exec_improvement > 0 else 0
        normalized_scores['performance_score'] = perf_score

        # Resource scores
        normalized_scores['cpu_score'] = min(10, max(0, improvements['cpu_usage_pct_improvement'] / 10))
        normalized_scores['memory_score'] = min(10, max(0, improvements['memory_usage_pct_improvement'] / 10))
        
        # Code quality score (already on 0-10 scale)
        normalized_scores['quality_score'] = float(optimized_metrics.get('code_quality_score', 0))

        # Test results score
        if test_results and test_results.get('total_tests', 0) > 0:
            test_score = (test_results['passed_tests'] / test_results['total_tests']) * 10
        else:
            test_score = 0
        normalized_scores['test_score'] = test_score

        # Calculate final weighted score
        weights = {
            'energy': 0.25,
            'performance': 0.20,
            'resources': 0.15,
            'quality': 0.20,
            'correctness': 0.20
        }

        resource_score = (normalized_scores['cpu_score'] + normalized_scores['memory_score']) / 2
        
        final_score = (
            weights['energy'] * normalized_scores['energy_score'] +
            weights['performance'] * normalized_scores['performance_score'] +
            weights['resources'] * resource_score +
            weights['quality'] * normalized_scores['quality_score'] +
            weights['correctness'] * normalized_scores['test_score']
        )

        return {
            'improvements': improvements,
            'normalized_scores': normalized_scores,
            'test_results': test_results,
            'final_score': round(final_score, 2)
        }

    except Exception as e:
        logger.error(f"Error in evaluate_repair: {str(e)}")
        return {
            'improvements': {},
            'normalized_scores': {
                'energy_score': 0,
                'performance_score': 0,
                'cpu_score': 0,
                'memory_score': 0,
                'quality_score': 0,
                'test_score': 0
            },
            'test_results': None,
            'final_score': 0.0
        }

def run_test_cases(code: str, test_code: str) -> Dict[str, Any]:
    """Run test cases on the code using the provided test code."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the solution code to 'solution.py'
            solution_file_path = os.path.join(temp_dir, 'solution.py')
            with open(solution_file_path, 'w') as solution_file:
                solution_file.write(code)

            # Write the test code to 'test_solution.py'
            test_file_path = os.path.join(temp_dir, 'test_solution.py')
            with open(test_file_path, 'w') as test_file:
                test_file.write("import solution\n")
                test_file.write(test_code)

            # Adjust sys.path to include the temporary directory
            sys.path.insert(0, temp_dir)

            # Execute the test code
            spec = importlib.util.spec_from_file_location("test_solution", test_file_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Clean up sys.path
            sys.path.pop(0)

            # If execution reaches this point, tests have passed
            return {
                'passed_tests': 1,
                'total_tests': 1,
                'pass_rate': 100.0,
                'test_results': [{'test_id': 'custom', 'status': 'passed', 'error': ''}]
            }
    except Exception as e:
        logger.error(f"Error running test cases: {e}")
        return {
            'passed_tests': 0,
            'total_tests': 1,
            'pass_rate': 0.0,
            'test_results': [{'test_id': 'custom', 'status': 'failed', 'error': str(e)}]
        }


def log_test_results(test_results: Optional[Dict[str, Any]], context: str = ""):
    """Log test results with detailed information."""
    if not test_results:
        logger.warning(f"No test results available for {context}")
        return

    logger.info(f"\n=== Test Results for {context} ===")
    logger.info(f"Pass Rate: {test_results.get('pass_rate', 0):.2f}%")
    logger.info(f"Tests Passed: {test_results.get('passed_tests', 0)}/{test_results.get('total_tests', 0)}")

    # Log individual test results
    test_details = test_results.get('test_results', [])
    if test_details:
        logger.info("\nDetailed Test Results:")
        for test in test_details:
            status = "✅" if test.get('status') == 'passed' else "❌"
            logger.info(f"{status} {test.get('test_id', 'Unknown Test')}")
            if test.get('status') == 'failed':
                logger.info(f"   Error: {test.get('error', 'No error message available')}")
    
    logger.info("=" * 40)

def generate_repair(code_snippet: str, energy_profile_data: Dict[str, Any], 
                   file_path: str, history_buffer: OptimizationHistoryBuffer,
                   test_code: str = None,
                   test_imports: List[str] = None,
                   model: str = 'llama3.1', 
                   temperature: float = 0.3, 
                   top_p: float = 1.0, 
                   max_tokens: Optional[int] = None,
                   max_attempts: int = 5) -> Optional[Dict[str, Any]]:
    try:
        # Add pre-execution analysis
        logger.info("Performing pre-execution analysis...")
        pre_exec_analysis = pre_execution_analysis(code_snippet)
        analysis_summary = generate_analysis_summary(pre_exec_analysis)
        logger.info("\nPre-execution Analysis Summary:")
        logger.info(analysis_summary)

        # Profile the original code first
        logger.info("Profiling original code...")
        original_metrics = profile_code_execution(code_snippet)
        if not original_metrics:
            logger.error("Failed to profile original code")
            return None

        # Report original code metrics
        logger.info("\nOriginal Code Metrics:")
        logger.info(f"- Energy: {original_metrics['energy_consumed']:.3f}J")
        logger.info(f"- Execution Time: {original_metrics['execution_time']:.3f}s")
        logger.info(f"- CPU Usage: {original_metrics['cpu_usage']:.2f}%")
        logger.info(f"- Memory Usage: {original_metrics['memory_usage']:.2f}%")
        logger.info(f"- Cyclomatic Complexity: {original_metrics['cyclomatic_complexity']}")
        logger.info(f"- Complexity Score: {original_metrics['complexity_score']:.2f}")
        logger.info(f"- Pylint Score: {original_metrics['pylint_score']:.2f}")
        logger.info(f"- Maintainability Score: {original_metrics['maintainability_score']:.2f}")
        logger.info(f"- Code Quality Score: {original_metrics['code_quality_score']:.2f}")

        # Run test cases on original code if available
        original_test_results = None
        if test_code:
            logger.info("Running test cases on original code...")
            original_test_results = run_test_cases(code_snippet, test_code)
            if original_test_results:
                log_test_results(original_test_results, "Original Code")
            else:
                logger.warning("Failed to run tests on original code")

        # Format test cases for the prompt
        test_cases_str = ""
        if test_code:
            test_cases_str = "\nTest Cases to Pass:\n"
            for i, test in enumerate(test_code.split('\n'), 1):
                test_cases_str += f"Test {i}:\n{test}\n"

        all_valid_repairs = []
        all_valid_explanations = []
        all_optimized_metrics = []
        all_test_results = []
        all_evaluations = []

        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            logger.info(f"\nAttempt {attempts} of {max_attempts}")

            try:
                # Adjust temperature for each attempt to get diverse solutions
                current_temperature = temperature * (1 + (attempts - 1) * 0.125)
                logger.info(f"Using temperature: {current_temperature:.2f}")

                # Generate optimization prompt with test cases
                prompt = PROMPT_TEMPLATE.safe_substitute(
                    code=code_snippet,
                    energy_consumed=energy_profile_data['energy_consumed'],
                    execution_time=energy_profile_data['execution_time'],
                    cpu_usage=energy_profile_data['cpu_usage'],
                    memory_usage=energy_profile_data['memory_usage'],
                    complexity_value=original_metrics.get('cyclomatic_complexity', 'N/A'),
                    complexity_score=original_metrics.get('complexity_score', 'N/A'),
                    pylint_score=original_metrics.get('pylint_score', 'N/A'),
                    maintainability_score=original_metrics.get('maintainability_score', 'N/A'),
                    code_quality_score=original_metrics.get('code_quality_score', 'N/A'),
                    combined_score=original_metrics.get('combined_score', 'N/A'),
                    file_path=file_path,
                    optimization_history=history_buffer.get_history_summary(),
                    best_practices=history_buffer.get_best_practices(),
                    test_cases=test_cases_str,
                    pre_execution_analysis=analysis_summary
                )

                # Make API request
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": current_temperature,
                            "top_p": top_p,
                            "num_predict": max_tokens if max_tokens else None
                        }
                    }),
                    stream=True
                )
                response.raise_for_status()

                # Process response and extract repairs
                repairs, explanations = extract_repairs_and_explanations(response)
                
                if not repairs:
                    logger.warning(f"No repairs found in attempt {attempts}")
                    continue

                logger.info(f"Found {len(repairs)} potential repairs in attempt {attempts}")

                # Log repairs before validation
                for i, (repair, explanation) in enumerate(zip(repairs, explanations), 1):
                    test_results = None
                    logger.info(f"\nRepair {i} from attempt {attempts}:")
                    logger.info("=" * 40)
                    logger.info("Code:")
                    logger.info(repair)
                    logger.info("\nExplanation:")
                    logger.info(explanation)
                    logger.info("=" * 40)

                    logger.info(f"\nValidating repair {i} from attempt {attempts}...")
                    
                    # Run test cases if available
                
                    if test_code:
                        logger.info(f"Running tests for repair {i}...")
                        test_results = run_test_cases(repair, test_code)
                        if test_results:
                            log_test_results(test_results, f"Repair {i}")
                            if test_results['pass_rate'] < 100:
                                logger.warning(f"Repair {i} failed test cases - skipping")
                                continue
                            logger.info(f"Repair {i} passed all test cases")
                        else:
                            logger.warning(f"No test results available for repair {i} - skipping")
                            continue

                    # Profile the repair
                    optimized_metrics = profile_code(repair)
                    if not optimized_metrics or 'energy_consumed' not in optimized_metrics:
                        logger.warning(f"Failed to profile repair {i} or missing 'energy_consumed' - skipping")
                        continue

                    # Compare with original metrics
                    if 'energy_consumed' not in original_metrics:
                        logger.error("Original metrics missing 'energy_consumed'")
                        continue
                    evaluation = evaluate_repair(original_metrics, optimized_metrics, test_results)
                    
                    # Store valid repair
                    all_valid_repairs.append(repair)
                    all_valid_explanations.append(explanation)
                    all_optimized_metrics.append(optimized_metrics)
                    all_test_results.append(test_results)  # This might be None if no tests
                    all_evaluations.append(evaluation)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed on attempt {attempts}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error in attempt {attempts}: {e}")
                continue

        # If we found any valid repairs, select the best one
        if all_valid_repairs:
            logger.info(f"\nFound {len(all_valid_repairs)} total valid repairs across {attempts} attempts")
            
            # Select the best repair based on energy consumption improvement
            best_idx = max(
                range(len(all_evaluations)),
                key=lambda i: all_evaluations[i]['improvements'].get('energy_consumed_improvement', 0)
            )
            
            logger.info(
                f"\nSelected best repair with "
                f"{all_evaluations[best_idx]['improvements']['energy_consumed_improvement']:.1f}J energy improvement"
            )
            
            return {
                "repairs": [all_valid_repairs[best_idx]],
                "explanations": [all_valid_explanations[best_idx]],
                "optimized_metrics": [all_optimized_metrics[best_idx]],
                "original_metrics": original_metrics,
                "evaluations": [all_evaluations[best_idx]],
                "test_results": [all_test_results[best_idx]],
                "all_repairs": {
                    "repairs": all_valid_repairs,
                    "explanations": all_valid_explanations,
                    "metrics": all_optimized_metrics,
                    "evaluations": all_evaluations,
                    "test_results": all_test_results
                }
            }


        logger.error(f"Failed to generate valid repair after {max_attempts} attempts")
        return None

    except Exception as e:
        logger.error(f"Error in generate_repair: {e}")
        return None

def extract_code_from_repair(repair_text: str) -> Optional[str]:
    """Extract the optimized code from the repair text."""
    try:
        # Find the section between SEARCH/REPLACE markers
        if '<<<<< SEARCH' in repair_text and '>>>>>> REPLACE' in repair_text:
            # Split by markers and get the REPLACE section
            parts = repair_text.split('=======')
            if len(parts) >= 2:
                replace_section = parts[1].split('>>>>>> REPLACE')[0].strip()
                return replace_section
        return None
    except Exception as e:
        logging.error(f"Error extracting code from repair: {e}")
        return None

def save_repair_to_csv(row_data: Dict[str, Any], repair_result: Dict[str, Any], output_file: str):
    try:
        file_exists = os.path.exists(output_file)
        
        # Update fieldnames to include method_energy_improvement
        fieldnames = [
            'timestamp', 'task_id', 'original_code', 
            'energy_consumed', 'power_draw', 'execution_time', 
            'cpu_usage', 'memory_usage',
            'optimized_code', 'optimization_explanation',
            'optimized_energy_consumed', 'optimized_power_draw',
            'optimized_execution_time', 'optimized_cpu_usage',
            'optimized_memory_usage',
            'energy_improvement', 'power_improvement',
            'time_improvement', 'cpu_improvement',
            'memory_improvement', 'method_energy_improvement',  # Added this field
            'test_pass_rate', 'total_tests', 'passed_tests',
            'failed_test_details'
        ]
        
        # Rest of the function remains the same until improvements calculation
        original_metrics = repair_result.get("original_metrics", {})
        optimized_metrics_list = repair_result.get("optimized_metrics", [])
        test_results_list = repair_result.get("test_results", [])
        
        best_metrics = optimized_metrics_list[-1] if optimized_metrics_list else None
        best_test_results = test_results_list[-1] if test_results_list else None
        
        # Extract failed test details
        failed_test_details = []
        if best_test_results:
            for test in best_test_results.get('test_results', []):
                if test.get('status') == 'failed':
                    failed_test_details.append(
                        f"Test {test['test_id']}: {test['test_case']} - {test.get('error', 'Unknown error')}"
                    )

        # Calculate improvements
        if best_metrics:
            try:
                improvements = {
                    'energy_improvement': safe_float(original_metrics.get('energy_consumed', 0)) - 
                                        safe_float(best_metrics.get('energy_consumed', 0)),
                    'power_improvement': safe_float(original_metrics.get('power_draw', 0)) - 
                                       safe_float(best_metrics.get('power_draw', 0)),
                    'time_improvement': safe_float(original_metrics.get('execution_time', 0)) - 
                                      safe_float(best_metrics.get('execution_time', 0)),
                    'cpu_improvement': safe_float(original_metrics.get('cpu_usage', 0)) - 
                                     safe_float(best_metrics.get('cpu_usage', 0)),
                    'memory_improvement': safe_float(original_metrics.get('memory_usage', 0)) - 
                                        safe_float(best_metrics.get('memory_usage', 0)),
                    'method_energy_improvement': compare_method_energy_data(
                        original_metrics.get('method_energy_data', '{}'),
                        best_metrics.get('method_energy_data', '{}')
                    ),
                    'test_pass_rate': best_test_results.get('pass_rate', 'N/A') if best_test_results else 'N/A',
                    'total_tests': best_test_results.get('total_tests', 'N/A') if best_test_results else 'N/A',
                    'passed_tests': best_test_results.get('passed_tests', 'N/A') if best_test_results else 'N/A',
                    'failed_test_details': '; '.join(failed_test_details) if failed_test_details else 'None'
                }
            except Exception as e:
                logger.error(f"Error calculating improvements: {e}")
                improvements = {key: 'N/A' for key in ['energy_improvement', 'power_improvement', 
                              'time_improvement', 'cpu_improvement', 'memory_improvement', 
                              'method_energy_improvement', 'test_pass_rate', 'total_tests', 
                              'passed_tests', 'failed_test_details']}
        else:
            improvements = {key: 'N/A' for key in ['energy_improvement', 'power_improvement', 
                          'time_improvement', 'cpu_improvement', 'memory_improvement', 
                          'method_energy_improvement', 'test_pass_rate', 'total_tests', 
                          'passed_tests', 'failed_test_details']}

        # Write to CSV
        with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = {
                'timestamp': datetime.now().isoformat(),
                'task_id': row_data.get('task_id', ''),
                'original_code': row_data.get('code', ''),
                'energy_consumed': original_metrics.get('energy_consumed', ''),
                'power_draw': original_metrics.get('power_draw', ''),
                'execution_time': original_metrics.get('execution_time', ''),
                'cpu_usage': original_metrics.get('cpu_usage', ''),
                'memory_usage': original_metrics.get('memory_usage', ''),
                'optimized_code': repair_result.get("repairs", [''])[-1],
                'optimization_explanation': repair_result.get("explanations", [''])[-1],
                'optimized_energy_consumed': best_metrics.get('energy_consumed', '') if best_metrics else '',
                'optimized_power_draw': best_metrics.get('power_draw', '') if best_metrics else '',
                'optimized_execution_time': best_metrics.get('execution_time', '') if best_metrics else '',
                'optimized_cpu_usage': best_metrics.get('cpu_usage', '') if best_metrics else '',
                'optimized_memory_usage': best_metrics.get('memory_usage', '') if best_metrics else '',
                **improvements
            }
            writer.writerow(row)

    except Exception as e:
        logger.error(f"Error in save_repair_to_csv: {str(e)}", exc_info=True)

# Add these validation functions
def is_valid_code(code: str) -> bool:
    """Validate if the code is syntactically correct Python code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def validate_code_structure(code: str) -> Tuple[bool, str]:
    """
    Validate if the code contains necessary components and proper structure.
    Returns (is_valid, error_message).
    """
    if not code or not isinstance(code, str):
        return False, "Code is empty or not a string"
        
    try:
        # First try to parse the code
        ast.parse(code)
        
        # Check for function definitions
        tree = ast.parse(code)
        has_function = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_function = True
                break
                
        if not has_function:
            return False, "No function definition found in code"
            
        # Check for placeholders
        placeholders = ['[existing code]', '[optimized code]', '<code>', '</code>']
        for placeholder in placeholders:
            if placeholder in code:
                return False, f"Code contains placeholder: {placeholder}"
        
        return True, "Code structure is valid"
        
    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"
    except Exception as e:
        return False, f"Error validating code structure: {str(e)}"

def fix_indentation(code: str) -> str:
    """Attempt to fix common indentation issues in the code."""
    try:
        # Split into lines while preserving empty lines
        lines = code.splitlines()
        fixed_lines = []
        current_indent = 0
        indent_stack = [0]
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            content = line.lstrip()
            
            # Detect if this line should increase indentation
            increase_indent = content.endswith(':')
            
            # Adjust indentation based on content
            if content.startswith(('return', 'break', 'continue', 'pass', 'else:', 'elif ', 'except:', 'finally:')):
                if indent_stack:
                    current_indent = indent_stack[-1]
            elif indent < current_indent:
                # Remove higher indentation levels
                while indent_stack and indent_stack[-1] > indent:
                    indent_stack.pop()
                current_indent = indent_stack[-1] if indent_stack else 0
            
            # Apply the current indentation
            fixed_line = ' ' * current_indent + content
            fixed_lines.append(fixed_line)
            
            # Update indentation for next line
            if increase_indent:
                next_indent = current_indent + 4
                indent_stack.append(next_indent)
                current_indent = next_indent
        
        return '\n'.join(fixed_lines)
        
    except Exception as e:
        logger.error(f"Error fixing indentation: {e}")
        return code

def get_default_arguments(func_def_node: ast.FunctionDef) -> Dict[str, Any]:
    """Generate default arguments for a function based on its parameters."""
    defaults = {}
    
    for arg in func_def_node.args.args:
        # Get argument name and its annotation if available
        arg_name = arg.arg
        arg_annotation = arg.annotation.id if hasattr(arg, 'annotation') and arg.annotation else None
        
        # Provide default values based on type hints or use general defaults
        if arg_annotation:
            if arg_annotation in ('str', 'String'):
                defaults[arg_name] = "test"
            elif arg_annotation in ('int', 'Integer'):
                defaults[arg_name] = 0
            elif arg_annotation in ('list', 'List'):
                defaults[arg_name] = []
            elif arg_annotation in ('dict', 'Dict'):
                defaults[arg_name] = {}
            elif arg_annotation in ('bool', 'Boolean'):
                defaults[arg_name] = False
            else:
                defaults[arg_name] = None
        else:
            # If no type hint, try to infer from parameter name
            if 'str' in arg_name.lower() or 'name' in arg_name.lower() or 'text' in arg_name.lower():
                defaults[arg_name] = "test"
            elif 'num' in arg_name.lower() or 'count' in arg_name.lower() or 'index' in arg_name.lower():
                defaults[arg_name] = 0
            elif 'list' in arg_name.lower() or 'array' in arg_name.lower():
                defaults[arg_name] = [1, 2, 3]
            elif 'dict' in arg_name.lower() or 'map' in arg_name.lower():
                defaults[arg_name] = {}
            elif 'matrix' in arg_name.lower() or 'grid' in arg_name.lower():
                defaults[arg_name] = [[1, 2], [3, 4]]
            elif arg_name.lower() in ('ch', 'char'):
                defaults[arg_name] = 'a'
            elif 'bool' in arg_name.lower() or 'flag' in arg_name.lower():
                defaults[arg_name] = False
            else:
                defaults[arg_name] = "dummy"
    
    return defaults

def profile_code_execution(code_snippet: str) -> Optional[Dict[str, Any]]:
    """Profile code execution with multiple error handling blocks."""
    
    # Step 0: Initial code validation
    try:
        # Basic syntax check
        if not is_valid_code(code_snippet):
            logger.error("Code contains syntax errors")
            return None
            
        # Structure validation
        is_valid, error_message = validate_code_structure(code_snippet)
        if not is_valid:
            logger.error(f"Code structure validation failed: {error_message}")
            return None
    except Exception as e:
        logger.error(f"Error in initial code validation: {e}")
        return None

    # Step 1: Parse AST and get function details
    try:
        tree = ast.parse(code_snippet)
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                func_name = node.name
                break
        
        if not func_def:
            logger.error("No function definition found")
            return None
            
        # Get default arguments for the function
        default_args = get_default_arguments(func_def)
            
    except SyntaxError as e:
        logger.error(f"SyntaxError in code snippet: {e}")
        logger.debug(f"Code snippet that failed:\n{code_snippet}")
        return None
    except Exception as e:
        logger.error(f"Error parsing AST: {e}")
        return None

    # Step 2: Transform AST with decorator
    try:
        class DecoratorAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                decorator = ast.Name(id='measure_energy', ctx=ast.Load())
                node.decorator_list.insert(0, decorator)
                logger.info(f"Added measure_energy decorator to function: {node.name}")
                return self.generic_visit(node)
        
        transformer = DecoratorAdder()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        # Step 3: Convert modified AST back to code
        try:
            modified_code = ast.unparse(modified_tree)
        except AttributeError:
            # For Python versions < 3.9
            import astor
            modified_code = astor.to_source(modified_tree)

        logger.debug(f"Modified code:\n{modified_code}")
        
    except Exception as e:
        logger.error(f"Error transforming AST or converting to code: {e}")
        return None

    # Step 4: Setup execution environment
    try:
        # Create a shared energy_meter instance
        energy_meter = RAPLEnergyMeter()
        measure_energy_decorator = energy_meter.measure_energy

        # Create a new dictionary for globals
        exec_globals = {
            '__builtins__': __builtins__,
            'measure_energy': measure_energy_decorator  # Use the shared instance's decorator
        }

        # Execute the modified code to define the function
        exec(modified_code, exec_globals)

        if func_name not in exec_globals:
            logger.error(f"Function {func_name} not defined after execution")
            return None

        # Get the function object
        func = exec_globals[func_name]

    except Exception as e:
        logger.error(f"Error setting up execution environment: {e}")
        return None

    # Step 5: Execute and collect metrics
    try:
        start_time = time.time()
        
        logger.info(f"Executing {func_name} with default arguments: {default_args}")
        func(**default_args)

        end_time = time.time()
        execution_time = end_time - start_time

        # Collect metrics
        cpu_percent, memory_percent = get_cpu_memory_usage()
        method_energy_data = energy_meter.get_method_energy_data()

        # Calculate total energy consumption
        if isinstance(method_energy_data, dict):
            energy_consumed = sum(method_energy_data.values())
        else:
            energy_consumed = 0
            logger.warning("No energy data collected")

        # Reset energy meter
        energy_meter.reset()

        # Calculate code quality metrics
        complexity_value = analyze_cyclomatic_complexity(code_snippet)
        complexity_score = calculate_complexity_score(complexity_value)
        pylint_score, maintainability_score, code_quality_score = analyze_code_quality(
            code_snippet, code_snippet
        )

        return {
            'execution_time': execution_time,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'cyclomatic_complexity': complexity_value,
            'complexity_score': complexity_score,
            'pylint_score': pylint_score,
            'maintainability_score': maintainability_score,
            'code_quality_score': code_quality_score,
            'method_energy_data': method_energy_data,
            'energy_consumed': energy_consumed,
        }

    except Exception as e:
        logger.error(f"Error in profile_code_execution: {str(e)}")
        logger.debug(f"Code snippet that failed:\n{code_snippet}")
        return None

def extract_python_code(repair_text: str) -> Optional[str]:
    """Extract and validate Python code from repair suggestion text."""
    try:
        def is_valid_python(code: str) -> bool:
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False

        # Extract code using code blocks
        code_blocks = []
        python_block_pattern = re.compile(r'```(?:python)?(.*?)```', re.DOTALL)
        matches = python_block_pattern.finditer(repair_text)
        for match in matches:
            code = match.group(1).strip()
            if code and is_valid_python(code) and 'def solution' in code:
                code_blocks.append(code)

        if code_blocks:
            # Return the first valid code block containing 'def solution'
            return code_blocks[0]

        # If no code blocks found, search for function definition
        function_pattern = re.compile(r'def\s+solution\s*\(.*?\):\s*(?:[^\n]*\n\s+.*?)+', re.DOTALL)
        match = function_pattern.search(repair_text)
        if match:
            code = match.group(0)
            if is_valid_python(code):
                return code

        logger.error("No valid code defining 'def solution' found in repair text.")
        return None

    except Exception as e:
        logger.error(f"Error extracting Python code: {e}")
        return None


def profile_with_powermeter(code: str, num_runs: int = 3) -> Optional[Dict[str, float]]:
    """Profile code with power meter multiple times and return average metrics."""
    try:
        total_metrics = {
            'energy_consumed': 0.0,
            'execution_time': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'power_draw': 0.0
        }
        
        successful_runs = 0
        energy_meter = RAPLEnergyMeter()
        
        for run in range(num_runs):
            try:
                # Execute with timeout
                with timeout(10):  # 10 second timeout
                    # Start energy measurement
                    energy_meter.start()
                    start_time = time.time()
                    
                    # Execute the code
                    exec(code, {}, {})
                    
                    # Stop energy measurement
                    energy_meter.stop()
                    end_time = time.time()
                    
                    # Get metrics
                    energy_data = energy_meter.get_trace()
                    energy_consumed = energy_data[-1] if energy_data else 0.0
                    cpu_usage, memory_usage = get_cpu_memory_usage()
                    execution_time = end_time - start_time
                    power_draw = energy_consumed / execution_time if execution_time > 0 else 0
                    
                    metrics = {
                        'energy_consumed': energy_consumed,
                        'execution_time': execution_time,
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'power_draw': power_draw
                    }
                    
                    # Update totals
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    successful_runs += 1
                    logger.debug(f"Run {run + 1} successful")
                    
            except TimeoutError:
                logger.error(f"Run {run + 1} timed out")
            except Exception as e:
                logger.error(f"Error in run {run + 1}: {e}")
                
        if successful_runs == 0:
            logger.error("No successful profiling runs")
            return None
            
        # Calculate averages
        for key in total_metrics:
            total_metrics[key] /= successful_runs
            
        logger.info(f"Average metrics over {successful_runs} runs:")
        logger.info(f"- Energy: {total_metrics['energy_consumed']:.3f}J")
        logger.info(f"- Time: {total_metrics['execution_time']:.3f}s")
        logger.info(f"- Power: {total_metrics['power_draw']:.3f}W")
        logger.info(f"- CPU: {total_metrics['cpu_usage']:.2f}%")
        logger.info(f"- Memory: {total_metrics['memory_usage']:.2f}%")
        
        return total_metrics
        
    except Exception as e:
        logger.error(f"Error in profile_with_powermeter: {e}")
        return None

def calculate_code_quality_metrics(code: str) -> Dict[str, float]:
    """Calculate code quality metrics for the given code."""
    try:
        # Calculate various code quality metrics
        metrics = {
            'pylint_score': analyze_code_quality(code),
            'complexity_score': calculate_complexity_score(code),
            'maintainability_score': analyze_maintainability(code),
            'code_quality_score': 0.0  # Will be calculated below
        }
        
        # Calculate overall code quality score
        quality_weights = {
            'pylint_score': 0.4,
            'complexity_score': 0.3,
            'maintainability_score': 0.3
        }
        
        weighted_sum = sum(metrics[key] * weight 
                         for key, weight in quality_weights.items() 
                         if key in metrics)
        
        metrics['code_quality_score'] = weighted_sum
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating code quality metrics: {e}")
        return {
            'pylint_score': 0.0,
            'complexity_score': 0.0,
            'maintainability_score': 0.0,
            'code_quality_score': 0.0
        }

def extract_repairs_and_explanations(response) -> Tuple[List[str], List[str]]:
    """Extract repairs and explanations from the API response."""
    repairs = []
    explanations = []
    
    try:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    full_response += json_line.get("response", "")
                except json.JSONDecodeError:
                    continue

        if not full_response.strip():
            logger.error("Received empty response from API")
            return [], []

        # Process the response to extract code blocks and explanations
        current_repair = ""
        current_explanation = ""
        in_repair = False
        in_explanation = False

        for line in full_response.split('\n'):
            if line.strip().startswith('```python'):
                in_repair = True
                current_repair = ""
            elif line.strip() == '```' and in_repair:
                in_repair = False
                repairs.append(current_repair.strip())
                in_explanation = True
            elif in_repair:
                current_repair += line + '\n'
            elif in_explanation and line.strip():
                current_explanation += line + '\n'
            elif in_explanation and not line.strip():
                if current_explanation.strip():
                    explanations.append(current_explanation.strip())
                current_explanation = ""
                in_explanation = False

        # Add the last explanation if it exists
        if current_explanation.strip():
            explanations.append(current_explanation.strip())

        # Ensure we have matching pairs
        while len(explanations) < len(repairs):
            explanations.append("")

        return repairs, explanations

    except Exception as e:
        logger.error(f"Error extracting repairs and explanations: {e}")
        logger.debug(f"Problematic response:\n{full_response}")
        return [], []

def pre_execution_analysis(code_snippet: str) -> Dict[str, Any]:
    """
    Perform static analysis on the code snippet to identify resource utilization patterns
    and potential performance bottlenecks.
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code_snippet)

        # Initialize analysis results
        analysis_results = {
            'loop_nesting_levels': [],
            'function_calls': [],
            'io_operations': [],
            'recursive_calls': [],
            'large_data_structures': [],
            'comprehensions': [],
            'threading_usage': False,
            'multiprocessing_usage': False,
            'external_libraries': set(),
        }

        # Define a NodeVisitor to walk the AST
        class CodeAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_loop_level = 0
                self.function_defs = set()
                self.call_stack = []

            def visit_For(self, node):
                self.current_loop_level += 1
                analysis_results['loop_nesting_levels'].append(self.current_loop_level)
                self.generic_visit(node)
                self.current_loop_level -= 1

            def visit_While(self, node):
                self.current_loop_level += 1
                analysis_results['loop_nesting_levels'].append(self.current_loop_level)
                self.generic_visit(node)
                self.current_loop_level -= 1

            def visit_FunctionDef(self, node):
                self.function_defs.add(node.name)
                self.generic_visit(node)

            def visit_Call(self, node):
                # Record function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    analysis_results['function_calls'].append(func_name)

                    # Check for recursion
                    if func_name in self.function_defs:
                        analysis_results['recursive_calls'].append(func_name)
                elif isinstance(node.func, ast.Attribute):
                    # For methods like sys.stdout.write
                    attr_names = []
                    value = node.func
                    while isinstance(value, ast.Attribute):
                        attr_names.insert(0, value.attr)
                        value = value.value
                    if isinstance(value, ast.Name):
                        attr_names.insert(0, value.id)
                    func_name = '.'.join(attr_names)
                    analysis_results['function_calls'].append(func_name)

                # Check for I/O operations
                io_functions = ['open', 'read', 'write', 'input', 'print']
                if func_name in io_functions or 'sys.stdin' in func_name or 'sys.stdout' in func_name:
                    analysis_results['io_operations'].append(func_name)

                # Check for threading or multiprocessing usage
                if func_name in ['Thread', 'Process', 'Pool']:
                    analysis_results['threading_usage'] = True

                # Record external libraries
                if '.' in func_name:
                    lib_name = func_name.split('.')[0]
                    analysis_results['external_libraries'].add(lib_name)

                self.generic_visit(node)

            def visit_ListComp(self, node):
                analysis_results['comprehensions'].append('ListComp')
                self.generic_visit(node)

            def visit_DictComp(self, node):
                analysis_results['comprehensions'].append('DictComp')
                self.generic_visit(node)

            def visit_SetComp(self, node):
                analysis_results['comprehensions'].append('SetComp')
                self.generic_visit(node)

            def visit_Assign(self, node):
                # Check for large data structures
                if isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
                    if len(getattr(node.value, 'elts', [])) > 1000:  # Arbitrary threshold
                        analysis_results['large_data_structures'].append(ast.unparse(node))
                self.generic_visit(node)

        # Visit the AST
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)

        # Post-process analysis results
        analysis_results['max_loop_nesting'] = max(analysis_results['loop_nesting_levels']) if analysis_results['loop_nesting_levels'] else 0
        analysis_results['recursive_calls'] = list(set(analysis_results['recursive_calls']))
        analysis_results['function_calls'] = list(set(analysis_results['function_calls']))
        analysis_results['io_operations'] = list(set(analysis_results['io_operations']))
        analysis_results['large_data_structures'] = list(set(analysis_results['large_data_structures']))
        analysis_results['comprehensions'] = list(set(analysis_results['comprehensions']))
        analysis_results['external_libraries'] = list(analysis_results['external_libraries'])

        return analysis_results

    except Exception as e:
        logger.error(f"Error in pre_execution_analysis: {e}")
        return {}

def generate_analysis_summary(analysis_results: Dict[str, Any]) -> str:
    """Generate a summary string of the pre-execution analysis results."""
    summary_lines = []

    if analysis_results.get('max_loop_nesting', 0) > 2:
        summary_lines.append(f"- Deeply nested loops detected (Max nesting level: {analysis_results['max_loop_nesting']}).")

    if analysis_results.get('recursive_calls'):
        recursions = ', '.join(analysis_results['recursive_calls'])
        summary_lines.append(f"- Recursive function calls identified: {recursions}.")

    if analysis_results.get('io_operations'):
        io_ops = ', '.join(analysis_results['io_operations'])
        summary_lines.append(f"- Heavy I/O operations detected: {io_ops}.")

    if analysis_results.get('large_data_structures'):
        summary_lines.append(f"- Large data structures initialized which may consume significant memory.")

    if analysis_results.get('comprehensions'):
        comps = ', '.join(analysis_results['comprehensions'])
        summary_lines.append(f"- Comprehensions used: {comps}.")

    if analysis_results.get('threading_usage'):
        summary_lines.append("- Threading usage detected.")

    if analysis_results.get('multiprocessing_usage'):
        summary_lines.append("- Multiprocessing usage detected.")

    if analysis_results.get('external_libraries'):
        libs = ', '.join(analysis_results['external_libraries'])
        summary_lines.append(f"- External libraries used: {libs}.")

    if not summary_lines:
        summary_lines.append("No significant resource utilization patterns detected.")

    return '\n'.join(summary_lines)

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    # Set the signal handler and a 5-second alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def check_and_install_dependencies(code: str) -> bool:
    """Check for required libraries and install if missing."""
    try:
        # Extract import statements
        import_pattern = re.compile(r'^import\s+(\w+)|^from\s+(\w+)\s+import', re.MULTILINE)
        matches = import_pattern.finditer(code)
        required_libs = set()
        
        for match in matches:
            lib_name = match.group(1) or match.group(2)
            if lib_name not in ('sys', 'os', 'math', 'threading', 'multiprocessing'):
                required_libs.add(lib_name)
        
        # Check and install missing libraries
        for lib in required_libs:
            try:
                importlib.import_module(lib)
                logger.info(f"Library {lib} is already installed")
            except ImportError:
                logger.info(f"Installing required library: {lib}")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                    logger.info(f"Successfully installed {lib}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {lib}: {e}")
                    return False
        return True
        
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return False

def extract_function_name(code: str) -> Optional[str]:
    """Extract the original function name from the code."""
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception as e:
        logger.error(f"Error extracting function name: {e}")
    return None

def modify_test_code(test_code, original_function_name):
    """
    Replaces all occurrences of the original function name with 'solution' in the test code.
    """
    pattern = r'\b{}\b'.format(re.escape(original_function_name))
    modified_test_code = re.sub(pattern, 'solution', test_code)
    return modified_test_code

def assemble_test_code(test_imports, modified_test_list_code, modified_test_code):
    """
    Assembles the full test code by combining imports, modified test list code, and modified test code.
    """
    imports = '\n'.join(test_imports) if test_imports else ''
    full_test_code = f"""
{imports}

{modified_test_list_code}

{modified_test_code}
"""
    return full_test_code

def compare_method_energy_data(original_data: str, optimized_data: str) -> str:
    """Compare the method energy data between original and optimized versions."""
    try:
        original = json.loads(original_data) if isinstance(original_data, str) else original_data
        optimized = json.loads(optimized_data) if isinstance(optimized_data, str) else optimized_data

        # Convert lists of tuples to dictionaries
        if isinstance(original, list):
            original = {k: v for k, v in original}
        if isinstance(optimized, list):
            optimized = {k: v for k, v in optimized}

        improvements = {}
        for method in set(original.keys()) | set(optimized.keys()):
            orig_energy = float(original.get(method, 0))
            opt_energy = float(optimized.get(method, 0))
            improvements[method] = orig_energy - opt_energy

        return json.dumps(improvements)
    except Exception as e:
        logger.error(f"Error comparing method energy data: {e}")
        return "{}"

def track_tpo_decision_process(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Track and log the TPO decision-making process."""
    logger.info("\n=== TPO Decision Process Analysis ===")
    
    strategy_scores = []
    for idx, strategy in enumerate(strategies, 1):
        # Calculate weighted score based on different factors
        weights = {
            'energy_efficiency': 0.5,
            'resource_impact': 0.3,
            'feasibility': 0.2
        }
        
        score = (
            float(strategy['overall_eval']['efficiency_improvement']) * weights['energy_efficiency'] +
            float(strategy['heuristic_eval'].get('h2_resources', 0)) * weights['resource_impact'] +
            float(strategy['heuristic_eval'].get('h3_feasibility', 0)) * weights['feasibility']
        )
        
        strategy_scores.append({
            'strategy_num': idx,
            'score': score,
            'description': strategy['description']
        })
        
        logger.info(f"\nStrategy {idx} Score Breakdown:")
        logger.info(f"- Energy Efficiency Weight: {weights['energy_efficiency']} × "
                   f"{strategy['overall_eval']['efficiency_improvement']}")
        logger.info(f"- Resource Impact Weight: {weights['resource_impact']} × "
                   f"{strategy['heuristic_eval'].get('h2_resources', 0)}")
        logger.info(f"- Feasibility Weight: {weights['feasibility']} × "
                   f"{strategy['heuristic_eval'].get('h3_feasibility', 0)}")
        logger.info(f"Total Score: {score:.2f}")
    
    # Select best strategy
    best_strategy = max(strategy_scores, key=lambda x: x['score'])
    
    logger.info("\nFinal Strategy Selection:")
    logger.info(f"Selected Strategy {best_strategy['strategy_num']} "
                f"(Score: {best_strategy['score']:.2f})")
    logger.info(f"Description: {best_strategy['description']}")
    
    return best_strategy

def main():
    try:
        logger.info("Starting energy repair process...")

        # Import energy_profile
        import energy_profile
        logger.info("Successfully imported energy_profile module")

        # Initialize the optimization history buffer
        history_buffer = OptimizationHistoryBuffer()
        logger.info("Initialized optimization history buffer")

        # Input and output CSV filenames
        input_csv = 'evalplus_profile.csv'  
        output_csv = f'repair_suggestions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        if not os.path.exists(input_csv):
            logger.error(f"{input_csv} not found. Please run energy_profile.py to generate it.")
            exit(1)

        logger.info(f"Input CSV: {input_csv}")
        logger.info(f"Output CSV: {output_csv}")

        # Read code snippets and energy profiles
        with open(input_csv, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                task_id = row.get('task_id', 'unknown')
                logger.info(f"\nProcessing task_id: {task_id}")

                try:
                    # Extract energy profile data
                    energy_profile_data = {
                        'energy_consumed': float(row.get('energy_consumed', 0)),
                        'execution_time': float(row.get('execution_time', 0)),
                        'cpu_usage': float(row.get('cpu_usage', 0)),
                        'memory_usage': float(row.get('memory_usage', 0)),
                        'power_draw': float(row.get('power_draw', 0)),
                        'method_energy_data': row.get('method_energy_data', '{}'),  # Added this line
                        'complexity_value': float(row.get('cyclomatic_complexity', 0)),
                        'complexity_score': float(row.get('complexity_score', 0)),
                        'pylint_score': float(row.get('pylint_score', 0)),
                        'maintainability_score': float(row.get('maintainability_score', 0)),
                        'code_quality_score': float(row.get('code_quality_score', 0)),
                        'combined_score': float(row.get('combined_score', 0))
                    }

                    # Extract test information
                    test_imports = row.get('test_imports', '').split('\n') if row.get('test_imports') else []
                    test_list = row.get('test_list', '').split('\n') if row.get('test_list') else []
                    test_code = row.get('test', '')

                    # Extract the original function name
                    original_code = row['code']
                    original_function_name = extract_function_name(original_code)
                    if not original_function_name:
                        logger.error("Failed to extract the original function name.")
                        continue

                    # Modify test_list and test_code
                    test_list_code = '\n'.join(test_list)
                    modified_test_list_code = modify_test_code(test_list_code, original_function_name)
                    modified_test_code = modify_test_code(test_code, original_function_name)

                    # Assemble the full test code
                    full_test_code = assemble_test_code(
                        test_imports=test_imports,
                        modified_test_list_code=modified_test_list_code,
                        modified_test_code=modified_test_code
                    )

                    # Generate repair suggestions
                    repair_result = generate_repair(
                        code_snippet=original_code,
                        energy_profile_data=energy_profile_data,
                        file_path=row.get('source_file', 'example_file.py'),
                        history_buffer=history_buffer,
                        test_code=full_test_code,
                        test_imports=[]
                    )

                    if repair_result:
                        logger.info(f"Successfully generated repairs for task_id: {task_id}")

                        # Log test results for each repair
                        repairs = repair_result.get("repairs", [])
                        test_results = repair_result.get("test_results", [])

                        for i, (repair, test_result) in enumerate(zip(repairs, test_results), 1):
                            logger.info(f"\nRepair {i} Test Results:")
                            if test_result is not None:
                                log_test_results(test_result, 'evalplus')
                            else:
                                logger.warning(f"No test results available for repair {i}")

                        save_repair_to_csv(row, repair_result, output_csv)
                    else:
                        logger.error(f"Failed to generate repair suggestions for task_id: {task_id}")

                except Exception as e:
                    logger.error(f"Error processing task_id {task_id}: {str(e)}", exc_info=True)
                    continue

        logger.info(f"\nRepair suggestions have been saved to: {output_csv}")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()