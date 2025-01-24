import unittest
from unittest.mock import patch, MagicMock
import ast
import time
from unittest.mock import patch, MagicMock, mock_open
import json
import subprocess

# Import the functions and classes from your script
from pylint.lint import Run
from radon.metrics import mi_visit

from src.energy_profile import (
    get_cpu_memory_usage,
    extract_test_cases,
    analyze_cyclomatic_complexity,
    calculate_complexity_score,
    analyze_code_quality,
    calculate_combined_score,
    reformat_code,
    PowermetricsEnergyMeter,
    read_power_data,
    run_human_eval_function,
    run_mbpp_function,
    replace_candidate_with_func,
    main,
    profiling_results,
    save_results_to_csv,
    generate_visualization
)

class TestEnergyProfile(unittest.TestCase):

    def setUp(self):
        # Clear profiling_results before each test
        global profiling_results
        profiling_results.clear()

    def test_get_cpu_memory_usage(self):
        cpu_usage, memory_usage = get_cpu_memory_usage()
        # Check that the usage percentages are between 0 and 100
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
        self.assertGreaterEqual(memory_usage, 0)
        self.assertLessEqual(memory_usage, 100)

    def test_extract_test_cases(self):
        test_str = """
def test_candidate():
    assert candidate(2) == 4
    assert candidate(3) == 9
"""
        expected = [('candidate(2)', '4'), ('candidate(3)', '9')]
        test_cases = extract_test_cases(test_str)
        # Convert AST nodes back to code strings for comparison
        extracted = []
        for left_expr, right_expr in test_cases:
            left_code = ast.unparse(left_expr)
            right_code = ast.unparse(right_expr)
            extracted.append((left_code, right_code))
        self.assertEqual(extracted, expected)

    def test_analyze_cyclomatic_complexity(self):
        func_code = """
def example_function(x):
    if x > 0:
        return x
    else:
        return -x
"""
        complexity = analyze_cyclomatic_complexity(func_code)
        self.assertEqual(complexity, 2)  # The function has a cyclomatic complexity of 2

    def test_calculate_complexity_score(self):
        score = calculate_complexity_score(5)
        self.assertEqual(score, 0.5)
        score = calculate_complexity_score(10)
        self.assertEqual(score, 0.0)
        score = calculate_complexity_score(0)
        self.assertEqual(score, 1.0)


    @patch('energy_profile.StringIO')
    @patch('energy_profile.mi_visit')
    @patch('energy_profile.Run')
    def test_analyze_code_quality(self, mock_Run, mock_mi_visit, mock_StringIO):
        # Mock mi_visit to return a maintainability index of 85.0
        mock_mi_visit.return_value = 85.0

        # Create a proper mock for StringIO and its output
        mock_output = MagicMock()
        # Mock the pylint output in the exact format it produces
        mock_output.getvalue.return_value = json.dumps([{
            "type": "report",
            "score": 80.0,  # Note: using float instead of string
            "module": "source"
        }])
        mock_StringIO.return_value = mock_output

        # Mock Run to ensure it doesn't actually run pylint
        mock_Run.return_value = None

        source_code = """
def sample_function():
    return True
"""
        test_code = "assert sample_function() == True"

        # Call the function under test
        pylint_score, maintainability_score, code_quality_score = analyze_code_quality(source_code, test_code)

        # Assertions
        self.assertEqual(pylint_score, 8.0)  # 80 / 10.0
        self.assertEqual(maintainability_score, 8.5)  # 85.0 / 10.0
        self.assertEqual(code_quality_score, (8.0 + 8.5) / 2)

    def test_calculate_combined_score(self):
        combined_score = calculate_combined_score(0.8, 8.0)
        self.assertEqual(combined_score, 0.8)  # (0.8 + (8.0/10)) / 2 = 0.8

    def test_reformat_code(self):
        code = "def example(x):return x*2"
        reformatted = reformat_code(code)
        expected = "\ndef example(x):\nreturn x*2"
        self.assertEqual(reformatted.strip(), expected.strip())

    @patch('energy_profile.subprocess.check_output')
    def test_powermetrics_energy_meter(self, mock_check_output):
        # Mock the powermetrics output
        mock_output = "CPU Energy: 50.0 Joules\n"
        mock_check_output.return_value = mock_output.encode('utf-8')

        energy_meter = PowermetricsEnergyMeter()
        energy_meter.start()
        # Simulate some time passing
        time.sleep(0.1)
        energy_meter.stop()

        energy_consumed = energy_meter.get_trace()[0]
        self.assertEqual(energy_consumed, 0.0)  # Since start and end energy are the same in mock

    @patch('energy_profile.os.geteuid')
    def test_read_power_data(self, mock_geteuid):
        # Simulate running as root
        mock_geteuid.return_value = 0
        with patch('energy_profile.subprocess.check_output') as mock_check_output:
            mock_output = "CPU Power: 1000 mW\n"
            mock_check_output.return_value = mock_output.encode('utf-8')

            power_data = read_power_data()
            self.assertIn("CPU Power: 1000.00 mW", power_data)

    def test_run_human_eval_function(self):
        func_str = """def add(x, y):
    return x + y"""
        test_cases = [(ast.parse('add(2, 3)').body[0].value, ast.parse('5').body[0].value)]
        
        test_namespace = {}
        
        def mock_exec(code, globals_dict=None, locals_dict=None):
            exec(code, test_namespace)
            
        def mock_eval(expr, globals_dict=None, locals_dict=None):
            return 5
            
        with patch('builtins.exec', side_effect=mock_exec), \
             patch('builtins.eval', side_effect=mock_eval):
            
            run_human_eval_function(func_str, test_cases)
            # Test passes if no exception is raised

    def test_run_mbpp_function(self):
        func_code = """def add(x, y):
    return x + y"""
        test_list = ["assert add(2, 3) == 5"]
        
        test_namespace = {}
        
        def mock_exec(code, globals_dict=None, locals_dict=None):
            exec(code, test_namespace)
        
        with patch('builtins.exec', side_effect=mock_exec):
            run_mbpp_function(func_code, test_list)

    def test_save_results_to_csv(self):
        global profiling_results
        test_result = {
            'task_id': 'test1',
            'execution_time': 1.0,
            'cpu_usage': 50.0,
            'memory_usage': 60.0
        }
        profiling_results.append(test_result)
        
        with patch('builtins.open', mock_open()) as mock_file:
            save_results_to_csv('test.csv')
            mock_file.assert_called_once_with('test.csv', mode='w', newline='')

    def test_generate_visualization(self):
        test_result = {
            'task_id': 'test1',
            'execution_time': 1.0,
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'cyclomatic_complexity': 2,
            'complexity_score': 0.8,
            'pylint_score': 8.0,
            'combined_score': 0.9,
            'energy_consumed': 100.0
        }
        profiling_results.append(test_result)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('os.makedirs'), \
             patch('seaborn.barplot'), \
             patch('seaborn.heatmap'), \
             patch('seaborn.scatterplot'), \
             patch('seaborn.boxplot'), \
             patch('seaborn.pairplot'):
            generate_visualization('test_dataset')

    def test_replace_candidate_with_func(self):
        node = ast.parse('candidate(5)').body[0].value
        replaced_node = replace_candidate_with_func(node, 'test_func')
        self.assertEqual(ast.unparse(replaced_node), 'test_func(5)')

    def test_read_power_data_no_root(self):
        with patch('energy_profile.os.geteuid', return_value=1000):
            result = read_power_data()
            self.assertEqual(result, "Power data not available (requires sudo privileges)")

    def test_read_power_data_permission_error(self):
        with patch('energy_profile.os.geteuid', return_value=0), \
             patch('energy_profile.subprocess.check_output', side_effect=PermissionError):
            result = read_power_data()
            self.assertEqual(result, "Power data not available (permission denied)")

    def test_main_function_humaneval(self):
        with patch('argparse.ArgumentParser.parse_args') as mock_args, \
             patch('datasets.load_dataset') as mock_dataset, \
             patch('energy_profile.PowermetricsEnergyMeter'), \
             patch('energy_profile.save_results_to_csv'), \
             patch('energy_profile.generate_visualization'):
            
            mock_args.return_value = MagicMock(dataset='humaneval')
            mock_dataset.return_value = [{'prompt': 'def test():', 
                                        'canonical_solution': 'return True',
                                        'test': 'assert test() == True',
                                        'task_id': 'TEST/0',
                                        'entry_point': 'test'}]
            
            main()

    def test_powermetrics_energy_meter_lifecycle(self):
        """Test the complete lifecycle of PowermetricsEnergyMeter."""
        with patch('energy_profile.subprocess.check_output') as mock_check_output:
            # Mock power readings
            mock_check_output.side_effect = [
                b"CPU Energy: 100.0 Joules\n",  # Start reading
                b"CPU Energy: 150.0 Joules\n"   # End reading
            ]
            
            meter = PowermetricsEnergyMeter()
            self.assertIsNone(meter.start_time)
            self.assertIsNone(meter.end_time)
            
            meter.start()
            self.assertIsNotNone(meter.start_time)
            
            meter.stop()
            self.assertIsNotNone(meter.end_time)
            
            energy_data = meter.get_trace()
            self.assertEqual(len(energy_data), 1)
            self.assertEqual(energy_data[0], 50.0)  # 150 - 100
            
            meter.reset()
            self.assertIsNone(meter.start_time)
            self.assertIsNone(meter.end_time)
            self.assertEqual(len(meter.energy_data), 0)

    def test_calculate_complexity_score_edge_cases(self):
        """Test complexity score calculation with edge cases."""
        # Test None input
        self.assertIsNone(calculate_complexity_score(None))
        
        # Test zero complexity
        self.assertEqual(calculate_complexity_score(0), 1.0)
        
        # Test maximum complexity
        self.assertEqual(calculate_complexity_score(10), 0.0)
        
        # Test beyond maximum complexity
        self.assertEqual(calculate_complexity_score(15), 0.0)
        
        # Test middle value
        self.assertEqual(calculate_complexity_score(5), 0.5)

    def test_reformat_code_comprehensive(self):
        """Test code reformatting with various cases."""
        test_cases = [
            # Simple function
            (
                "def example(x):return x*2",
                "def example(x):\nreturn x*2"
            ),
            # Multiple keywords
            (
                "def test():if True:return False",
                "def test():\nif True:\nreturn False"
            ),
            # Complex structure
            (
                "def complex():try:raise Exception()except:return False",
                "def complex():\ntry:\nraise Exception()\nexcept:\nreturn False"
            ),
            # Class definition
            (
                "class Test:def method(self):return True",
                "class Test:\ndef method(self):\nreturn True"
            )
        ]
        
        for input_code, expected_output in test_cases:
            result = reformat_code(input_code)
            # Normalize line endings and whitespace
            result = result.strip().replace('\r\n', '\n')
            expected = expected_output.strip().replace('\r\n', '\n')
            self.assertEqual(result, expected)

    def test_extract_test_cases_comprehensive(self):
        """Test extraction of test cases with various assertion patterns."""
        test_str = """
def test_function():
    assert candidate(5) == 25
    assert candidate(-2) == 4
    assert candidate(3.14) == 9.8596
    assert candidate([1,2,3]) == [1,4,9]
"""
        test_cases = extract_test_cases(test_str)
        
        # Convert AST nodes to strings for easier comparison
        extracted = []
        for left_expr, right_expr in test_cases:
            left_code = ast.unparse(left_expr).strip()
            right_code = ast.unparse(right_expr).strip()
            extracted.append((left_code, right_code))
        
        expected = [
            ('candidate(5)', '25'),
            ('candidate(-2)', '4'),
            ('candidate(3.14)', '9.8596'),
            ('candidate([1, 2, 3])', '[1, 4, 9]')
        ]
        
        # Sort both lists to ensure order doesn't matter
        self.assertEqual(sorted(extracted), sorted(expected))

    def test_calculate_combined_score_edge_cases(self):
        """Test combined score calculation with edge cases."""
        # Both scores None
        self.assertIsNone(calculate_combined_score(None, None))
        
        # One score None
        self.assertEqual(calculate_combined_score(None, 8.0), 0.8)
        self.assertEqual(calculate_combined_score(0.5, None), 0.5)
        
        # Normal case
        self.assertEqual(calculate_combined_score(0.5, 8.0), 0.65)
        
        # Perfect scores
        self.assertEqual(calculate_combined_score(1.0, 10.0), 1.0)
        
        # Minimum scores
        self.assertEqual(calculate_combined_score(0.0, 0.0), 0.0)

    @patch('energy_profile.os.makedirs')
    @patch('pandas.DataFrame')
    def test_save_results_to_csv_empty(self, mock_df, mock_makedirs):
        """Test saving results when no data is available."""
        global profiling_results
        profiling_results.clear()
        
        with patch('builtins.open', mock_open()) as mock_file:
            save_results_to_csv('test.csv')
            mock_file.assert_not_called()

    def test_analyze_cyclomatic_complexity_invalid_code(self):
        """Test cyclomatic complexity analysis with invalid code."""
        # Test with syntax error
        invalid_code = "def invalid_function(x: return x"
        result = analyze_cyclomatic_complexity(invalid_code)
        self.assertIsNone(result)
        
        # Test with empty string
        result = analyze_cyclomatic_complexity("")
        self.assertIsNone(result)
        
        # Test with non-function code
        result = analyze_cyclomatic_complexity("x = 5")
        self.assertIsNone(result)

    @patch('energy_profile.subprocess.check_output')
    @patch('energy_profile.os.geteuid')
    def test_read_power_data_parsing(self, mock_geteuid, mock_check_output):
        """Test power data parsing with various outputs."""
        # Mock root privileges
        mock_geteuid.return_value = 0
        
        # Test valid power data
        mock_check_output.return_value = b"CPU Power: 1234.56 mW\n"
        result = read_power_data()
        self.assertEqual(result, "CPU Power: 1234.56 mW")
        
        # Test invalid power data format
        mock_check_output.return_value = b"Invalid format\n"
        result = read_power_data()
        self.assertEqual(result, "Power data not available (no power data found in output)")
        
        # Test subprocess error
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'cmd')
        result = read_power_data()
        self.assertEqual(result, "Power data not available (powermetrics error)")

    # Add more test methods as needed

if __name__ == '__main__':
    unittest.main()
