import unittest
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
import src.energy_repair as energy_repair
import logging

class TestEnergyRepair(unittest.TestCase):
    def setUp(self):
        # Set up common variables and mocks
        self.logger = logging.getLogger('energy_repair')
        self.logger.disabled = True  # Disable logging during tests

    def test_safe_float(self):
        # Test cases for safe_float function
        self.assertEqual(energy_repair.safe_float('3.14'), 3.14)
        self.assertEqual(energy_repair.safe_float('42'), 42.0)
        self.assertEqual(energy_repair.safe_float('abc'), 0.0)
        self.assertEqual(energy_repair.safe_float(''), 0.0)
        self.assertEqual(energy_repair.safe_float(None), 0.0)
        self.assertEqual(energy_repair.safe_float(2.71), 2.71)
        self.assertEqual(energy_repair.safe_float(10), 10.0)
        self.assertEqual(energy_repair.safe_float(1+2j), 0.0)
        self.assertEqual(energy_repair.safe_float(True), 1.0)
        self.assertEqual(energy_repair.safe_float(False), 0.0)

    def test_optimization_history_buffer(self):
        buffer = energy_repair.OptimizationHistoryBuffer()

        # Add an optimization
        buffer.add_optimization(
            original_code='print("Hello, World!")',
            optimized_code='print("Hi!")',
            original_metrics={'energy_consumed': 10.0},
            optimized_metrics={'energy_consumed': 8.0},
            explanation='Reduced string length to save energy.',
            evaluation_score=0.85,
            test_results={
                'pass_rate': 100,
                'passed_tests': 5,
                'total_tests': 5,
                'test_results': []
            }
        )

        # Get the summary
        summary = buffer.get_history_summary()

        # Assertions
        self.assertIn('* Energy: 2.000J', summary)
        self.assertIn('Reduced string length to save energy', summary)
        self.assertIn('Score: 0.85', summary)
        self.assertIn('Pass Rate: 100%', summary)

    def test_calculate_combined_score(self):
        # Test normal values
        self.assertAlmostEqual(energy_repair.calculate_combined_score(7.0, 8.0), 7.6)
        
        # Test minimum values
        self.assertAlmostEqual(energy_repair.calculate_combined_score(0, 0), 0.0)
        
        # Test maximum values
        self.assertAlmostEqual(energy_repair.calculate_combined_score(10, 10), 10.0)
        
        # Test out-of-range values
        self.assertAlmostEqual(energy_repair.calculate_combined_score(-1, 11), 6.0)
        
        # Test with non-numeric inputs (should raise TypeError)
        with self.assertRaises(TypeError):
            energy_repair.calculate_combined_score('a', None)

    @patch('energy_repair.requests.post')
    @patch('energy_repair.profile_code_execution')
    @patch('energy_repair.log_tot_analysis')
    @patch('energy_repair.log_heuristic_evaluation')
    def test_generate_repair(self, mock_heuristic, mock_tot, mock_profile, mock_requests_post):
        # Set up mock for requests.post - match format from test_main_execution_block
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "def solution(): pass"  # Match format from test_main_execution_block
        }
        mock_requests_post.return_value = mock_response
        
        # Set up mock for profile_code_execution - match metrics from test_main_execution_block
        mock_profile.return_value = {
            'energy_consumed': 8.0,
            'execution_time': 1.5,
            'cpu_usage': 45.0,
            'memory_usage': 25.0,
            'power_draw': 4.0
        }

        # Test data - match format from test_main_execution_block
        code_snippet = 'def solution(): pass'
        energy_profile_data = {
            'energy_consumed': 10.0,
            'execution_time': 2.0,
            'cpu_usage': 50.0,
            'memory_usage': 30.0,
            'power_draw': 5.0
        }
        history_buffer = energy_repair.OptimizationHistoryBuffer()

        # Call function
        result = energy_repair.generate_repair(
            code_snippet,
            energy_profile_data,
            'test.py',
            history_buffer
        )

        # Basic assertions - match expectations from test_main_execution_block
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('repairs', result)
        self.assertIn('optimized_metrics', result)

    def test_extract_code_from_repair(self):
        # Test with correct markers
        repair_text = '```python\n<<<<< SEARCH\n[old code]\n=======\n[optimized code]\n>>>>>> REPLACE\n```'
        extracted_code = energy_repair.extract_code_from_repair(repair_text)
        self.assertEqual(extracted_code, '[optimized code]')

        # Test without markers
        repair_text = 'Some text without markers.'
        extracted_code = energy_repair.extract_code_from_repair(repair_text)
        self.assertIsNone(extracted_code)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_save_repair_to_csv(self, mock_exists, mock_file):
        mock_exists.return_value = False

        # Prepare test data
        row_data = {
            'task_id': 'task123',
            'code': 'print("Hello")',
            'energy_consumed': '10.0',
            'power_draw': '5.0',
            'execution_time': '2.0',
            'cpu_usage': '50.0',
            'memory_usage': '30.0'
        }
        repair_result = {
            'repairs': ['print("Hi")'],
            'explanations': ['Shortened the greeting.'],
            'optimized_metrics': [{
                'energy_consumed': 8.0,
                'power_draw': 4.0,
                'execution_time': 1.5,
                'cpu_usage': 45.0,
                'memory_usage': 25.0
            }],
            'original_metrics': {
                'energy_consumed': 10.0,
                'power_draw': 5.0,
                'execution_time': 2.0,
                'cpu_usage': 50.0,
                'memory_usage': 30.0
            }
        }
        output_file = 'output.csv'

        # Call the function
        energy_repair.save_repair_to_csv(row_data, repair_result, output_file)

        # Check that file was opened
        mock_file.assert_called_with(output_file, mode='a', newline='', encoding='utf-8')

        # Check that writeheader and writerow were called
        handle = mock_file()
        self.assertTrue(handle.write.called)

    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    @patch('energy_repair.PowermetricsEnergyMeter')
    @patch('energy_repair.get_cpu_memory_usage')
    @patch('energy_repair.analyze_cyclomatic_complexity')
    @patch('energy_repair.calculate_complexity_score')
    @patch('energy_repair.analyze_code_quality')
    def test_profile_code_execution(
        self, mock_code_quality, mock_complexity_score, mock_cyclomatic_complexity,
        mock_cpu_memory, mock_energy_meter, mock_unlink, mock_tempfile
    ):
        # Set up mocks
        mock_temp_file = MagicMock()
        mock_temp_file.name = 'tempfile.py'
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file
        mock_energy_meter_instance = MagicMock()
        mock_energy_meter.return_value = mock_energy_meter_instance
        mock_energy_meter_instance.get_trace.return_value = [0.0, 10.0]
        mock_cpu_memory.return_value = (50.0, 30.0)
        mock_cyclomatic_complexity.return_value = 5
        mock_complexity_score.return_value = 7.0
        mock_code_quality.return_value = (8.0, 7.5, 7.8)

        # Prepare test data
        code_snippet = 'def example_function(): pass'

        # Call the function
        result = energy_repair.profile_code_execution(code_snippet)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['energy_consumed'], 10.0)
        self.assertEqual(result['cpu_usage'], 50.0)
        self.assertEqual(result['memory_usage'], 30.0)
        self.assertEqual(result['cyclomatic_complexity'], 5)
        self.assertEqual(result['complexity_score'], 7.0)
        self.assertEqual(result['pylint_score'], 8.0)
        self.assertEqual(result['maintainability_score'], 7.5)
        self.assertEqual(result['code_quality_score'], 7.8)

    def test_extract_python_code(self):
        # Test with direct code - using exact format from other tests
        repair_text = 'def solution(): pass'  # Simplified to match format in test_main_execution_block
        extracted_code = energy_repair.extract_python_code(repair_text)
        self.assertIsNotNone(extracted_code)
        self.assertEqual(extracted_code, 'def solution(): pass')

        # Test with empty input
        repair_text = ''
        extracted_code = energy_repair.extract_python_code(repair_text)
        self.assertIsNone(extracted_code)

        # Test with None input
        extracted_code = energy_repair.extract_python_code(None)
        self.assertIsNone(extracted_code)

        # Test with code block format
        repair_text = '```python\ndef solution(): pass\n```'
        extracted_code = energy_repair.extract_python_code(repair_text)
        self.assertIsNone(extracted_code)

    @patch('energy_repair.PowermetricsEnergyMeter')
    @patch('energy_repair.get_cpu_memory_usage')
    def test_profile_with_powermeter(self, mock_cpu_memory, mock_energy_meter):
        # Set up mocks
        mock_energy_meter_instance = MagicMock()
        mock_energy_meter.return_value = mock_energy_meter_instance
        mock_energy_meter_instance.get_trace.return_value = [0.0, 10.0]
        mock_cpu_memory.return_value = (50.0, 30.0)

        # Prepare test data
        code_snippet = 'def example_function(): pass'

        # Call the function
        result = energy_repair.profile_with_powermeter(code_snippet, num_runs=3)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result['energy_consumed'], 10.0)
        self.assertEqual(result['cpu_usage'], 50.0)
        self.assertEqual(result['memory_usage'], 30.0)

    def test_log_functions(self):
        # Test that logging functions execute without error
        energy_profile_data = {
            'energy_consumed': 10.0,
            'execution_time': 2.0,
            'cpu_usage': 50.0,
            'memory_usage': 30.0
        }
        original_metrics = {
            'complexity_score': 7.0,
            'maintainability_score': 7.5,
            'code_quality_score': 7.8,
            'pylint_score': 8.0
        }

        try:
            energy_repair.log_tot_analysis(energy_profile_data, original_metrics)
            energy_repair.log_heuristic_evaluation(energy_profile_data, original_metrics)
        except Exception as e:
            self.fail(f"log functions raised an exception: {e}")

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('energy_repair.generate_repair')
    @patch('csv.DictReader')
    def test_main_execution_block(self, mock_csv_reader, mock_generate_repair, mock_exists, mock_file):
        # Mock CSV reading
        mock_exists.return_value = True
        mock_csv_reader.return_value = iter([{
            'task_id': 'task123',
            'code': 'def solution(): pass',
            'energy_consumed': '10.0',
            'power_draw': '5.0',
            'execution_time': '2.0',
            'cpu_usage': '50.0',
            'memory_usage': '30.0'
        }])

        # Mock generate_repair function
        mock_generate_repair.return_value = {
            'repairs': ['print("Hi")'],
            'explanations': ['Shortened the greeting.'],
            'optimized_metrics': [{
                'energy_consumed': 8.0,
                'power_draw': 4.0,
                'execution_time': 1.5,
                'cpu_usage': 45.0,
                'memory_usage': 25.0
            }],
            'original_metrics': {
                'energy_consumed': 10.0,
                'power_draw': 5.0,
                'execution_time': 2.0,
                'cpu_usage': 50.0,
                'memory_usage': 30.0
            }
        }

        # Run the main function
        with patch('sys.exit') as mock_exit:
            energy_repair.main()
            # Assertions
            self.assertTrue(mock_file.called)
            self.assertTrue(mock_generate_repair.called)
            self.assertFalse(mock_exit.called)

if __name__ == '__main__':
    unittest.main()