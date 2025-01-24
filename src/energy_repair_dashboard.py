import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from scipy.stats import f_oneway, wilcoxon, bootstrap, kruskal
import numpy as np
import re
import os
from dash import dash_table
from difflib import SequenceMatcher
import ast
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

# Add external scripts for code highlighting
external_scripts = [
    {'src': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/highlight.min.js'},
    {'src': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/styles/default.min.css'},
    {'src': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/styles/github.min.css'}
]
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.BOOTSTRAP], 
          external_scripts=external_scripts,
          suppress_callback_exceptions=True)

# Define dataset file paths more dynamically
def get_dataset_paths():
    """Get dataset paths dynamically based on common patterns."""
    # Base paths to check
    possible_base_paths = [
        '/data/',
        './data/',
        '../data/'
    ]
    
    # Dataset patterns to look for
    dataset_patterns = [
        'eval_plus_powermetrics_llama3_1_exp_1_with_explanations.csv',
        'eval_plus_rapl_llama3_1_70b_exp_1_with_explanations.csv',
        'eval_plus_rapl_llama3_1_70b_exp_2_with_explanations.csv',
        'eval_plus_rapl_llama3_1_8b_exp_1_with_explanations.csv',
        'eval_plus_rapl_llama3_1_8b_exp_2_with_explanations.csv'
    ]
    
    found_datasets = []
    
    # Try each base path
    for base_path in possible_base_paths:
        for pattern in dataset_patterns:
            full_path = os.path.join(base_path, pattern)
            if os.path.isfile(full_path):
                found_datasets.append(full_path)
                break  # Found this dataset, move to next pattern
    
    if not found_datasets:
        print("Warning: No dataset files found in any of the expected locations.")
        print(f"Searched in: {possible_base_paths}")
    
    return found_datasets

# Replace the hardcoded dataset_files with dynamic path detection
dataset_files = get_dataset_paths()

# Verify that all dataset files exist
for file in dataset_files:
    if not os.path.isfile(file):
        print(f"Warning: Dataset file {file} does not exist.")

# Load each dataset
df_list = []
for file in dataset_files:
    if os.path.isfile(file):
        df_temp = pd.read_csv(file)
        # Add columns indicating the dataset, source, and model
        dataset_name = file.split('/')[-1].replace('.csv', '')
        df_temp['dataset'] = dataset_name
        
        # Parse dataset_name to get source and model
        match = re.match(r'.*(rapl|powermetrics).*_(llama3_1)_(\d+)b?_exp_(\d+)', dataset_name)
        if match:
            source, model, size, exp = match.groups()
            df_temp['source'] = source
            df_temp['model'] = f"{model}:{size}b"
            
            # Add new user-friendly dataset name
            if source == 'rapl':
                df_temp['dataset_name'] = f"RAPL - LLaMA {size}B (Experiment {exp})"
            else:
                df_temp['dataset_name'] = f"Powermetrics - LLaMA {size}B (Experiment {exp})"
        else:
            df_temp['source'] = 'powermetrics'
            df_temp['model'] = 'llama3.1:8b'
            df_temp['dataset_name'] = 'Powermetrics - LLaMA 8B'
            
        df_list.append(df_temp)
    else:
        print(f"Skipping {file} as it does not exist.")

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)

# Data preprocessing
df['core_efficiency'] = (df['energy_improvement'] + df['time_improvement'] + df['cpu_improvement']) / 3
df['optimization_success'] = pd.qcut(df['core_efficiency'], q=4, labels=['Low', 'Moderate', 'Good', 'Excellent'])

# Add normalized versions of improvement metrics
for metric in ['energy_improvement', 'time_improvement', 'cpu_improvement']:
    df[f'{metric}_normalized'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Filter data for RAPL datasets only
df_rapl = df[df['source'] == 'rapl']

# Add these helper functions at the top level, before any other functions
def clean_code_str(code_str):
    """Clean code string by removing special characters and normalizing whitespace."""
    if pd.isna(code_str):
        return ""
    code_str = str(code_str).strip()
    if code_str.startswith('# '):
        code_str = code_str[2:]
    return code_str.replace('\r\n', '\n')

def get_method_ast(code_str):
    """Extract method AST from code."""
    try:
        code_str = clean_code_str(code_str)
        if not code_str:
            return None
            
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node
        return None
    except Exception:
        return None

def normalize_ast(node):
    """Normalize AST by removing docstrings, comments, and normalizing strings."""
    if isinstance(node, ast.FunctionDef):
        # Remove docstring if present
        body = node.body
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Str)):
            body = body[1:]
        
        # Recursively normalize the body
        normalized_body = []
        for n in body:
            if isinstance(n, ast.Return):
                normalized_body.append(ast.Return(value=normalize_ast(n.value)))
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Str):
                # Skip string expressions (comments/docstrings)
                continue
            elif isinstance(n, ast.Assert):
                # Normalize assert statements
                normalized_body.append(ast.Assert(
                    test=normalize_ast(n.test),
                    msg=normalize_ast(n.msg) if n.msg else None
                ))
            else:
                normalized_body.append(normalize_ast(n))
        
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=normalized_body,
            decorator_list=[],
            returns=None
        )
    elif isinstance(node, ast.Str):
        # Normalize string literals by converting to lowercase
        return ast.Str(s=node.s.lower())
    elif isinstance(node, ast.Call):
        return ast.Call(
            func=normalize_ast(node.func),
            args=[normalize_ast(arg) for arg in node.args],
            keywords=[]
        )
    elif isinstance(node, ast.Compare):
        # Normalize comparisons
        return ast.Compare(
            left=normalize_ast(node.left),
            ops=node.ops,
            comparators=[normalize_ast(comp) for comp in node.comparators]
        )
    elif isinstance(node, (ast.Name, ast.Num)):
        return node
    elif isinstance(node, list):
        return [normalize_ast(n) for n in node]
    return node

def has_different_implementation(orig_code, opt_code):
    """Check if original and optimized code have same method name but different implementations."""
    orig_ast = get_method_ast(orig_code)
    opt_ast = get_method_ast(opt_code)
    
    if orig_ast is None or opt_ast is None:
        return False
        
    # Check if method names match
    if orig_ast.name != opt_ast.name:
        return False
        
    # Normalize both ASTs
    orig_ast_normalized = normalize_ast(orig_ast)
    opt_ast_normalized = normalize_ast(opt_ast)
    
    # Compare the normalized ASTs
    return ast.dump(orig_ast_normalized) != ast.dump(opt_ast_normalized)

# Define visualizations as separate functions
def core_performance_dashboard(data):
    # Create separate dataframes for each source
    rapl_data = data[data['source'] == 'rapl'].copy()
    powermetrics_data = data[data['source'] == 'powermetrics'].copy()
    
    # Normalize values within each source
    for df in [rapl_data, powermetrics_data]:
        if not df.empty:
            # Min-max normalization for each source
            df['energy_consumed_normalized'] = (df['energy_consumed'] - df['energy_consumed'].min()) / (df['energy_consumed'].max() - df['energy_consumed'].min())
            df['optimized_energy_consumed_normalized'] = (df['optimized_energy_consumed'] - df['optimized_energy_consumed'].min()) / (df['optimized_energy_consumed'].max() - df['optimized_energy_consumed'].min())
    
    # Combine normalized data
    normalized_data = pd.concat([rapl_data, powermetrics_data])
    
    # Create scatter plot with normalized values
    fig = px.scatter(
        normalized_data,
        x='energy_consumed_normalized',
        y='optimized_energy_consumed_normalized',
        color='source',
        symbol='model',
        title='Normalized Energy Optimization Impact by Source and Model',
        labels={
            'energy_consumed_normalized': 'Original Energy Consumed (Normalized)', 
            'optimized_energy_consumed_normalized': 'Optimized Energy Consumed (Normalized)'
        },
        hover_data=['dataset']
    )
    
    # Add diagonal reference line
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color='Red', dash='dash')
    )
    
    fig.update_layout(
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def optimization_trends(data):
    # Ensure data is sorted by 'task_id'
    data = data.sort_values('task_id')

    # Compute rolling mean and standard error for each source (Darwin or Linux)
    window_size = 10
    data_grouped = data.groupby('source')

    fig_trends = make_subplots(rows=1, cols=1)

    for name, group in data_grouped:
        group = group.sort_values('task_id')
        group['rolling_energy'] = group['energy_improvement'].rolling(window_size).mean()
        group['rolling_energy_se'] = group['energy_improvement'].rolling(window_size).sem()

        # Energy Improvement Trend per source
        fig_trends.add_trace(go.Scatter(
            x=group['task_id'], y=group['rolling_energy'],
            mode='lines', name=f'Energy Improvement ({name})',
            error_y=dict(
                type='data',
                array=group['rolling_energy_se'],
                visible=True
            )
        ))

    fig_trends.update_layout(
        title='Energy Improvement Trends with Error Bars by Source',
        xaxis_title='Task ID', yaxis_title='Improvement (%)',
        height=500
    )
    return fig_trends

def get_top_examples_per_dataset(data, metric, top_n=25):
    """Get exactly top N examples per dataset based on a specified metric."""
    top_examples_list = []
    grouped = data.groupby(['dataset', 'model'])
    
    for (dataset_name, model_name), group in grouped:
        # Filter to keep only examples with matching method names but different implementations
        filtered_group = group[group.apply(
            lambda row: has_different_implementation(row['original_code'], row['optimized_code']), 
            axis=1
        )]
        
        # Sort by metric in descending order and get top N
        sorted_group = filtered_group.nlargest(top_n, metric)
        if len(sorted_group) > 0:
            top_examples_list.append(sorted_group)
    
    return pd.concat(top_examples_list, ignore_index=True) if top_examples_list else pd.DataFrame()

def format_python_code(code_str):
    """Format Python code with exact spacing and alignment."""
    import tokenize
    from io import StringIO
    
    def get_token_style(token):
        """Get token style based on type and content."""
        if token.type == tokenize.NAME:
            if token.string in ['def', 'return']:
                return {'color': '#0033B3', 'fontWeight': 'bold'}  # Blue, bold
            elif token.string in ['set', 'tuple']:
                return {'color': '#0033B3'}  # Blue
            return {'color': '#000000'}  # Black
        elif token.type == tokenize.STRING:
            if token.string.startswith('"""'):
                return {'color': '#067D17', 'fontStyle': 'italic'}  # Docstring green
            return {'color': '#067D17'}  # String green
        elif token.type == tokenize.COMMENT:
            return {'color': '#8C8C8C', 'fontStyle': 'italic'}  # Gray, italic
        return {'color': '#000000'}  # Default black

    try:
        tokens = []
        current_line = []
        current_position = 0
        
        for token in tokenize.generate_tokens(StringIO(code_str).readline):
            # Handle line breaks
            if token.type in [tokenize.NEWLINE, tokenize.NL]:
                if current_line:
                    tokens.extend(current_line)
                tokens.append(html.Br())
                current_line = []
                current_position = 0
                continue
            
            # Add spaces before token
            spaces_before = token.start[1] - current_position
            if spaces_before > 0:
                current_line.append(html.Span('\u00A0' * spaces_before))
            
            # Add the token with its style
            style = get_token_style(token)
            current_line.append(html.Span(token.string, style=style))
            
            current_position = token.end[1]
        
        # Add any remaining tokens
        if current_line:
            tokens.extend(current_line)
        
        return html.Div(tokens, style={
            'fontFamily': 'Consolas, Monaco, "Courier New", monospace',
            'fontSize': '14px',
            'lineHeight': '1.6',
            'whiteSpace': 'pre',
            'padding': '15px',
            'backgroundColor': '#FFFFFF',
            'border': '1px solid #E8E8E8',
            'borderRadius': '4px',
            'overflowX': 'auto'
        })
    except Exception as e:
        # Fallback for malformed code
        try:
            lines = code_str.splitlines()
            formatted_lines = []
            for line in lines:
                chars = []
                for char in line:
                    if char == ' ':
                        chars.append(html.Span('\u00A0'))
                    else:
                        chars.append(html.Span(char))
                formatted_lines.append(html.Div(chars))
                formatted_lines.append(html.Br())
            
            return html.Div(formatted_lines, style={
                'fontFamily': 'Consolas, Monaco, "Courier New", monospace',
                'fontSize': '14px',
                'lineHeight': '1.6',
                'whiteSpace': 'pre',
                'padding': '15px',
                'backgroundColor': '#FFFFFF',
                'border': '1px solid #E8E8E8',
                'borderRadius': '4px',
                'overflowX': 'auto'
            })
        except:
            return html.Div(str(e), style={'color': 'red'})

def format_code(code, improvement_value):
    """Format code with IDE-like syntax highlighting and improvement-based container styling."""
    if pd.isna(code):
        return html.Div("# No code available")
    
    # Clean up code string
    code = str(code).strip()
    if code.startswith('# '):
        code = code[2:]
        
    return html.Div([
        format_python_code(code)
    ], style={
        'marginBottom': '15px',
        'backgroundColor': get_highlight_color(improvement_value),
        'padding': '10px',
        'borderRadius': '8px',
        'borderLeft': f'5px solid {get_border_color(improvement_value)}',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })

def get_highlight_color(value):
    """Get container background color based on improvement value."""
    if value > 75:
        return 'rgba(46, 125, 50, 0.05)'  # Very light green
    elif value > 50:
        return 'rgba(76, 175, 80, 0.03)'  # Lighter green
    elif value > 25:
        return 'rgba(129, 199, 132, 0.02)'  # Very light green
    return 'transparent'

def get_border_color(value):
    """Get border color based on improvement value."""
    if value > 75:
        return '#4CAF50'  # Strong green
    elif value > 50:
        return '#81C784'  # Medium green
    elif value > 25:
        return '#A5D6A7'  # Light green
    return '#E8F5E9'  # Very light green

def create_dataset_selector():
    """Create dropdowns to select specific datasets and filter by improvement."""
    return html.Div([
        html.Label('Select Dataset to View Examples:'),
        dcc.Dropdown(
            id='example-dataset-dropdown',
            options=[
                {'label': 'All RAPL (Linux) Datasets', 'value': 'rapl'},
                {'label': 'All Powermetrics (macOS) Datasets', 'value': 'powermetrics'},
            ],
            value='all',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
        # Add improvement type selector
        html.Label('Select Improvement Metric:'),
        dcc.Dropdown(
            id='improvement-type-dropdown',
            options=[
                {'label': 'Energy Improvement', 'value': 'energy_improvement'},
                {'label': 'CPU Usage Improvement', 'value': 'cpu_improvement'},
                {'label': 'Execution Time Improvement', 'value': 'time_improvement'}
            ],
            value='energy_improvement',
            style={'width': '100%', 'marginBottom': '20px'}
        ),
        # Add improvement range slider
        html.Label('Filter by Improvement Range):'),
        dcc.RangeSlider(
            id='improvement-range-slider',
            min=-100,
            max=100,
            step=5,
            marks={
                i: {'label': f'{i}%', 
                    'style': {'transform': 'rotate(45deg)', 
                             'transform-origin': 'top right'}}
                for i in range(-30, 30, 20)
            },
            value=[0, 100],
            allowCross=False,
            tooltip={
                'placement': 'bottom',
                'always_visible': True,
                'template': '{value}%'
            },
            className='range-slider'
        )
    ], style={'width': '50%', 'margin': '20px auto'})

def filter_dataset_by_selection(data, dataset_selection, improvement_range, improvement_type='energy_improvement'):
    """Filter dataset based on selection and improvement range."""
    # Default to energy_improvement if improvement_type is None
    if improvement_type is None:
        improvement_type = 'energy_improvement'
    
    # First filter by dataset selection
    if dataset_selection == 'all':
        filtered_data = data  # Use all data
    if dataset_selection == 'rapl':
        filtered_data = data[data['source'] == 'rapl']
    elif dataset_selection == 'powermetrics':
        filtered_data = data[data['source'] == 'powermetrics']

    # Verify improvement_range is valid
    if improvement_range is None or len(improvement_range) != 2:
        improvement_range = [0, 100]

    # Verify the column exists
    if improvement_type not in filtered_data.columns:
        print(f"Warning: {improvement_type} not found in columns. Available columns: {filtered_data.columns.tolist()}")
        return filtered_data

    # Then filter by improvement range for the selected improvement type
    filtered_data = filtered_data[
        (filtered_data[improvement_type] >= improvement_range[0]) &
        (filtered_data[improvement_type] <= improvement_range[1])
    ]
    
    # Sort by the selected improvement type in descending order
    return filtered_data.sort_values(improvement_type, ascending=False)

def top_examples_layout(data):
    """Create tabs showing top examples for each metric."""
    return html.Div([
        create_dataset_selector(),
        html.Div(id='top-examples-content')
    ])

def parse_markdown_explanation(markdown_text):
    """Parse the markdown explanation into structured HTML with better code block and section handling."""
    if not markdown_text or pd.isna(markdown_text):
        return html.Div("No explanation available.")
    
    def parse_title(line):
        """Parse titles marked with ###."""
        if line.startswith('###'):
            return html.H3(line.replace('#', '').strip(), style={
                'color': '#2c3e50',
                'marginTop': '25px',
                'marginBottom': '15px',
                'fontSize': '20px',
                'fontWeight': 'bold',
                'borderBottom': '1px solid #eee',
                'paddingBottom': '8px'
            })
        return None
    
    def parse_bold_text(text):
        """Parse bold text marked with **."""
        parts = re.split(r'(\*\*.*?\*\*)', text)
        formatted_parts = []
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                formatted_parts.append(html.Strong(part[2:-2]))
            else:
                formatted_parts.append(part)
        return formatted_parts
    
    def parse_list_item(line):
        """Parse list items and their formatting."""
        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):  # Numbered list
            content = line[line.find('.') + 1:].strip()
            formatted_content = parse_bold_text(content)
            return html.Li(formatted_content, style={
                'marginBottom': '12px',
                'lineHeight': '1.6'
            })
        return None

    # Split into sections by ###
    sections = re.split(r'\n(?=###)', markdown_text)
    content = []
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.split('\n')
        section_content = []
        
        # Parse title
        if lines[0].startswith('###'):
            section_content.append(parse_title(lines[0]))
            lines = lines[1:]
        
        # Process remaining lines
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Handle numbered list items
            list_item = parse_list_item(line)
            if list_item:
                list_items = [list_item]
                peek = i + 1
                while peek < len(lines):
                    next_item = parse_list_item(lines[peek].strip())
                    if next_item:
                        list_items.append(next_item)
                        peek += 1
                    else:
                        break
                section_content.append(html.Ol(list_items, style={
                    'marginLeft': '20px',
                    'marginTop': '10px',
                    'marginBottom': '15px'
                }))
                i = peek
                continue
            
            # Handle regular text with potential bold parts
            formatted_text = parse_bold_text(line)
            section_content.append(html.P(formatted_text, style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }))
            i += 1
        
        # Add section content to the final content
        content.append(html.Div(section_content, style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '8px',
            'border': '1px solid #dee2e6',
            'marginTop': '15px',
            'fontSize': '14px',
            'lineHeight': '1.6',
            'fontFamily': 'Arial, sans-serif'
        }))
    
    return html.Div(content, style={
        'backgroundColor': '#f8f9fa',
        'padding': '20px',
        'borderRadius': '8px',
        'border': '1px solid #dee2e6',
        'marginTop': '15px',
        'fontSize': '14px',
        'lineHeight': '1.6',
        'fontFamily': 'Arial, sans-serif'
    })

def create_examples_content(data, dataset_selection, improvement_range, improvement_type='energy_improvement'):
    """Create the actual content showing examples."""
    # First filter out examples that don't have actual implementation differences
    filtered_data = data[data.apply(
        lambda row: has_different_implementation(row['original_code'], row['optimized_code']), 
        axis=1
    )]
    
    # If no examples remain after filtering, show a message
    if len(filtered_data) == 0:
        return html.Div([
            html.H3("No Examples Found", 
                   style={'textAlign': 'center', 'color': '#666'}),
            html.P("No code examples with meaningful implementation differences were found for the selected criteria.",
                  style={'textAlign': 'center', 'color': '#666'})
        ])
    
    # Continue with regular filtering
    filtered_data = filter_dataset_by_selection(filtered_data, dataset_selection, improvement_range, improvement_type)
    
    # Get friendly name for improvement type
    improvement_name = {
        'energy_improvement': 'Energy',
        'cpu_improvement': 'CPU Usage',
        'time_improvement': 'Execution Time'
    }.get(improvement_type, 'Energy')
    
    # Create summary section
    summary = html.Div([
        html.H3(f"Dataset: {dataset_selection}", 
               style={'fontSize': '16px', 'fontWeight': 'normal', 'marginBottom': '10px'}),
        html.Div([
            html.P(f"Total examples with implementation changes: {len(filtered_data)}", 
                  style={'margin': '0', 'padding': '2px 0'}),
            html.P(f"Average {improvement_name} improvement: {filtered_data[improvement_type].mean():.2f}%",
                  style={'margin': '0', 'padding': '2px 0', 'color': '#28a745'}),
            html.P(f"Maximum {improvement_name} improvement: {filtered_data[improvement_type].max():.2f}%",
                  style={'margin': '0', 'padding': '2px 0', 'color': '#28a745'})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'borderRadius': '4px',
            'marginBottom': '20px'
        })
    ])
    
    # Create table with examples that have actual implementation differences
    table_header = html.Thead(html.Tr([
        html.Th('Model',
               style={'backgroundColor': '#2d3436', 'color': 'white', 'padding': '10px', 'textAlign': 'left'}),
        html.Th('Improvement\n(%)',
               style={'backgroundColor': '#2d3436', 'color': 'white', 'padding': '10px', 'textAlign': 'left',
                     'whiteSpace': 'pre-line'}),
        html.Th('Code Comparison',
               style={'backgroundColor': '#2d3436', 'color': 'white', 'padding': '10px', 'textAlign': 'left'})
    ]))
    
    table_rows = []
    for _, row in filtered_data.iterrows():
        improvement_value = row[improvement_type]
        
        # Create a unique ID for this example's explanation
        example_id = f"explanation-{hash(str(row['original_code']) + str(row['optimized_code']))}"
        
        table_rows.append(html.Tr([
            html.Td(row['model']),
            html.Td(
                f"{improvement_value:.2f}%",
                style={'verticalAlign': 'top'}
            ),
            html.Td([
                html.Div([
                    html.P("Original Code:", 
                          style={'margin': '0 0 5px 0', 'color': '#666'}),
                    format_code(row['original_code'], improvement_value)
                ]),
                html.Div([
                    html.P("Optimized Code:", 
                          style={'margin': '15px 0 5px 0', 'color': '#666'}),
                    format_code(row['optimized_code'], improvement_value)
                ]),
                # Add Learn More button and hidden explanation
                html.Div([
                    html.Button(
                        "Learn More",
                        id={'type': 'learn-more-button', 'index': example_id},
                        style={
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '10px 15px',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'marginTop': '10px'
                        }
                    ),
                    html.Div(
                        parse_markdown_explanation(row.get('gpt4_explanation', '')),
                        id={'type': 'explanation-text', 'index': example_id},
                        style={
                            'display': 'none',
                            'marginTop': '10px'
                        }
                    )
                ])
            ], style={'width': '100%'})
        ], style={'borderBottom': '1px solid #eee'}))
    
    return html.Div([
        summary,
        html.Table(
            [table_header, html.Tbody(table_rows)],
            style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'backgroundColor': 'white',
                'fontFamily': 'Arial, sans-serif'
            }
        )
    ])

# Add new callback for Learn More button clicks
@app.callback(
    Output({'type': 'explanation-text', 'index': MATCH}, 'style'),
    Input({'type': 'learn-more-button', 'index': MATCH}, 'n_clicks'),
    State({'type': 'explanation-text', 'index': MATCH}, 'style'),
    prevent_initial_call=True
)
def toggle_explanation(n_clicks, current_style):
    """Toggle the visibility of the explanation text."""
    if current_style['display'] == 'none':
        current_style['display'] = 'block'
    else:
        current_style['display'] = 'none'
    return current_style

# Update the callback to use the new filter function
@app.callback(
    Output('top-examples-content', 'children'),
    [Input('example-dataset-dropdown', 'value'),
     Input('improvement-range-slider', 'value'),
     Input('improvement-type-dropdown', 'value'),
     Input('dataset-dropdown', 'value')]
)
def update_examples_content(dataset_selection, improvement_range, improvement_type, selected_datasets):
    """Update examples content based on dataset selection and filters."""
    # First filter by selected datasets
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    
    # Then apply the improvement range and type filters
    filtered_df = filter_dataset_by_selection(
        filtered_df, 
        dataset_selection, 
        improvement_range, 
        improvement_type
    )
    
    return create_examples_content(
        filtered_df, 
        dataset_selection, 
        improvement_range, 
        improvement_type
    )

def optimization_trends_normalized(data):
    # Similar to optimization_trends but for normalized data
    # Ensure data is sorted by 'task_id'
    data = data.sort_values('task_id')

    # Compute rolling mean and standard error for each source
    window_size = 10
    data_grouped = data.groupby('source')

    fig_trends = make_subplots(rows=1, cols=1)

    for name, group in data_grouped:
        group = group.sort_values('task_id')
        group['rolling_energy'] = group['energy_improvement_normalized'].rolling(window_size).mean()
        group['rolling_energy_se'] = group['energy_improvement_normalized'].rolling(window_size).sem()

        # Normalized Energy Improvement Trend per source
        fig_trends.add_trace(go.Scatter(
            x=group['task_id'], y=group['rolling_energy'],
            mode='lines', name=f'Normalized Energy Improvement ({name})',
            error_y=dict(
                type='data',
                array=group['rolling_energy_se'],
                visible=True
            )
        ))

    fig_trends.update_layout(
        title='Normalized Energy Improvement Trends with Error Bars by Source',
        xaxis_title='Task ID', yaxis_title='Normalized Improvement (0-1)',
        height=500
    )
    return fig_trends

def correlation_heatmap(data):
    # Correlation heatmap per source
    performance_metrics = ['energy_consumed', 'execution_time', 'cpu_usage', 'memory_usage']

    # Compute correlation for each source
    data_grouped = data.groupby('source')
    figs = []
    for name, group in data_grouped:
        correlation = group[performance_metrics].corr()
        fig_corr = px.imshow(
            correlation, text_auto=True, aspect='auto', color_continuous_scale='thermal',
            title=f'Core Performance Metrics Correlation ({name})'
        )
        fig_corr.update_layout(height=600)
        figs.append(fig_corr)

    return figs  # Return a list of figures

def success_categories_bar(data):
    # Success categories bar chart per source
    core_metrics = ['energy_improvement', 'time_improvement', 'cpu_improvement']
    data_grouped = data.groupby('source')

    figs = []
    for name, group in data_grouped:
        success_metrics = group.groupby('optimization_success')[core_metrics].mean().reset_index()
        fig_success = px.bar(
            success_metrics.melt(id_vars='optimization_success', var_name='Metric', value_name='Average Improvement'),
            x='optimization_success', y='Average Improvement', color='Metric', barmode='group',
            title=f'Average Improvements by Success Category ({name})'
        )
        fig_success.update_layout(height=500)
        figs.append(fig_success)

    return figs

def compute_bootstrap_ci(data, metric, n_bootstraps=1000, ci_level=0.95):
    data = data[metric].dropna().values
    res = bootstrap((data,), np.mean, n_resamples=n_bootstraps, confidence_level=ci_level, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

def summary_statistics_table(data):
    # Summary statistics table per source
    core_metrics = ['energy_improvement', 'time_improvement', 'cpu_improvement']
    data_grouped = data.groupby('source')

    figs = []
    for name, group in data_grouped:
        summary_stats = pd.DataFrame(columns=['Metric', 'Mean Improvement', '95% CI Lower', '95% CI Upper', 'Std Dev', 'Success Rate (%)'])

        for m in core_metrics:
            mean_improvement = group[m].mean()
            std_dev = group[m].std()
            success_rate = (group[m] > 0).mean() * 100
            ci_lower, ci_upper = compute_bootstrap_ci(group, m)

            new_row = pd.DataFrame({
                'Metric': [m.replace('_', ' ').title()],
                'Mean Improvement': [mean_improvement],
                '95% CI Lower': [ci_lower],
                '95% CI Upper': [ci_upper],
                'Std Dev': [std_dev],
                'Success Rate (%)': [success_rate]
            })
            summary_stats = pd.concat([summary_stats, new_row], ignore_index=True)

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(summary_stats.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[summary_stats[col] for col in summary_stats.columns], fill_color='lavender', align='left')
        )])
        fig_table.update_layout(title=f'Core Optimization Metrics Summary with Confidence Intervals ({name})', height=300)
        figs.append(fig_table)

    return figs

def statistical_tests_graph(data):
    """
    Generate a bar graph of statistical test results comparing metrics across sources.

    Args:
    data (pd.DataFrame): The DataFrame containing the metrics for analysis.

    Returns:
    fig (plotly.graph_objs.Figure): Plotly figure object displaying the test results.
    """
    # Define metrics to compare across sources
    metrics = [
        'energy_improvement',
        'time_improvement',
        'cpu_improvement',
        # Add other metrics as needed
    ]

    test_results = []

    # Perform Kruskal-Wallis H-test for each metric across sources
    for metric in metrics:
        data_by_source = [group[metric].dropna() for name, group in data.groupby('source')]
        kruskal_test = kruskal(*data_by_source)

        test_results.append({
            'Metric': metric.replace('_', ' ').title(),
            'Kruskal Statistic': kruskal_test.statistic,
            'P-Value': kruskal_test.pvalue
        })

    # Convert the results list to a DataFrame
    df_tests = pd.DataFrame(test_results)

    # Create bar chart for Kruskal-Wallis test results
    fig = px.bar(
        df_tests,
        x='Metric',
        y='Kruskal Statistic',
        text=df_tests['P-Value'].apply(lambda x: f"p={x:.3e}"),
        title='Kruskal-Wallis Test Results Across Sources'
    )

    fig.update_traces(textposition='outside', marker_color='lightskyblue')
    fig.update_layout(
        yaxis_title='Kruskal-Wallis Statistic',
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig

# Define visualizations as separate functions
def model_performance_comparison(data):
    # Scatter plot comparing original and optimized energy consumption, colored by model (8b vs 70b)
    fig = px.scatter(
        data,
        x='energy_consumed',
        y='optimized_energy_consumed',
        color='model',
        title='Energy Optimization Impact by Model (RAPL Datasets)',
        labels={'energy_consumed': 'Original Energy Consumed', 'optimized_energy_consumed': 'Optimized Energy Consumed'},
        hover_data=['dataset']
    )
    fig.add_shape(
        type='line',
        x0=data['energy_consumed'].min(),
        y0=data['energy_consumed'].min(),
        x1=data['energy_consumed'].max(),
        y1=data['energy_consumed'].max(),
        line=dict(color='Red', dash='dash')
    )
    fig.update_layout(height=600)
    return fig

def model_optimization_trends(data):
    # Ensure data is sorted by 'task_id'
    data = data.sort_values('task_id')

    # Compute rolling mean and standard error for each model (8b vs 70b)
    window_size = 10
    data_grouped = data.groupby('model')

    fig_trends = make_subplots(rows=1, cols=1)

    for name, group in data_grouped:
        group = group.sort_values('task_id')
        group['rolling_energy'] = group['energy_improvement'].rolling(window_size).mean()
        group['rolling_energy_se'] = group['energy_improvement'].rolling(window_size).sem()

        # Energy Improvement Trend per model
        fig_trends.add_trace(go.Scatter(
            x=group['task_id'], y=group['rolling_energy'],
            mode='lines', name=f'Energy Improvement ({name})',
            error_y=dict(
                type='data',
                array=group['rolling_energy_se'],
                visible=True
            )
        ))

    fig_trends.update_layout(
        title='Energy Improvement Trends with Error Bars by Model (RAPL Datasets)',
        xaxis_title='Task ID', yaxis_title='Improvement (%)',
        height=500
    )
    return fig_trends

def model_success_categories_bar(data):
    # Success categories bar chart per model
    core_metrics = ['energy_improvement', 'time_improvement', 'cpu_improvement']
    data_grouped = data.groupby('model')

    figs = []
    for name, group in data_grouped:
        success_metrics = group.groupby('optimization_success')[core_metrics].mean().reset_index()
        fig_success = px.bar(
            success_metrics.melt(id_vars='optimization_success', var_name='Metric', value_name='Average Improvement'),
            x='optimization_success', y='Average Improvement', color='Metric', barmode='group',
            title=f'Average Improvements by Success Category ({name})'
        )
        fig_success.update_layout(height=500)
        figs.append(fig_success)

    return figs

def model_summary_statistics_table(data):
    # Summary statistics table per model
    core_metrics = ['energy_improvement', 'time_improvement', 'cpu_improvement']
    data_grouped = data.groupby('model')

    figs = []
    for name, group in data_grouped:
        summary_stats = pd.DataFrame(columns=['Metric', 'Mean Improvement', '95% CI Lower', '95% CI Upper', 'Std Dev', 'Success Rate (%)'])

        for m in core_metrics:
            mean_improvement = group[m].mean()
            std_dev = group[m].std()
            success_rate = (group[m] > 0).mean() * 100
            ci_lower, ci_upper = compute_bootstrap_ci(group, m)

            new_row = pd.DataFrame({
                'Metric': [m.replace('_', ' ').title()],
                'Mean Improvement': [mean_improvement],
                '95% CI Lower': [ci_lower],
                '95% CI Upper': [ci_upper],
                'Std Dev': [std_dev],
                'Success Rate (%)': [success_rate]
            })
            summary_stats = pd.concat([summary_stats, new_row], ignore_index=True)

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(summary_stats.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[summary_stats[col] for col in summary_stats.columns], fill_color='lavender', align='left')
        )])
        fig_table.update_layout(title=f'Model Optimization Metrics Summary with Confidence Intervals ({name})', height=300)
        figs.append(fig_table)

    return figs

def model_statistical_tests_graph(data):
    """
    Generate a bar graph of statistical test results comparing metrics across models.

    Args:
    data (pd.DataFrame): The DataFrame containing the metrics for analysis.

    Returns:
    fig (plotly.graph_objs.Figure): Plotly figure object displaying the test results.
    """

    # Define metrics to compare across models
    metrics = [
        'energy_improvement',
        'time_improvement',
        'cpu_improvement',
    ]

    test_results = []

    # Perform Kruskal-Wallis H-test for each metric across models
    for metric in metrics:
        data_by_model = [group[metric].dropna() for name, group in data.groupby('model')]
        kruskal_test = kruskal(*data_by_model)

        test_results.append({
            'Metric': metric.replace('_', ' ').title(),
            'Kruskal Statistic': kruskal_test.statistic,
            'P-Value': kruskal_test.pvalue
        })

    # Convert the results list to a DataFrame
    df_tests = pd.DataFrame(test_results)

    # Create bar chart for Kruskal-Wallis test results
    fig = px.bar(
        df_tests,
        x='Metric',
        y='Kruskal Statistic',
        text=df_tests['P-Value'].apply(lambda x: f"p={x:.3e}"),
        title='Kruskal-Wallis Test Results Across Models (RAPL Datasets)'
    )

    fig.update_traces(textposition='outside', marker_color='lightskyblue')
    fig.update_layout(
        yaxis_title='Kruskal-Wallis Statistic',
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig

def create_profiling_visualizations(data):
    """Creates profiling visualizations using plotly instead of matplotlib"""
    
    # 1. Bar plots for main metrics
    metrics = ['execution_time', 'cpu_usage', 'memory_usage', 'cyclomatic_complexity', 
              'complexity_score', 'pylint_score', 'combined_score']
    
    figs = []
    
    # Bar plots for each metric
    for metric in metrics:
        fig = px.bar(
            data,
            x='task_id',
            y=metric,
            title=f'{metric.replace("_", " ").title()} by Function',
            labels={metric: metric.replace('_', ' ').title()}
        )
        fig.update_layout(height=400)
        figs.append(dcc.Graph(figure=fig))
    
    # Correlation heatmap
    correlation_matrix = data[metrics].corr()
    fig_heatmap = px.imshow(
        correlation_matrix,
        labels=dict(color="Correlation"),
        title='Correlation Heatmap of Metrics',
        color_continuous_scale='RdBu'
    )
    fig_heatmap.update_layout(height=600)
    figs.append(dcc.Graph(figure=fig_heatmap))
    
    # Scatter plot: Execution Time vs Energy Consumed
    fig_scatter = px.scatter(
        data,
        x='execution_time',
        y='energy_consumed',
        title='Execution Time vs Energy Consumed',
        labels={
            'execution_time': 'Execution Time (seconds)',
            'energy_consumed': 'Energy Consumed (Joules)'
        }
    )
    fig_scatter.update_layout(height=500)
    figs.append(dcc.Graph(figure=fig_scatter))
    
    # Box plot of main metrics
    df_melted = data[metrics].melt()
    fig_box = px.box(
        df_melted,
        x='variable',
        y='value',
        title='Distribution of Metrics'
    )
    fig_box.update_layout(height=500)
    figs.append(dcc.Graph(figure=fig_box))
    
    return figs

def execution_time_vs_energy_consumed(data):
    """Create a scatter plot comparing normalized execution time and energy consumed for RAPL data."""
    # Filter for RAPL data only
    rapl_data = data[data['source'] == 'rapl'].copy()
    
    # Create normalized columns if they don't exist
    if 'execution_time_normalized' not in rapl_data.columns:
        rapl_data['execution_time_normalized'] = (rapl_data['execution_time'] - rapl_data['execution_time'].min()) / \
                                               (rapl_data['execution_time'].max() - rapl_data['execution_time'].min())
    
    if 'energy_consumed_normalized' not in rapl_data.columns:
        rapl_data['energy_consumed_normalized'] = (rapl_data['energy_consumed'] - rapl_data['energy_consumed'].min()) / \
                                                (rapl_data['energy_consumed'].max() - rapl_data['energy_consumed'].min())
    
    # Create scatter plot with normalized values
    fig = px.scatter(
        rapl_data,
        x='execution_time_normalized',
        y='energy_consumed_normalized',
        color='model',
        title='Normalized Execution Time vs Energy Consumed (RAPL Data)',
        labels={
            'execution_time_normalized': 'Normalized Execution Time (0-1)',
            'energy_consumed_normalized': 'Normalized Energy Consumed (0-1)'
        },
        hover_data=['dataset']
    )
    
    # Add a diagonal reference line
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color='red', dash='dash'),
        name='Perfect Correlation Line'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        annotations=[
            dict(
                text="Perfect correlation line (y=x)",
                x=0.5,
                y=1.05,
                showarrow=False,
                font=dict(color='red')
            )
        ]
    )
    
    # Add correlation coefficient annotation
    correlation = rapl_data['execution_time_normalized'].corr(rapl_data['energy_consumed_normalized'])
    fig.add_annotation(
        text=f"Correlation: {correlation:.3f}",
        x=0.05,
        y=0.95,
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
    
    return fig  # Return only the figure

# Define the layout of the Dash app
app.layout = html.Div([
    # Add the CSS styles using a style dictionary
    html.Div(style={
        'cssText': '''
            .range-slider {
                padding: 25px 0 35px 0;
            }
            .rc-slider-mark-text {
                white-space: nowrap;
                margin-top: 10px;
            }
            .rc-slider-handle {
                border-color: #4CAF50;
                background-color: #4CAF50;
            }
            .rc-slider-track {
                background-color: #4CAF50;
            }
            .rc-slider-rail {
                background-color: #e9ecef;
            }
        '''
    }),
    
    html.H1("Optimization Analysis Dashboard", style={'text-align': 'center'}),
    
    html.Div([
        html.Label('Select Dataset(s):'),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in df['dataset'].unique()],
                     value=df['dataset'].unique().tolist(),
                     multi=True
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    dcc.Tabs([
        dcc.Tab(label='Analysis', children=[
            dcc.Graph(id='core-performance-dashboard'),
            dcc.Graph(id='optimization-trends'),
            dcc.Graph(id='optimization-trends-normalized'),
            dcc.Graph(id='execution-time-vs-energy-consumed'),  # Graph only
            html.Div([
                html.Button(
                    "Learn More About Time-Energy Relationship",
                    id='time-energy-learn-more-button',
                    style={
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'padding': '10px 15px',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'marginTop': '10px'
                    }
                ),
                html.Div(
                    id='time-energy-explanation',
                    style={'display': 'none'},
                    children=html.Div([
                        html.H3("Understanding the Time-Energy Relationship", 
                               style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        html.Div([
                            html.P([
                                html.Strong("The Core Hypothesis: "),
                                "The fundamental hypothesis in energy-efficient programming suggests that reducing execution time typically leads to proportional reductions in energy consumption."
                            ]),
                            
                            html.H4("Key Observations from the Data:", 
                                   style={'color': '#2c3e50', 'marginTop': '20px'}),
                            
                            html.Ul([
                                html.Li([
                                    html.Strong("Strong Correlation: "),
                                    "The scatter plot demonstrates a strong positive correlation between execution time and energy consumption, supporting the basic hypothesis."
                                ]),
                                html.Li([
                                    html.Strong("Linear Relationship: "),
                                    "Points clustering around the diagonal line (y=x) indicate that changes in execution time often result in proportional changes in energy consumption."
                                ]),
                                html.Li([
                                    html.Strong("Model Variations: "),
                                    "Different models (8B vs 70B) show similar patterns, suggesting this relationship holds across model sizes."
                                ])
                            ], style={'lineHeight': '1.6'}),
                            
                            html.H4("Understanding the Visualization:", 
                                   style={'color': '#2c3e50', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li([
                                    html.Strong("Diagonal Line: "),
                                    "Represents perfect correlation between time and energy (y=x). Points on this line indicate proportional relationships."
                                ]),
                                html.Li([
                                    html.Strong("Below Diagonal: "),
                                    "Points here show better energy efficiency than expected based on execution time alone."
                                ]),
                                html.Li([
                                    html.Strong("Above Diagonal: "),
                                    "Points here indicate higher energy consumption than expected from execution time."
                                ])
                            ], style={'lineHeight': '1.6'})
                        ])
                    ])
                )
            ]),
            html.Div(id='correlation-heatmap'),
            html.Div(id='success-categories-bar'),
            html.Div(id='summary-statistics-table'),
            dcc.Graph(id='statistical-tests'),
        ]),
        
        dcc.Tab(label='Model Comparison (RAPL)', children=[
            dcc.Graph(id='model-performance-comparison'),
            dcc.Graph(id='model-optimization-trends'),
            html.Div(id='model-success-categories-bar'),
            html.Div(id='model-summary-statistics-table'),
            dcc.Graph(id='model-statistical-tests')
        ]),
        
        dcc.Tab(label='Code Examples', children=[
            html.Div([
                html.Label('Select Dataset to View Examples:'),
                dcc.Dropdown(
                    id='example-dataset-dropdown',
                    options=[
                        {'label': 'All Datasets (RAPL + Powermetrics)', 'value': 'all'},
                        {'label': 'All RAPL (Linux) Datasets', 'value': 'rapl'},
                        {'label': 'All Powermetrics (macOS) Datasets', 'value': 'powermetrics'},
                    ],
                    value='rapl',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
                
                html.Label('Select Improvement Metric:'),
                dcc.Dropdown(
                    id='improvement-type-dropdown',
                    options=[
                        {'label': 'Energy Improvement', 'value': 'energy_improvement'},
                        {'label': 'CPU Usage Improvement', 'value': 'cpu_improvement'},
                        {'label': 'Execution Time Improvement', 'value': 'time_improvement'}
                    ],
                    value='energy_improvement',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
                
                html.Label('Filter by Improvement Range:'),
                dcc.RangeSlider(
                    id='improvement-range-slider',
                    min=-100,
                    max=100,
                    step=5,
                    marks={
                        i: {'label': f'{i}%', 
                            'style': {'transform': 'rotate(45deg)', 
                                     'transform-origin': 'top right'}}
                        for i in range(-30, 30, 20)
                    },
                    value=[0, 100],
                    allowCross=False,
                    tooltip={
                        'placement': 'bottom',
                        'always_visible': True,
                        'template': '{value}%'
                    },
                    className='range-slider'
                )
            ], style={
                'width': '50%', 
                'margin': '20px auto',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
            
            # Content area for examples
            html.Div(id='top-examples-content')
        ])
    ])
])


# Callbacks to update figures based on selected datasets
@app.callback(
    Output('core-performance-dashboard', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_core_performance_dashboard(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return core_performance_dashboard(filtered_df)

@app.callback(
    Output('optimization-trends', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_optimization_trends(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return optimization_trends(filtered_df)

@app.callback(
    Output('optimization-trends-normalized', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_optimization_trends_normalized(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return optimization_trends_normalized(filtered_df)

@app.callback(
    Output('correlation-heatmap', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_correlation_heatmap(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    figs = correlation_heatmap(filtered_df)
    return [dcc.Graph(figure=fig) for fig in figs]

@app.callback(
    Output('success-categories-bar', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_success_categories_bar(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    figs = success_categories_bar(filtered_df)
    return [dcc.Graph(figure=fig) for fig in figs]

@app.callback(
    Output('summary-statistics-table', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_summary_statistics_table(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    figs = summary_statistics_table(filtered_df)
    return [dcc.Graph(figure=fig) for fig in figs]

@app.callback(
    Output('statistical-tests', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_statistical_tests(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return statistical_tests_graph(filtered_df)

@app.callback(
    Output('model-performance-comparison', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_model_performance_comparison(selected_datasets):
    filtered_df = df_rapl[df_rapl['dataset'].isin(selected_datasets)]
    return model_performance_comparison(filtered_df)

@app.callback(
    Output('model-optimization-trends', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_model_optimization_trends(selected_datasets):
    filtered_df = df_rapl[df_rapl['dataset'].isin(selected_datasets)]
    return model_optimization_trends(filtered_df)

@app.callback(
    Output('model-success-categories-bar', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_model_success_categories_bar(selected_datasets):
    filtered_df = df_rapl[df_rapl['dataset'].isin(selected_datasets)]
    figs = model_success_categories_bar(filtered_df)
    return [dcc.Graph(figure=fig) for fig in figs]

@app.callback(
    Output('model-summary-statistics-table', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_model_summary_statistics_table(selected_datasets):
    filtered_df = df_rapl[df_rapl['dataset'].isin(selected_datasets)]
    figs = model_summary_statistics_table(filtered_df)
    return [dcc.Graph(figure=fig) for fig in figs]

@app.callback(
    Output('model-statistical-tests', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_model_statistical_tests(selected_datasets):
    filtered_df = df_rapl[df_rapl['dataset'].isin(selected_datasets)]
    return model_statistical_tests_graph(filtered_df)

@app.callback(
    Output('top-examples-layout-2', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_top_examples_layout_2(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return top_examples_layout(filtered_df)

@app.callback(
    Output('profiling-visualizations', 'children'),
    [Input('dataset-dropdown', 'value')]
)
def update_profiling_visualizations(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return create_profiling_visualizations(filtered_df)

@app.callback(
    Output('execution-time-vs-energy-consumed', 'figure'),
    [Input('dataset-dropdown', 'value')]
)
def update_execution_time_vs_energy_consumed(selected_datasets):
    filtered_df = df[df['dataset'].isin(selected_datasets)]
    return execution_time_vs_energy_consumed(filtered_df)

# Callback to toggle the explanation visibility
@app.callback(
    Output('time-energy-explanation', 'style'),
    Input('time-energy-learn-more-button', 'n_clicks'),
    State('time-energy-explanation', 'style'),
    prevent_initial_call=True
)
def toggle_time_energy_explanation(n_clicks, current_style):
    """Toggle the visibility of the time-energy relationship explanation."""
    if current_style['display'] == 'none':
        return {'display': 'block', 'marginTop': '20px'}
    return {'display': 'none'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
