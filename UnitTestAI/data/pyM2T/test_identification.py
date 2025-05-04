import argparse
import ast
import json
import os
import sys


def get_test_framework(file_path):
    """Returns the discovered Python test framework: unittest or pytest"""

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read(), filename=file_path)
    except SyntaxError:
        return False

    has_unittest = False
    has_pytest = False

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for name in stmt.names:
                if name.name == 'unittest':
                    has_unittest = True
                if name.name == 'pytest':
                    has_pytest = True
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module == 'unittest':
                has_unittest = True
            if stmt.module == 'pytest':
                has_pytest = True

    if has_unittest and has_pytest:
        return 'both'
    if has_unittest:
        return 'unittest'
    if has_pytest:
        return 'pytest'

    return None


def get_name_from_node(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return get_name_from_node(node.attr)
    elif isinstance(node, str):
        return node
    return None


def get_fqn_from_node(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr + '.' + get_fqn_from_node(node.value)
    return ''


def extract_test_methods_and_calls(file_path, framework, source_modules, source_methods):
    """parse Python files and extract test methods and method calls"""

    test_methods = []
    test_imports = []
    test_import_modules = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read(), filename=file_path)

        in_unittest_testcase = False

        # Traverse through all the functions in the Python file
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in source_modules:
                        test_imports.append(name.name)
                        test_import_modules[name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                if node.module in source_modules:
                    for name in node.names:
                        test_imports.append(name.name)
                        if node.module == '.':
                            test_import_modules[name.name] = name.name
                        else:
                            test_import_modules[name.name] = node.module + '.' + name.name
            elif isinstance(node, ast.ClassDef) and any(get_name_from_node(base) == 'TestCase' for base in node.bases):
                in_unittest_testcase = True
            elif (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)):
                if node.name.lower().startswith('test') and (framework == 'pytest' or in_unittest_testcase):
                    method_info = {
                        'method_name': node.name,
                        'line': node.lineno,
                        'line_end': node.end_lineno,
                        'indent': node.col_offset,
                        'called_methods': []
                    }

                    # Walk through the body of the function to find method calls
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Call):
                            # Handle method calls on functions
                            if isinstance(subnode.func, ast.Name) and (subnode.func.id in test_imports or subnode.func.id in source_methods):
                                method_info['called_methods'].append(subnode.func.id)
                            # Handle method calls with attributes (e.g., self.method())
                            elif isinstance(subnode.func, ast.Attribute):
                                fqn = ast.unparse(subnode.func)
                                parts = fqn.split('.')
                                first = parts[0]
                                name = parts[-1]
                                if first in test_imports or name in test_imports or name in source_methods:
                                    if len(parts) > 1 and parts[0] in test_import_modules:
                                        full_import = test_import_modules[parts[0]]
                                        method_info['called_methods'].append(full_import + '.' + '.'.join(parts[1:]))
                                    else:
                                        method_info['called_methods'].append(fqn)

                    if len(method_info['called_methods']):
                        method_info['called_methods'] = list(set(method_info['called_methods']))
                        test_methods.append(method_info)
    except SyntaxError as e:
        print(f'Error parsing file {file_path}: {e}', file=sys.stderr)

    return test_methods, test_import_modules


def find_test_methods_in_repo(repo_path, source_modules, source_methods):
    """Traverse the repository and find all Python files with test methods and the appropriate imports"""

    test_data = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Only process files with unittest or pytest imports
                framework = get_test_framework(file_path)
                if framework:
                    # pytest requires test files to start with 'test_*.py' or end with '*_test.py'
                    if framework == 'pytest' and not (file.startswith('test_') or file.endswith('_test.py')):
                        continue

                    test_methods, test_imports = extract_test_methods_and_calls(file_path, framework, source_modules, source_methods)
                    if test_methods:
                        test_data.append({
                            'file_path': os.path.relpath(file_path, repo_path),
                            'test_framework': framework,
                            'test_imports': test_imports,
                            'test_methods': test_methods
                        })

    return test_data


def find_modules(all_files_data):
    """Return all found source modules"""

    data = {}

    for file_path, file_data in all_files_data.items():
        if '__modulename__' in file_data and not file_data['__modulename__'].startswith('test'):
            data[file_data['__modulename__']] = file_path

    return data


def find_methods(all_files_data):
    """Return all found source methods"""

    data = set()

    for key, value in all_files_data.items():
        if '__modulename__' in value and not value['__modulename__'].startswith('test'):
            for key2, methods in value.items():
                if key2 != '__modulename__':
                    for method in methods.keys():
                        if not method.startswith('test'):
                            data.add(method)

    return data


def save_to_json(data, output_file):
    """Save the extracted data to a JSON file"""

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


def get_module_name(repo_path, path):
    """Find the nearest module name"""

    if not path or path.endswith(os.path.sep + 'src'):
        return ''

    if os.path.isfile(path):
        base = get_module_name(repo_path, os.path.dirname(path))
        last = os.path.basename(path).replace('.py', '')
        if last != '__init__':
            return base + '.' + last
        return base

    if os.path.isfile(os.path.join(path, '__init__.py')):
        base = get_module_name(repo_path, path[:path.rfind(os.path.sep)])
        last = os.path.basename(path).replace(os.path.sep, '.')
        if base:
            return base + '.' + last
        return last

    return ''


def find_classes_and_methods_in_repo(repo_path):
    """Traverse the repository, find the Python files in the directories from module_dirs,
    and return methods along with their classes (if any)"""

    focal_files_data = {}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):  # Process only Python files
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=file)
                        data = {
                            '__modulename__': get_module_name(repo_path, file_path),
                            '__global__': {},
                        }

                        # Extract classes and methods from the AST
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                data[node.name] = {}
                                # Look for methods within the class
                                for sub_node in node.body:
                                    if isinstance(sub_node, ast.FunctionDef) or isinstance(sub_node, ast.AsyncFunctionDef):
                                        sub_node.parent = node
                                        data[node.name][sub_node.name] = {
                                            'line': sub_node.lineno,
                                            'line_end': sub_node.end_lineno,
                                            'indent': sub_node.col_offset,
                                        }
                                if '__init__' not in data[node.name]:
                                    data[node.name]['__init__'] = {
                                        'line': node.lineno,
                                        'line_end': node.lineno,
                                        'indent': node.col_offset,
                                    }
                            elif (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)) and hasattr(node, 'parent'):
                                for nested in node.body:
                                    if isinstance(nested, ast.FunctionDef):
                                        nested.parent = node
                            elif (isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)) and not hasattr(node, 'parent'):
                                # Functions defined outside of classes
                                data['__global__'][node.name] = {
                                    'line': node.lineno,
                                    'line_end': node.end_lineno,
                                    'indent': node.col_offset,
                                }

                        # Store the methods data for the focal file
                        focal_files_data[file_path] = data
                    except SyntaxError as e:
                        print(f'Error parsing file {file_path}: {e}', file=sys.stderr)

    return focal_files_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_identification.py', 'Extract file/class/method info from a Python repository')
    parser.add_argument('--outfile', type=str,
                        help='Filename to save the extracted data')
    parser.add_argument('--outtests', type=str,
                        help='Filename to save the extracted tests data')
    parser.add_argument('repo_path', type=str, help='Path to the repository')
    args = parser.parse_args()

    all_files_data = find_classes_and_methods_in_repo(args.repo_path)

    source_modules = find_modules(all_files_data)
    source_methods = find_methods(all_files_data)

    # Find test methods and method calls in the cloned repository
    test_data = find_test_methods_in_repo(args.repo_path, source_modules, source_methods)

    # Save the results to a JSON file
    if len(test_data):
        output_file = args.outfile if args.outfile else os.path.basename(args.repo_path) + '.json'
        save_to_json(all_files_data, output_file)

        tests_output_file = args.outtests if args.outtests else os.path.basename(args.repo_path) + '.tests.json'
        save_to_json(test_data, tests_output_file)
    else:
        print('No test methods found in the repository', file=sys.stderr)
        sys.exit(-1)
