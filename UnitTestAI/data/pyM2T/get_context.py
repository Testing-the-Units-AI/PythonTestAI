import argparse
import ast
import json
import os
import re
import subprocess
import sys


parser = argparse.ArgumentParser('get_context.py', 'Generate the context for a focal method')
parser.add_argument('json_file', help='Full path to the focal JSON file to generate context for')
parser.add_argument('--repos_dir', default='repos', help='Folder that stores GitHub repositories')
parser.add_argument('--test_file', help='The test file to provide context for')
parser.add_argument('--test_method', help='The test method to provide context for')
parser.add_argument('--output_file', help='The output JSON file to save the result')

args = parser.parse_args()

if not args.json_file.endswith('focal.json'):
    print('File must be a focal JSON file')
    exit(1)

parts = args.json_file.split(os.path.sep)
repo_user = parts[-3]
repo_name = parts[-2]
sha = parts[-1].replace('.focal.json', '')
dir = os.path.join(args.repos_dir, repo_user, repo_name)

if not os.path.exists(dir):
    url = f'https://github.com/{repo_user}/{repo_name}.git'

    dest_dir = os.path.join(args.repos_dir, repo_user)
    os.makedirs(dest_dir, exist_ok=True)

    subprocess.run(['git', 'clone', url], cwd=dest_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if not os.path.exists(os.path.join(dir, '.git')):
    print(f'Unable to clone "{repo_user}/{repo_name}" from GitHub.')
    sys.exit(1)

subprocess.run(['git', 'checkout', sha], cwd=dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

with open(args.json_file, 'r') as infile:
    focal_data = json.load(infile)


def get_method_sig(method: ast.FunctionDef, indent=0):
    s = ''
    s += f'{"    " * indent}def {method.name}('
    s += ast.unparse(method.args)
    s += ')'
    if method.returns:
        s += ' -> ' + ast.unparse(method.returns)
    s += ': ...\n'
    return s


def get_context(testmethod: ast.FunctionDef, focalmethod: ast.FunctionDef, focalclass: ast.ClassDef, globals):
    ctx = {}

    ctx['test_method'] = ast.unparse(testmethod)

    focal = ''
    if focalclass:
        focal += f'class {focalclass.name}({ast.unparse(focalclass.bases)}):' + '\n'

        s = ast.unparse(focalmethod) + '\n'
        focal += re.sub(r'^', '    ', s, flags=re.MULTILINE)

        # other functions
        funcs = [n for n in focalclass.body if (isinstance(n, ast.AsyncFunctionDef) or isinstance(n, ast.FunctionDef)) and n != focalmethod]
        if funcs:
            focal += '\n'
            for f in funcs:
                focal += get_method_sig(f, 1)

        # instance attributes
        if funcs:
            init = [f for f in funcs if f.name == '__init__']
            if init:
                init = init[0]
                assigns = [ast.unparse(n) for n in init.body if isinstance(n, ast.Assign)]
                assigns = [a for a in assigns if a.startswith('self.')]
                if assigns:
                    focal += '\n'
                    for a in assigns:
                        focal += '    ' + a + '\n'

        # class attributes
        attrs = [ast.unparse(n) for n in focalclass.body if isinstance(n, ast.Assign)]
        if attrs:
            focal += '\n'
            for a in attrs:
                focal += '    ' + a + '\n'
    else:
        focal += ast.unparse(focalmethod)

        if globals:
            focal += '\n'
            for m in globals:
                focal += get_method_sig(m)

    ctx['focal_context'] = focal

    return ctx


for testfile, test_data in focal_data.items():
    if args.test_file and args.test_file != testfile:
        continue

    testfile = os.path.join(dir, testfile)
    with open(testfile, 'r') as infile:
        testcontent = infile.read()

    testtree = ast.parse(testcontent)

    focalfile = os.path.join(dir, test_data['focal_file'])
    with open(focalfile, 'r') as infile:
        filecontent = infile.read()

    tree = ast.parse(filecontent)
    tree.parent = None
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    globals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if node.parent == tree:
                globals.append(node)

    for m in test_data['methods']:
        if args.test_method and args.test_method != m:
            continue

        testmethod = None

        for node in ast.walk(testtree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if node.lineno == test_data['methods'][m]['line']:
                    testmethod = node
                    break

        if not testmethod:
            print(f"Could not find test method {m} in {testfile}")
            continue

        focal = test_data['methods'][m]['focal_method']

        focalclass = None
        focalmethod = None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if node.lineno == focal['line']:
                    focalmethod = node
                    if isinstance(focalmethod.parent, ast.ClassDef):
                        focalclass = focalmethod.parent
                    break

        if not focalmethod:
            print(f"Could not find focal method {test_data['methods'][m]['focal_method']['name']} in {focalfile}")
            continue

        ctx = get_context(testmethod, focalmethod, focalclass, globals)
        test_data['methods'][m]['test_method'] = ctx['test_method']
        test_data['methods'][m]['focal_context'] = ctx['focal_context']

# Saving the updated focal_data to the output file
if args.output_file:
    with open(args.output_file, 'w') as outfile:
        json.dump(focal_data, outfile, indent=4)
    print(f"Output saved to {args.output_file}")
else:
    print(json.dumps(focal_data, indent=4))
