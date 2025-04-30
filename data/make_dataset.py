import argparse
import ast
import json
import os
from shutil import rmtree
import re
from pathlib import Path
from tqdm import tqdm

import requests

parser = argparse.ArgumentParser('get_context.py', 'Generate the context for a focal method')
parser.add_argument('--repos_dir', default='repos', help='Temporary folder to store GitHub repository')
parser.add_argument('--test_file', help='The test file to provide context for')
parser.add_argument('--test_method', help='The test method to provide context for')
parser.add_argument('--output_file', help='The output JSONL file to save the result to')
parser.add_argument('--max_repos', type=int, default=None, help='Max number of repos to collect samples from')
parser.add_argument('--starting_point', type=int, default=None, help='Starting point in case your run bailed')

args = parser.parse_args()


def get_method_sig(method: ast.FunctionDef, indent=0):
    s = ''
    s += f'{"    " * indent}def {method.name}('
    s += ast.unparse(method.args)
    s += ')'
    if method.returns:
        s += ' -> ' + ast.unparse(method.returns)
    s += ': ...\n'
    return s

def download_to(path, url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        if r.status_code == 200:
            with open(path, 'w') as f:
                f.write(r.text)
        else:
            raise Exception(f"Failed to download {url} (status {r.status_code})")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download {url} ({e})")


def get_github_files(repo_user, repo_name, sha, test_path, focal_path):
    try:
        base_url = f"https://raw.githubusercontent.com/{repo_user}/{repo_name}/{sha}"
        raw_test_url = f"{base_url}/{test_path}"
        raw_focal_url = f"{base_url}/{focal_path}"

        download_path = os.path.join(args.repos_dir, repo_user, repo_name)

        test_download_path = os.path.join(download_path, test_path)
        os.makedirs(os.path.dirname(test_download_path), exist_ok=True)
        download_to(test_download_path, raw_test_url)

        focal_download_path = os.path.join(download_path, focal_path)
        os.makedirs(os.path.dirname(focal_download_path), exist_ok=True)
        download_to(focal_download_path, raw_focal_url)

        return True
    except Exception as e:
        print(f"❌ Skipping {repo_user}/{repo_name}@{sha} — {e}")
        return False

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

def get_test_framework(tests_json_data, test_file, test_method):
    for entry in tests_json_data:
        if entry['file_path'] == test_file:
            for method in entry['test_methods']:
                if method['method_name'] == test_method:
                    return entry['test_framework']
    return "unknown"

def make_samples(tests_json_data, focal_data):
    samples = []
    for test_file, test_data in focal_data.items():
        for method_name, method_info in test_data['methods'].items():
            code = method_info['focal_context']
            test = method_info['test_method']
            framework = get_test_framework(tests_json_data, test_file, method_name)
            samples.append({
                "code": f"{code}",
                "test": f"{test}",
                "framework": framework
            })
        return samples

def make_samples_repo(repo_dir):

    json_files = list(Path(repo_dir).glob('*.focal.json'))
    if not json_files:
        raise FileNotFoundError(f"No .focal.json file found in {repo_dir}")
    json_file = json_files[0]
    json_file = str(json_file)

    parts = json_file.split(os.path.sep)
    repo_user = parts[-3]
    repo_name = parts[-2]
    sha = parts[-1].replace('.focal.json', '')
    dir = os.path.join(args.repos_dir, repo_user, repo_name)
    test_dir = os.path.join("data_test/data", repo_user, repo_name)

    if not os.path.exists(test_dir):
        print(test_dir)
        print('Directory \'/data_test/<user>/<repo>/\' must exist. You probably didn\'t unzip the raw data.')
        exit(1)

    test_files = list(Path(test_dir).glob('*.tests.json'))
    if not test_files:
        raise FileNotFoundError(f"No .tests.json file found in {test_dir}")
    test_file = test_files[0]
    #
    # if not os.path.exists(dir):
    #     url = f'https://github.com/{repo_user}/{repo_name}.git'
    #
    #     dest_dir = os.path.join(args.repos_dir, repo_user)
    #     os.makedirs(dest_dir, exist_ok=True)
    #
    #     subprocess.run(['git', 'clone', url], cwd=dest_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #
    # if not os.path.exists(os.path.join(dir, '.git')):
    #     print(f'Unable to clone "{repo_user}/{repo_name}" from GitHub.')
    #     return
    #
    # subprocess.run(['git', 'checkout', sha], cwd=dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(test_file, 'r') as tfile:
        tests_json_data = json.load(tfile)

    with open(json_file, 'r') as infile:
        focal_data = json.load(infile)

    # Identify all needed files and fetch them
    try:
        needed_files = set()
        for testfile, test_data in focal_data.items():
            needed_files.add(testfile)
            needed_files.add(test_data['focal_file'])

        # download once per unique file
        for path in needed_files:
            ok = get_github_files(repo_user, repo_name, sha, path, path)
            if not ok:
                print(f"Skipping repo {repo_user}/{repo_name} due to download error.")
                return

    except Exception as e:
        print(f"❌ Failed to fetch required files for {repo_user}/{repo_name}@{sha} — {e}")
        return

    # Continue with main context-fetching logic
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

    # Delete repo, not needed anymore
    if args.repos_dir and os.path.exists(args.repos_dir):
        rmtree(args.repos_dir)

    # Saving the updated focal_data to the output file
    if args.output_file:
        with open(args.output_file, 'a') as outfile:
            samples = make_samples(tests_json_data, focal_data)
            for s in samples:
                jsonl = json.dumps(s)
                outfile.write(jsonl + "\n")
        print(f"Output saved to {args.output_file}")
    else:
        print(json.dumps(focal_data, indent=4))

# For all repos, make data samples and store in the output file

focal_path = "data_focal/data/"

# Compute total number of repos in advance
total_repos = sum(
    len([repo for repo in os.listdir(os.path.join(focal_path, user))
         if os.path.isdir(os.path.join(focal_path, user, repo))])
    for user in os.listdir(focal_path)
)

max_repos = total_repos
if args.max_repos is not None and args.max_repos < total_repos:
    max_repos = args.max_repos

# Setup counter and tqdm
counter = 0
really_fucked_repo_counter = 0
pbar = tqdm(total=max_repos, desc="Processing repos")
for user in os.listdir(focal_path):
    user_path = os.path.join(focal_path, user)
    for repo in os.listdir(user_path):
        # fast forward
        if args.starting_point is not None and counter < args.starting_point:
            counter += 1
            pbar.update(1)
            continue

        # real logic
        repo_path = os.path.join(user_path, repo)
        if os.path.isdir(repo_path):
            try:
                make_samples_repo(repo_path)
            except Exception as e:
                really_fucked_repo_counter += 1
                print(f"❌ Error in repo {repo_path}: {e}")
            counter += 1
            pbar.update(1)

        if counter >= max_repos:
            pbar.close()
            print(f"Completed dataset generation. Saved output to {args.output_file}")
            print(f"Number of repos that didn't complete correctly (no bearing on dataset): {really_fucked_repo_counter}")
            exit(0)


pbar.close()
