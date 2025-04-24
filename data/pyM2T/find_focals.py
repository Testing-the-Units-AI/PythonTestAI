import argparse
from fuzzywuzzy import fuzz
import json
import os
import sys


def find_focal_file(all_data, test_file, imports):
    imports = [v for k, v in imports.items()] + ['.'.join(v.split('.')[0:-1]) for k, v in imports.items() if '.' in v]

    possible_files = []
    for k, v in all_data.items():
        if v['__modulename__'] in imports:
            possible_files.append(k)

    if not possible_files:
        filenames = { k: os.path.basename(k) for k in all_data.keys() }
        for k, v in filenames.items():
            if test_file.endswith(v):
                possible_files.append(k)

    if not possible_files:
        return None

    if len(possible_files) == 1:
        return possible_files[0]

    filename = os.path.basename(test_file)
    for k, v in {os.path.basename(f): f for f in possible_files}.items():
        if test_file.endswith(k):
            return v

    ratios = [(f, fuzz.ratio(filename, os.path.basename(f))) for f in possible_files]
    best = max(ratios, key=lambda x: x[1])
    return best[0]


def find_focal_class(file_data, called_methods):
    for m in called_methods:
        parts = m.split('.')
        if parts[-1] in file_data:
            return m

    return None


def find_focal_method(file_data, test_method, called_methods):
    focal = None
    for m in called_methods:
        parts = m.split('.')
        if test_method.endswith(parts[-1]):
            focal = m

    if not focal and called_methods:
        ratios = [(m, fuzz.ratio(test_method, m)) for m in called_methods]
        best = max(ratios, key=lambda x: x[1])
        if best[1] > 50:
            focal = best[0]

    if focal:
        focal = focal.split('.')[-1]
        if '__global__' in file_data and focal in file_data['__global__']:
            data = dict(file_data['__global__'][focal])
            data['name'] = focal
            return data
        for k, v in file_data.items():
            if k != '__modulename__' and focal in v:
                data = dict(v[focal])
                data['name'] = focal
                return data

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('find_focals.py', 'Find focal tests in a Python repo')
    parser.add_argument('infile', type=str,
                        help='Filename to the extracted data')
    parser.add_argument('intests', type=str,
                        help='Filename to the extracted tests data')
    parser.add_argument('-outfile', type=str,
                        help='Filename to store discovered focal data')
    args = parser.parse_args()

    with open(args.infile, 'r') as infile:
        all_data = json.load(infile)
    with open(args.intests, 'r') as intests:
        all_tests = json.load(intests)

    data = {}

    for test_file in all_tests:
        print(f'Processing test file "{test_file["file_path"]}"')

        focal_file = find_focal_file(all_data, test_file['file_path'], test_file['test_imports'])

        if focal_file:
            data[test_file['file_path']] = {
                'focal_file': focal_file,
                'methods': {},
            }

            for method in test_file['test_methods']:
                if len(method['called_methods']) == 0:
                    continue

                print(f'  -> Processing test "{method["method_name"]}"')
                focal_method = find_focal_method(all_data[focal_file], method['method_name'], method['called_methods'])

                if focal_method:
                    data[test_file['file_path']]['methods'][method['method_name']] = {
                        'line': method['line'],
                        'line_end': method['line_end'],
                        'indent': method['indent'],
                        'focal_class': find_focal_class(all_data[focal_file], method['called_methods']),
                        'focal_method': focal_method,
                    }

            if not len(data[test_file['file_path']]['methods']):
                del data[test_file['file_path']]

    if data:
        outfile = args.outfile if args.outfile else args.infile[0:-5] + '.focal.json'
        with open(outfile, 'w', encoding='utf-8') as out:
            json.dump(data, out, indent=4)
        print(f'Wrote focal data to "{outfile}"')
    else:
        print('No focal methods found in the repository', file=sys.stderr)
        sys.exit(-1)
