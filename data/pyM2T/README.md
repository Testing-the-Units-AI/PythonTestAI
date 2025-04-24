# pyMethods2Test

This repository contains a curated dataset of Python unit test cases of over 2 million test methods, extracted from a collection of 88,846 Python projects. The dataset is designed to support research and development in the areas of software testing, test-to-code traceability, and automated test generation.

The data is stored in two ZIP files.  If you only want to use the pre-mined focal data, download the `focal-data.zip` file. The larger `raw-data.zip` file includes the raw data used to generate the focal data, such as the classes and methods extracted from the repositories.  This second file may be useful as the starting point of other tasks.

## Key Features of the Dataset

1. Test Methods:
More than 2 million unit test methods extracted from Python projects. Each test method is linked to the original method it is testing, allowing for detailed analysis.
2. Test Frameworks:
The dataset includes test cases written in two of the most widely used Python testing frameworks:
    + `Unittest`
    + `Pytest`
3. Focal File:
The dataset includes the focal file for each test method, which contains the code related to the test.
4. Focal Class:
For object-oriented projects, the dataset identifies the focal class that contains the method being tested.
5. Focal Methods:
Each test method is associated with a focal method. This enables analysis of the relationship between tests and the software they validate.
6. Annotations: 
Detailed annotations at both the file and method levels, including:
    + `Instance Attributes:` Attributes associated with the class instance.
    + `Method Signatures:`  The signatures of methods within a class, standalone functions, and the specific focal method being tested.

    These annotations help in linking test methods to their corresponding focal methods and classes, supporting test-to-code traceability.

## Installation

Install the dependencies in the `requirements.txt`.

### Test Identification

If you wish to run the test identification script on your own repositories, follow these steps:

- Run `test_identification.py` as follows 
    `python test_identification.py --outfile <output_file.json> --outtests  <tests_output_file.json> <repo_path>`
    #### Arguments:
    + `<repo_path>` (required): Path to the repository to analyze.
    + `<output_file.json>` (optional): File to save the extracted classes and methods data. Defaults to <repo_name>.json.
    + `<tests_output_file.json>` (optional): File to save the extracted tests data. Defaults to <repo_name>.tests.json.
    
    #### Output examples:
    + `output_file.json`
        ```json
        "build-scripts/add_stig_references.py": {
            "__modulename__": ".add_stig_references",
            "__global__": {
                "parse_args": {
                    "line": 9,
                    "line_end": 15,
                    "indent": 0
                },
                "main": {
                    "line": 18,
                    "line_end": 22,
                    "indent": 0
                }
            }
        }

    + `tests_output_file.json`
        ```json
        [
            {
                "file_path": "tests/unit/ssg-module/test_ansible.py",
                "test_framework": "pytest",
                "test_imports": {
                    "ssg.ansible": "ssg.ansible",
                    "min_ansible_version": "ssg.constants.min_ansible_version"
                },
                "test_methods": [
                    {
                        "method_name": "test_add_minimum_version",
                        "line": 16,
                        "line_end": 65,
                        "indent": 0,
                        "called_methods": [
                            "strings_equal_except_whitespaces",
                            "ssg.ansible.add_minimum_version"
                        ]
                    }
                ]
            },
        ```

### Find Focals  
This script, named find_focals.py, is designed to analyze  focal files, classes, and methods associated with test methods in the repository and establish the mapping. Below is how to run it:
- Run `find_focals.py` as follows 
    `python find_focals.py <infile> <intests> [-outfile <output_filename>]`
    #### Arguments:
    + `<infile>`: File containing implementation data.
    + `<intests>`: File containing test data.
    + [`-outfile <output_filename>`] (optional): Specify the output file name. If omitted, the output file will have the same name as `<infile>` but with `.focal.json` extension.

    #### Output examples:
    + `hash.focal.json`
        ```json
        {
            "tests/unit/test_main.py": {
                "focal_file": "gordon/main.py",
                "methods": {
                    "test_shutdown": {
                        "line": 32,
                        "line_end": 44,
                        "indent": 0,
                        "focal_class": null,
                        "focal_method": {
                            "line": 51,
                            "line_end": 65,
                            "indent": 0,
                            "name": "shutdown"
                        }
                    },

### Focal context  

This script is designed to extract and generate context for test methods and their corresponding focal methods in Python repositories, based on a JSON file containing focal information. By downloading get_context.py and the focal-data.zip files from pyMethods2Test, you can build the focal context as follows:
- Run `get_context.py` as
    `python get_context.py <json_file> [OPTIONS]`
    #### Arguments:
    + `<json_file>`: Path to the focal JSON file (must end with `.focal.json`). The json file should be in such a directory `user/repo/hash.focal.json`. The commit hash ensures that the focal context, including method line numbers, remains accurate for the specific commit, as developers may alter code during updates.

    Example:  `python get_context.py data/spotify/gordon/39d9a8408b617be46694df268d57f63daa9ab340.focal.json `

    #### Optional Arguments:
    + [--repos_dir]: Directory to store cloned repositories (default: repos).
    + [--output_file]: File to save the output JSON.
    + [--test_file]: Filter processing to a specific test file.
    + [--test_method]: Filter processing to a specific test method.

    #### Focal context examples:
    ```json
    {
    "pyscriptic/tests/measures_test.py": {
        "focal_file": "pyscriptic/measures.py",
        "methods": {
            "test_check_volume": {
                "line": 7,
                "line_end": 14,
                "indent": 4,
                "focal_class": null,
                "focal_method": {
                    "line": 27,
                    "line_end": 38,
                    "indent": 0,
                    "name": "check_volume"
                },
                "test_method": "def test_check_volume(self):\n    self.assertTrue(measures.check_volume('1:nanoliter'))\n    self.assertTrue(measures.check_volume('0.5:microliter'))\n    self.assertTrue(measures.check_volume('10:milliliter'))\n    self.assertFalse(measures.check_volume('a:nanoliter'))\n    self.assertFalse(measures.check_volume('1:femtoliter'))\n    self.assertFalse(measures.check_volume('1:liter'))",
                "focal_context": "def check_volume(volume):\n    \"\"\"\n    Checks that a volume has a correct quantity and allowed units.\n\n    Parameters\n    ----------\n    volume : str\n    \"\"\"\n    return _check_measurement(volume, ['nanoliter', 'microliter', 'milliliter'])\ndef _check_measurement(measurement, allowed_units): ...\ndef check_volume(volume): ...\ndef check_duration(duration): ...\ndef check_speed(speed): ...\ndef check_length(length): ...\ndef check_temperature(temperature): ...\ndef check_matter(matter): ...\ndef check_flowrate(flowrate): ...\ndef check_acceleration(acceleration): ...\n"
            }
        }
    },
The extracted data from the focal context can be organized and aligned to better train Large Language Models (LLMs) for automated test generation. Below is an example of the focal context that we have extracted and aligned from the repository spotify/gordon, available at the url `https://github.com/spotify/gordon.git`
 ```python
    class LogRelay():                                                             # focal class
    def _create_metric(self, metric_name, value, context, **kwargs):              # focal method
    context = context or {}
    return {’metric_name’: metric_name, ’value’: value, ’context’: context}
    def __init__(self, config): ...                                               # constructor
    def incr(self, metric_name, value=1, context=None, **kwargs): ...             # methods
    def timer(self, metric_name, context=None, **kwargs): ...
    def set(self, metric_name, value, context=None, **kwargs): ...
    def cleanup(self): ...
                                                                                  # class attributes (none)
    self.time_unit = config.get(’time_unit’, 1)                                   # instance attributes
    self.logger = LoggerAdapter(level)
    self.counters = collections.defaultdict(int)
