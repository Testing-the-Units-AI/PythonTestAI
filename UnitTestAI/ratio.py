import json

py_count = 0
un_count = 0

with open('./data/all.jsonl', mode="r") as data_file:
    for i, line in enumerate(data_file):
        jsonl = json.loads(line)
        framework = jsonl.get("framework")

        if framework == "pytest":
            py_count += 1

        else:
            un_count += 1

print('py: ', py_count, '  un: ', un_count)