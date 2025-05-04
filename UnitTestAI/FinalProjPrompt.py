import os

from UnitTestAI.FinalProject import MODEL_INPUT_DIR, MODEL_OUTPUT_DIR, transformer_model, prompt_model, tokenizer

# TRANSFORMER ED

# If input file given, use that; otherwise, read input files from special prompt directory

# Collect all relative file paths from MODEL_INPUT_DIR
file_paths = [
    os.path.relpath(os.path.join(dirpath, filename), MODEL_INPUT_DIR)
    for dirpath, _, filenames in os.walk(MODEL_INPUT_DIR)
    for filename in filenames
]

# For each one generate unit tests with both frameworks
framework_options = ['pytest', 'unittest']
for fw in framework_options:
    for fp in file_paths:
        in_path = os.path.join(MODEL_INPUT_DIR, fp)
        out_path = os.path.join(MODEL_OUTPUT_DIR, f"{fw}_{fp}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        prompt_model(transformer_model, tokenizer, fw, in_path, out_path)
