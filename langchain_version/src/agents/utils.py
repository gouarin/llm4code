import re


def read_prompt_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def extract_and_write_code(content, filename):
    # Regex to match ```python ... ```
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    code_blocks = pattern.findall(content)

    # Combine extracted code
    extracted_code = "\n\n".join(code_blocks)

    # Save to a Python file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(extracted_code)

    return extracted_code
