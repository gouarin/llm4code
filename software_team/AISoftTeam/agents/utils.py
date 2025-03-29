import re


def extract_code_blocks(markdown_text):
    """
    Extract code blocks from markdown text.
    Returns a list of tuples (language, code_content)
    """
    # Pattern to match code blocks with language specification
    pattern = r"<root>(.*?)</root>"
    # Find all matches with re.DOTALL to match across multiple lines
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    return len(matches), "<root>\n" + "".join(matches) + "</root>\n"
