from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup
import markdownify


def bs4_extractor(html: str) -> str:
    soup = Soup(html, "html.parser")
    for script in soup(["script", "style", "nav"]):
        script.decompose()
    return markdownify.markdownify(str(soup), heading_style="ATX")


def extract_content(urls, max_depth=1):
    docs = [
        RecursiveUrlLoader(url, extractor=bs4_extractor, max_depth=max_depth).load()
        for url in urls
    ]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list
