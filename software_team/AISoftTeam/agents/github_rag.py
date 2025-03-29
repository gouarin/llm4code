import os
import tempfile
import shutil
from pathlib import Path
from git import Repo
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import concurrent.futures


OLLAMA_BASE_URL = "http://localhost:8888"


class GitHubRAG:
    def __init__(self, persist_dir="github_rag_storage"):
        """
        Initialize the RAG system for GitHub repositories

        Args:
            persist_dir (str): Directory to save the index
        """
        self.persist_dir = persist_dir
        self.exclusions = [
            "node_modules",
            ".git",
            "__pycache__",
            "venv",
            ".env",
            "build",
            "dist",
            ".idea",
            ".vscode",
            # "examples",
            # "example",
            # "tests",
            # "test",
        ]
        self.extensions = [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".md",
            ".txt",
            ".yml",
            ".yaml",
            ".json",
            ".html",
            ".css",
        ]
        self.processed_repos = []

        # Initialize LlamaIndex contexts
        llm = Ollama(
            model="deepseek-r1:7b", base_url=OLLAMA_BASE_URL, request_timeout=300.0
        )
        embed_model = OllamaEmbedding(
            model_name="mxbai-embed-large:latest",
            base_url=OLLAMA_BASE_URL,
            request_timeout=300.0,
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

        # Load existing index if available
        if os.path.exists(persist_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                self.index = load_index_from_storage(storage_context)
                print(f"Index loaded from {persist_dir}")

                # Load the list of processed repositories
                repo_list_path = os.path.join(persist_dir, "processed_repos.txt")
                if os.path.exists(repo_list_path):
                    with open(repo_list_path, "r") as f:
                        self.processed_repos = [line.strip() for line in f.readlines()]
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
        else:
            self.index = None
            os.makedirs(persist_dir, exist_ok=True)

    def _is_valid_file(self, filepath):
        """
        Check if a file should be included

        Args:
            filepath (str): File path

        Returns:
            bool: True if the file should be included
        """
        # Check exclusions
        for exclusion in self.exclusions:
            if exclusion in filepath:
                return False

        # Check extensions
        _, ext = os.path.splitext(filepath)
        if ext not in self.extensions:
            return False

        return True

    def process_repository(self, repo_url, branch="main"):
        """
        Process a GitHub repository

        Args:
            repo_url (str): GitHub repository URL
            branch (str): Branch to use

        Returns:
            list: Processed documents
        """
        # Check if repository has already been processed
        if repo_url in self.processed_repos:
            print(f"Repository {repo_url} already processed")
            return []

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Cloning repository {repo_url}...")
            Repo.clone_from(repo_url, temp_dir)  # , branch=branch)

            # Load files with FlatReader
            reader = FlatReader()
            all_documents = []

            nfiles = 0
            for root, dirs, files in os.walk(temp_dir):

                # Ignore excluded directories
                dirs[:] = [d for d in dirs if d not in self.exclusions]

                for file in files:
                    if nfiles > 40:
                        break
                    filepath = os.path.join(root, file)
                    if self._is_valid_file(filepath):
                        try:
                            print(f"Loading {filepath}")
                            documents = reader.load_data(Path(filepath))
                            print("Documents: ", documents)
                            for doc in documents:
                                # Add metadata
                                doc.metadata["repo_url"] = repo_url
                                doc.metadata["filepath"] = os.path.relpath(
                                    filepath, temp_dir
                                )
                            all_documents.extend(documents)
                            nfiles += 1
                        except Exception as e:
                            print(f"Error loading {filepath}: {e}")

            print(f"Number of documents loaded from {repo_url}: {len(all_documents)}")

            # Mark repository as processed
            self.processed_repos.append(repo_url)
            with open(os.path.join(self.persist_dir, "processed_repos.txt"), "w") as f:
                for repo in self.processed_repos:
                    f.write(f"{repo}\n")

            return all_documents

        except Exception as e:
            print(f"Error processing {repo_url}: {e}")
            return []
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            pass

    def add_repositories(self, repo_urls, max_workers=2):
        """
        Add multiple repositories to the index

        Args:
            repo_urls (list): List of repository URLs
            max_workers (int): Number of parallel workers
        """
        all_documents = []

        # Parallel processing of repositories
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_repository, url): url for url in repo_urls
            }
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    print(f"Processing of {url} completed")
                except Exception as e:
                    print(f"Error processing {url}: {e}")

        if not all_documents:
            print("No new documents to add")
            return

        # Create or update the index
        if self.index is None:
            # Create a new index
            self.index = VectorStoreIndex.from_documents(all_documents)
        else:
            # Add to existing index
            for doc in all_documents:
                self.index.insert(doc)

        # Persist the index
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        print(f"Index saved to {self.persist_dir}")

    def query(self, question, top_k=5):
        """
        Query the index to get an answer

        Args:
            question (str): Question to ask
            top_k (int): Number of documents to retrieve

        Returns:
            dict: Result with answer and sources
        """
        if self.index is None:
            return {"answer": "The index has not been created yet", "sources": []}

        # Configure the retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )

        # Post-processor to improve relevance
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

        # Create the query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[postprocessor],
        )

        # Execute the query
        response = query_engine.query(question)

        # Extract and organize sources by repository
        sources_by_repo = {}
        for node in response.source_nodes:
            repo_url = node.metadata.get("repo_url", "Unknown")
            filepath = node.metadata.get("filepath", "Unknown")

            if repo_url not in sources_by_repo:
                sources_by_repo[repo_url] = []

            sources_by_repo[repo_url].append(filepath)

        return {"answer": str(response), "sources_by_repo": sources_by_repo}


# Usage example
if __name__ == "__main__":
    # List of GitHub repositories
    repos = ["https://github.com/user/repo1", "https://github.com/user/repo2"]

    # Create and initialize the RAG system
    rag = GitHubRAG(persist_dir="github_rag_index")

    # Add repositories
    rag.add_repositories(repos)

    # Ask questions about the repositories
    questions = [
        "What are the main features of these projects?",
        "How is authentication handled in these projects?",
        "What technologies are used in these projects?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")

        print("\nSources:")
        for repo, files in result["sources_by_repo"].items():
            print(f"  Repository: {repo}")
            for file in files:
                print(f"    - {file}")
        print("-" * 50)
