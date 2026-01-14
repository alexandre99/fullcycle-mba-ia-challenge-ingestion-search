import os_util
from dotenv import load_dotenv
import time
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
from factory import get_embeddings

PDF_PATH = os_util.get_env("PDF_PATH")
DATABASE_URL = os_util.get_env("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os_util.get_env("PG_VECTOR_COLLECTION_NAME")


def load_pdf(file_path: str):
    print(f"Loading PDF from {file_path}...")
    return PyPDFLoader(file_path).load()


def split_documents(documents: list[Document]):
    print(f"Splitting {len(documents)} documents...")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False
    ).split_documents(documents)


def clean_documents(documents: list[Document]):
    """
    Removes empty documents and cleans null metadata.
    This reduces the chance of empty chunks reaching the hashing stage.
    """
    print(f"Cleaning and enriching {len(documents)} documents...")

    cleaned = []
    for d in documents:
        if not d.page_content or not d.page_content.strip():
            # Document is truly empty — skip it
            continue

        metadata = {k: v for k, v in d.metadata.items() if v not in ("", None)}

        cleaned.append(
            Document(
                page_content=d.page_content.strip(),
                metadata=metadata
            )
        )

    print(f"Remaining valid documents after cleaning: {len(cleaned)}")
    return cleaned


def get_vector_store(embeddings):
    return PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def doc_id_from_content(doc: Document) -> str:
    """
    Generates a deterministic ID with a hash based on the chunk content
    and important metadata (e.g., page number).

    Raises an exception if the content is empty after normalization.
    """
    if not isinstance(doc, Document):
        raise TypeError("doc must be a langchain_core.documents.Document")

    text = (doc.page_content or "").strip()

    if not text:
        raise ValueError("Cannot generate ID: document content is empty")

    page = str(doc.metadata.get("page", ""))

    normalized = f"{text}\nPAGE:{page}"

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def ingest_to_store(store: PGVector, documents: list[Document], batch_size: int = 50):
    print(f"Ingesting {len(documents)} documents into PGVector in batches of {batch_size}...")

    ids = [doc_id_from_content(d) for d in documents]

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]

        print(f"Sending batch {i // batch_size + 1} with {len(batch)} docs...")

        store.add_documents(
            documents=batch,
            ids=batch_ids,
            upsert=True  # ensures idempotency without duplication
        )

        # pause to avoid RPM limit
        time.sleep(1)


def ingest_pdf():
    embeddings = get_embeddings()
    print(f"Using embeddings: {type(embeddings).__name__}")

    documents = load_pdf(PDF_PATH)
    splits = split_documents(documents)

    if not splits:
        print("No documents found to ingest.")
        return

    enriched = clean_documents(splits)

    if not enriched:
        print("All documents were empty after cleaning — nothing to ingest.")
        return

    store = get_vector_store(embeddings)

    ingest_to_store(store, enriched)

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pdf()