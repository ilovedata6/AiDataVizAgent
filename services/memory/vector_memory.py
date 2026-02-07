"""
Pluggable vector memory adapter for semantic search.
Supports ChromaDB and future vector stores.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class VectorDocument(BaseModel):
    """Document with embedding for vector store."""

    id: str = Field(description="Unique document ID")
    content: str = Field(description="Document content")
    embedding: list[float] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """Search result from vector store."""

    document: VectorDocument = Field(description="Matched document")
    score: float = Field(description="Similarity score")


class VectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_documents(self, documents: list[VectorDocument]) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_by_id(self, doc_id: str) -> None:
        """Delete document by ID."""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store implementation.
    Provides semantic search over chat history and datasets.
    """

    def __init__(self, collection_name: str = "chat_memory") -> None:
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection to use
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            settings = get_settings()

            # Initialize ChromaDB client
            self.client = chromadb.Client(
                ChromaSettings(
                    persist_directory=settings.vector_store_path,
                    anonymized_telemetry=False,
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                "ChromaDB vector store initialized",
                collection=collection_name,
                path=settings.vector_store_path,
            )

        except ImportError:
            logger.warning("ChromaDB not installed, vector store unavailable")
            raise ImportError("chromadb package required for vector store")

    def add_documents(self, documents: list[VectorDocument]) -> None:
        """
        Add documents to ChromaDB.

        Args:
            documents: List of VectorDocument to add
        """
        if not documents:
            return

        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # ChromaDB will generate embeddings automatically if not provided
        embeddings = [doc.embedding for doc in documents if doc.embedding]

        if embeddings and len(embeddings) == len(documents):
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        else:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

        logger.debug(f"Added {len(documents)} documents to vector store")

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of VectorSearchResult ordered by relevance
        """
        results = self.collection.query(query_texts=[query], n_results=limit)

        # Convert to VectorSearchResult
        search_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = VectorDocument(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                score = 1.0 - results["distances"][0][i] if results["distances"] else 0.0

                search_results.append(VectorSearchResult(document=doc, score=score))

        logger.debug(f"Vector search returned {len(search_results)} results")
        return search_results

    def delete_by_id(self, doc_id: str) -> None:
        """
        Delete document by ID.

        Args:
            doc_id: Document ID to delete
        """
        self.collection.delete(ids=[doc_id])
        logger.debug(f"Deleted document {doc_id} from vector store")


class VectorMemory:
    """
    High-level vector memory interface.
    Wraps vector store for chat history and dataset metadata.
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        """
        Initialize vector memory.

        Args:
            store: Optional VectorStore implementation (creates ChromaDB if None)
        """
        if store is None:
            try:
                self.store = ChromaVectorStore()
            except ImportError:
                logger.warning("Vector store unavailable, using None")
                self.store = None
        else:
            self.store = store

        logger.info("Vector memory initialized", store_type=type(self.store).__name__)

    def add_chat_message(self, session_id: str, message_id: int, content: str) -> None:
        """
        Add chat message to vector store for semantic search.

        Args:
            session_id: Session identifier
            message_id: Message ID
            content: Message content
        """
        if self.store is None:
            return

        doc = VectorDocument(
            id=f"{session_id}_{message_id}",
            content=content,
            metadata={"session_id": session_id, "message_id": message_id, "type": "chat"},
        )

        self.store.add_documents([doc])

    def search_similar_messages(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        """
        Search for similar chat messages.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of similar messages
        """
        if self.store is None:
            return []

        return self.store.search(query, limit)
