"""
SQLite-based memory backend for append-only chat history.
Stores messages per session with optional embeddings for retrieval.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings
from core.exceptions import MemoryError
from core.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Message(Base):
    """Message model for chat history."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_metadata = Column(Text, nullable=True)  # JSON string
    embedding = Column(Text, nullable=True)  # JSON array


class ChatMessage(BaseModel):
    """Pydantic model for chat messages."""

    id: int | None = Field(default=None, description="Message ID")
    session_id: str = Field(description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str = Field(description="Message role (user/assistant)")
    content: str = Field(description="Message content")
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None)

    class Config:
        """Pydantic config."""

        from_attributes = True


class SQLMemory:
    """
    SQLite-based append-only memory backend.
    Provides persistent chat history storage per session.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize SQL memory backend.

        Args:
            db_path: Path to SQLite database file (uses config default if None)
        """
        settings = get_settings()
        self.db_path = db_path or settings.sqlite_db_path

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

        logger.info("SQL memory initialized", db_path=self.db_path)

    def add_message(self, message: ChatMessage) -> ChatMessage:
        """
        Add a message to chat history (append-only).

        Args:
            message: ChatMessage to store

        Returns:
            ChatMessage with assigned ID

        Raises:
            MemoryError: If storage fails
        """
        try:
            with self.SessionLocal() as session:
                # Convert to ORM model
                db_message = Message(
                    session_id=message.session_id,
                    timestamp=message.timestamp,
                    role=message.role,
                    content=message.content,
                    message_metadata=json.dumps(message.metadata) if message.metadata else None,
                    embedding=json.dumps(message.embedding) if message.embedding else None,
                )

                session.add(db_message)
                session.commit()
                session.refresh(db_message)

                # Convert back to Pydantic
                result = self._to_chat_message(db_message)

                logger.debug(
                    "Message stored",
                    session_id=message.session_id,
                    role=message.role,
                    message_id=result.id,
                )

                return result

        except Exception as e:
            logger.error("Failed to store message", error=str(e))
            raise MemoryError(f"Failed to store message: {str(e)}") from e

    def get_session_history(
        self, session_id: str, limit: int | None = None
    ) -> list[ChatMessage]:
        """
        Retrieve chat history for a session.

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages (most recent)

        Returns:
            List of ChatMessage in chronological order

        Raises:
            MemoryError: If retrieval fails
        """
        try:
            with self.SessionLocal() as session:
                query = session.query(Message).filter(Message.session_id == session_id)

                # Apply limit (get most recent N messages)
                if limit:
                    query = query.order_by(Message.timestamp.desc()).limit(limit)
                    messages = query.all()
                    # Reverse to get chronological order
                    messages = list(reversed(messages))
                else:
                    messages = query.order_by(Message.timestamp.asc()).all()

                result = [self._to_chat_message(msg) for msg in messages]

                logger.debug(
                    "Session history retrieved", session_id=session_id, message_count=len(result)
                )

                return result

        except Exception as e:
            logger.error("Failed to retrieve session history", error=str(e))
            raise MemoryError(f"Failed to retrieve session history: {str(e)}") from e

    def get_all_sessions(self) -> list[str]:
        """
        Get list of all session IDs.

        Returns:
            List of unique session identifiers
        """
        try:
            with self.SessionLocal() as session:
                result = (
                    session.query(Message.session_id).distinct().order_by(Message.session_id).all()
                )
                sessions = [row[0] for row in result]

                logger.debug("All sessions retrieved", session_count=len(sessions))
                return sessions

        except Exception as e:
            logger.error("Failed to retrieve sessions", error=str(e))
            raise MemoryError(f"Failed to retrieve sessions: {str(e)}") from e

    def search_messages(self, query: str, session_id: str | None = None) -> list[ChatMessage]:
        """
        Simple text search across messages.

        Args:
            query: Search query string
            session_id: Optional session filter

        Returns:
            List of matching ChatMessage

        Raises:
            MemoryError: If search fails
        """
        try:
            with self.SessionLocal() as session:
                db_query = session.query(Message).filter(Message.content.contains(query))

                if session_id:
                    db_query = db_query.filter(Message.session_id == session_id)

                messages = db_query.order_by(Message.timestamp.desc()).limit(50).all()

                result = [self._to_chat_message(msg) for msg in messages]

                logger.debug("Messages searched", query=query[:50], result_count=len(result))
                return result

        except Exception as e:
            logger.error("Failed to search messages", error=str(e))
            raise MemoryError(f"Failed to search messages: {str(e)}") from e

    def clear_session(self, session_id: str) -> int:
        """
        Clear all messages for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of messages deleted

        Raises:
            MemoryError: If deletion fails
        """
        try:
            with self.SessionLocal() as session:
                count = session.query(Message).filter(Message.session_id == session_id).delete()
                session.commit()

                logger.info("Session cleared", session_id=session_id, deleted_count=count)
                return count

        except Exception as e:
            logger.error("Failed to clear session", error=str(e))
            raise MemoryError(f"Failed to clear session: {str(e)}") from e

    def _to_chat_message(self, db_message: Message) -> ChatMessage:
        """Convert ORM model to Pydantic model."""
        return ChatMessage(
            id=db_message.id,
            session_id=db_message.session_id,
            timestamp=db_message.timestamp,
            role=db_message.role,
            content=db_message.content,
            metadata=json.loads(db_message.message_metadata) if db_message.message_metadata else {},
            embedding=json.loads(db_message.embedding) if db_message.embedding else None,
        )

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        logger.info("SQL memory closed")
