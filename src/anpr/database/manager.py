import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator

from .models import Base, Detection


class DatabaseManager:
    """
    Database manager class that handles database connection initialization,
    table creation with proper constraints, session management, and methods
    to initialize the database (create tables).
    """
    
    def __init__(self, database_url: str = None, echo: bool = False):
        """
        Initialize the database manager.
        
        Args:
            database_url: URL for the database connection (defaults to SQLite in data folder)
            echo: Whether to echo SQL statements for debugging
        """
        if database_url is None:
            # Default to SQLite database in a data folder
            db_dir = os.path.join(os.getcwd(), 'data')
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{os.path.join(db_dir, 'anpr.db')}"
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def init_database(self):
        """
        Initialize the database by creating all tables with proper constraints.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            print(f"Database initialized successfully at {self.database_url}")
        except SQLAlchemyError as e:
            print(f"Error initializing database: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for getting a database session.
        
        Yields:
            Session: An SQLAlchemy session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """
        Get a direct database session (caller is responsible for closing).
        
        Returns:
            Session: An SQLAlchemy session object
        """
        return self.SessionLocal()
    
    def close(self):
        """
        Close the database engine.
        """
        self.engine.dispose()
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            with self.get_session() as session:
                # Try to query something simple to test connection
                session.query(Detection).limit(1).all()
                return True
        except Exception:
            return False
