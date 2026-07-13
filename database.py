from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# SQLAlchemy needs its own driver in the URL (psycopg2), separate from
# the psycopg_pool connection used by LangGraph's PostgresSaver/PostgresStore.
# Just swap the scheme — same DB_URI, same Supabase Postgres instance.
RAW_DB_URI = os.getenv('DB_URI')
SQLALCHEMY_DB_URI = RAW_DB_URI.replace('postgresql://', 'postgresql+psycopg2://', 1)

engine = create_engine(
    SQLALCHEMY_DB_URI,
    pool_pre_ping=True,   # avoids stale-connection errors after idle periods
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()