from sqlalchemy import create_engine

db_uri = "postgresql+psycopg2://postgres:1234@localhost:5432/chatdb"
engine = create_engine(db_uri)
print(engine)