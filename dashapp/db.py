import os

from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL)
Session = sessionmaker(engine)
Base = declarative_base()


image_label_table = Table(
    "image_labels",
    Base.metadata,
    Column("image_id", ForeignKey("images.id")),
    Column("label_id", ForeignKey("labels.id")),
)


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    name = Column(String(254), unique=True, nullable=False)
    labels = relationship("Label", secondary=image_label_table)
