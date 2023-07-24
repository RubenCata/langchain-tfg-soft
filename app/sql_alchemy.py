from datetime import datetime
from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, ForeignKey, DateTime, UnicodeText, create_engine
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    Session,
)
import streamlit as st
import os
from uuid import uuid4

#
# --- SQL ALCHEMY ORM BUILD ---
#
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    username = Column(String(100))
    creation_date = Column(DateTime, default=datetime.now)
    interactions = relationship("Interaction", back_populates="conversation", cascade="all, delete", collection_class=list)

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"))
    timestamp = Column(DateTime, default=datetime.now)
    question = Column(UnicodeText)
    response = Column(UnicodeText)
    config = Column(JSON)
    tokens = Column(Integer)
    cost = Column(Float)
    feedback = Column(Boolean)
    conversation = relationship("Conversation", back_populates="interactions")


#
# --- SQL ALCHEMY FUNCTIONS ---
#
def create_database():
    tmp_engine = create_engine(os.environ["DB_CONNECTION_STR"])
    Base.metadata.create_all(tmp_engine)


def get_sql_session():
    engine = create_engine(
        os.environ["DB_CONNECTION_STR"]
    )

    return Session(engine)


def save_conversation(username):
    if 'sql_conversation_id' not in st.session_state:
        with get_sql_session() as session:
            session.begin()
            sql_conversation = Conversation(id=str(uuid4()), username=username)
            try:
                session.add(sql_conversation)
                session.commit()
            except:
                session.rollback()
                raise
            else:
                st.session_state.sql_conversation_id = sql_conversation.id


def save_interaction(query, response, config):
    with get_sql_session() as session:
        session.begin()
        interaction = Interaction(
                id=str(uuid4()),
                question=query,
                response=response,
                config=config,
                tokens=st.session_state.question_in_tokens + st.session_state.question_out_tokens,
                cost=st.session_state.question_cost,
            )
        st.session_state.sql_interaction_id = interaction.id
        try:
            sql_conversation = session.query(Conversation).filter_by(id=st.session_state.sql_conversation_id).first()

            sql_conversation.interactions.append(interaction)
            session.add(sql_conversation)
            session.commit()
        except:
            session.rollback()
            print("Interaccion NO GUARDADA", interaction.id)
            raise


def save_feedback(feedback: bool):
    with get_sql_session() as session:
        session.begin()
        try:
            sql_interaction = session.query(Interaction).filter_by(id=st.session_state.sql_interaction_id).first()
            sql_interaction.feedback = feedback
            session.add(sql_interaction)
            session.commit()
        except:
            session.rollback()
            print("Feedback NO GUARDADO")
            raise