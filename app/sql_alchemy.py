from datetime import datetime
from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, ForeignKey, DateTime, UnicodeText, create_engine
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    Session,
)
import streamlit as st
import vars
import os
from uuid import uuid4

import app_functions as app
import app_langchain.chains as chains

#
# --- SQL ALCHEMY ORM BUILD ---
#
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    username = Column(String(100))
    name = Column(UnicodeText)
    creation_date = Column(DateTime, default=datetime.now)
    active = Column(Boolean)
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
    ai_feedback = Column(Boolean)
    deixis_query = Column(UnicodeText)
    chunks = Column(JSON)
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


def save_conversation():
    if 'sql_conversation_id' not in st.session_state:
        with get_sql_session() as session:
            session.begin()
            sql_conversation = Conversation(
                id=str(uuid4()),
                username=vars.username,
                active=True,
            )
            try:
                session.add(sql_conversation)
                session.commit()
            except:
                session.rollback()
                raise
            else:
                st.session_state.sql_conversation_id = sql_conversation.id


def save_interaction(query, response, config, ai_feedback, chunks, deixis_query):
    with get_sql_session() as session:
        session.begin()

        chunks_list = []
        for i in range(0, len(chunks)):
            chunks_list.append({
                "metadata": chunks[i]["metadata"],
                "score": chunks[i]["score"],
            })

        interaction = Interaction(
                id=str(uuid4()),
                question=query,
                response=response,
                config=config,
                tokens=st.session_state.question_in_tokens + st.session_state.question_out_tokens,
                cost=st.session_state.question_cost,
                ai_feedback=ai_feedback,
                deixis_query=deixis_query,
                chunks={"chunks": chunks_list},
            )
        st.session_state.sql_interaction_id = interaction.id
        try:
            sql_conversation = session.query(Conversation).filter_by(id=st.session_state.sql_conversation_id).first()

            if sql_conversation.name == None:
                conversation_name = chains.naming_chain(query, response).content
                sql_conversation.name = conversation_name

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


def get_user_conversations(container):
    with get_sql_session() as session:
        try:
            user_conversations = session.query(Conversation).filter(
                Conversation.username == vars.username,
                Conversation.active == True,
                Conversation.name.is_not(None),
                ).all()
            clean_conversations = []
            for conver in user_conversations:
                if len(conver.interactions) > 0:
                    clean_conversations.append(conver)
            sorted_user_conversations = sorted(clean_conversations, key=lambda x: x.interactions[0].timestamp, reverse=True)
        except:
            session.rollback()
            print("Could not load user conversations: ", vars.username)
            raise
        else:
            app.user_conversations_display(sorted_user_conversations, container)


def load_conversation(id):
    with get_sql_session() as session:
        try:
            interactions = session.query(Interaction).filter_by(conversation_id=id).order_by(Interaction.timestamp).all()
        except:
            session.rollback()
            print("Could not load conversation: ", id)
            raise
        else:
            st.session_state.sql_conversation_id = id
            if "total_cost" in st.session_state:
                st.session_state.total_cost = 0
            app.create_memory(recreate=True)
            st.session_state.memory.chat_memory.add_ai_message(f"Hola {vars.username}, ¿en qué puedo ayudarte?")
            for inter in interactions:
                st.session_state.memory.chat_memory.add_user_message(inter.question)
                st.session_state.memory.chat_memory.add_ai_message(inter.response)
                st.session_state.total_cost += inter.cost


def edit_conversation_name(conversation_name):
    if conversation_name != "":
        with get_sql_session() as session:
                try:
                    sql_conversation = session.query(Conversation).filter_by(id=st.session_state.edit_conversation_id).first()
                    sql_conversation.name = conversation_name
                    session.add(sql_conversation)
                    session.commit()
                except:
                    session.rollback()
                    print("Could not edit conversation's name: ", st.session_state.edit_conversation_id)
                    raise
        del st.session_state.edit_conversation_id
        del st.session_state.edit_conversation_name


def delete_conversation(bool):
    if bool:
        with get_sql_session() as session:
            try:
                sql_conversation = session.query(Conversation).filter_by(id=st.session_state.delete_conversation_id).first()
                sql_conversation.active = False
                session.add(sql_conversation)
                session.commit()
            except:
                session.rollback()
                print("Could not delete conversation: ", st.session_state.edit_conversation_id)
                raise
            else:
                app.clear_history()
    del st.session_state.delete_conversation_id
    del st.session_state.delete_conversation_name