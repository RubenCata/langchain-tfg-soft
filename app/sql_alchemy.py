from datetime import datetime
from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, ForeignKey, DateTime, UnicodeText, create_engine, desc, func
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    joinedload,
    Session,
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import streamlit as st
import os
from uuid import uuid4

import vars


#
# --- SQL ALCHEMY ORM BUILD ---
#
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
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


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    filename = Column(UnicodeText)
    title = Column(UnicodeText)
    upload_date = Column(DateTime, default=datetime.now)
    selected = Column(Boolean, default=True)
    md5 = Column(String(32))


#
# --- SESSION STATE MEMORY ---
#
def create_memory(recreate: bool = False):
    # Create a ConversationEntityMemory object if not already created
    if 'memory' not in st.session_state or recreate:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="history",
            k=vars.MEMORY_K,
            ai_prefix=vars.AI_NAME,
            human_prefix=vars.username,
            # return_messages=True,
        )

def clear_history():
    if len(st.session_state.memory.chat_memory.messages) > 1:
        del st.session_state.memory
        del st.session_state.sql_conversation_id
        st.session_state.total_in_tokens = 0
        st.session_state.total_out_tokens = 0
        st.session_state.total_cost = 0


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


def create_conversation():
    with get_sql_session() as session:
        session.begin()
        sql_conversation = Conversation(
            id=str(uuid4()),
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


def save_interaction(conver_name, query, response, config, ai_feedback, chunks, deixis_query):
    with get_sql_session() as session:
        session.begin()

        chunks_list = []
        for chunk in chunks:
            if "published" in chunk["metadata"]:
                chunk["metadata"].pop("published")
            chunks_list.append({
                "metadata": chunk["metadata"],
                "score": chunk["score"],
            })

        try:
            sql_conversation = session.query(Conversation).filter_by(id=st.session_state.sql_conversation_id).first()
            sql_conversation.name = conver_name

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

            sql_conversation.interactions.append(interaction)
            session.add(sql_conversation)
            session.commit()
        except:
            session.rollback()
            print("Interaccion NO GUARDADA", interaction.id)
            st.warning({"chunks": chunks_list})
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


def get_conversations():
    with get_sql_session() as session:
        try:
            conversations = session.query(Conversation).filter(
                Conversation.active == True,
                Conversation.name.is_not(None),
                Conversation.interactions.any(),
                ).outerjoin(Conversation.interactions).group_by(Conversation.id).order_by(desc(func.max(Interaction.timestamp))).options(joinedload(Conversation.interactions)).all()
        except:
            session.rollback()
            print("Could not load conversations")
            raise
        else:
            return conversations


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
                st.session_state.total_in_tokens = 0
                st.session_state.total_out_tokens = 0
            create_memory(recreate=True)
            st.session_state.memory.chat_memory.add_ai_message(f"Hola {vars.username}, ¿en qué puedo ayudarte?")
            for inter in interactions:
                st.session_state.memory.chat_memory.add_user_message(inter.question)
                st.session_state.memory.chat_memory.add_ai_message(inter.response)
                st.session_state.total_cost += inter.cost
                st.session_state.total_out_tokens += inter.tokens
                st.session_state.sql_interaction_id = interactions[len(interactions)-1].id

def get_conversation(conversation_id):
    with get_sql_session() as session:
        try:
            sql_conversation = session.query(Conversation).filter_by(id=conversation_id).first()
        except:
            session.rollback()
            raise
        return sql_conversation

def edit_conversation_name():
    conversation_name = st.session_state.conversation_name
    if conversation_name != "" and "edit_conversation_id" in st.session_state:
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
                print("Could not delete conversation: ", st.session_state.delete_conversation_id)
                raise
            else:
                clear_history()
    del st.session_state.delete_conversation_id
    del st.session_state.delete_conversation_name


def exists_document_md5(md5):
    with get_sql_session() as session:
        try:
            document = session.query(Document).filter(Document.md5 == md5).first()
        except:
            session.rollback()
            print("Could not check if exists a document with MD5: ", md5)
            raise
    return document is not None


def save_document(id, filename, title, md5):
    with get_sql_session() as session:
        session.begin()
        sql_document = Document(
            id=id,
            filename=filename,
            title=title,
            md5=md5,
        )
        try:
            session.add(sql_document)
            session.commit()
        except:
            session.rollback()
            raise


def get_documents():
    with get_sql_session() as session:
        try:
            documents = session.query(Document).order_by(Document.upload_date.desc()).all()
        except:
            session.rollback()
            print("Could not load documents")
            raise
        else:
            return documents


def get_selected_documents():
    with get_sql_session() as session:
        try:
            selected_documents = session.query(Document).filter(
                Document.selected == True,
                ).all()
        except:
            session.rollback()
            print("Could not load selected documents")
            raise
        else:
            return [doc.md5 for doc in selected_documents]

def get_document_title(document_md5):
    with get_sql_session() as session:
        try:
            sql_document = session.query(Document).filter(
                Document.md5 == document_md5,
                ).first()
        except:
            session.rollback()
            print("Could not find document: ", document_md5)
            raise
        else:
            return sql_document.title

def update_select_doc(doc_id, selected):
    with get_sql_session() as session:
        try:
            sql_document = session.query(Document).filter_by(id=doc_id).first()
            sql_document.selected = selected
            session.add(sql_document)
            session.commit()
        except:
            session.rollback()
            print("Could not select document in database: ", doc_id)
            raise
        else:
            if selected:
                st.session_state.selected_documents.add(doc_id)
            else:
                st.session_state.selected_documents.discard(doc_id)

def edit_document_title():
    document_name = st.session_state.document_title
    if document_name != "" and "edit_document_id" in st.session_state:
        with get_sql_session() as session:
            try:
                sql_document = session.query(Document).filter_by(id=st.session_state.edit_document_id).first()
                sql_document.title = document_name
                session.add(sql_document)
                session.commit()
            except:
                session.rollback()
                print("Could not edit document's name: ", st.session_state.edit_document_id)
                raise
        del st.session_state.edit_document_id
        del st.session_state.edit_document_title

def delete_document(bool):
    if bool:
        with get_sql_session() as session:
            try:
                sql_document = session.query(Document).filter_by(id=st.session_state.delete_document_id).first()
                md5_documents = session.query(Document).filter_by(md5=sql_document.md5).count()
                session.delete(sql_document)
                session.commit()
            except:
                session.rollback()
                print("Could not delete conversation: ", st.session_state.delete_document_id)
                raise
            else:
                if md5_documents == 1:
                    vars.index.delete(
                        filter={
                            "document_md5": {"$eq": sql_document.md5}
                        },
                        namespace="uploaded-documents"
                    )
    del st.session_state.delete_document_id
    del st.session_state.delete_document_title