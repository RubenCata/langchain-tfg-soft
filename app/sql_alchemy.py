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
        for chunk in chunks:
            chunks_list.append({
                "metadata": {
                    'entry_id': chunk["metadata"]['entry_id'],
                    'title': chunk["metadata"]['title'],
                    'authors': chunk["metadata"]['authors'],
                    'primary_category': chunk["metadata"]['primary_category'],
                    'categories': chunk["metadata"]['categories'],
                    'links': chunk["metadata"]['links'],
                    'text': chunk["metadata"]['text'],
                    },
                "score": chunk["score"],
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


def get_conversations(container):
    with get_sql_session() as session:
        try:
            conversations = session.query(Conversation).filter(
                Conversation.active == True,
                Conversation.name.is_not(None),
                ).all()
            clean_conversations = []
            for conver in conversations:
                if len(conver.interactions) > 0:
                    clean_conversations.append(conver)
            sorted_conversations = sorted(clean_conversations, key=lambda x: x.interactions[0].timestamp, reverse=True)
        except:
            session.rollback()
            print("Could not load conversations")
            raise
        else:
            app.conversations_display(sorted_conversations, container)


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
            app.create_memory(recreate=True)
            st.session_state.memory.chat_memory.add_ai_message(f"Hola {vars.username}, ¿en qué puedo ayudarte?")
            for inter in interactions:
                st.session_state.memory.chat_memory.add_user_message(inter.question)
                st.session_state.memory.chat_memory.add_ai_message(inter.response)
                st.session_state.total_cost += inter.cost
                st.session_state.total_out_tokens += inter.tokens
                st.session_state.sql_interaction_id = interactions[len(interactions)-1].id


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
                app.clear_history()
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


def get_documents(index, container):
    with get_sql_session() as session:
        try:
            documents = session.query(Document).order_by(Document.upload_date.desc()).all()
        except:
            session.rollback()
            print("Could not load documents")
            raise
        else:
            app.documents_display(documents, index, container)


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

def delete_document(index, bool):
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
                    index.delete(
                        filter={
                            "document_md5": {"$eq": sql_document.md5}
                        },
                        namespace="uploaded-documents"
                    )
    del st.session_state.delete_document_id
    del st.session_state.delete_document_title