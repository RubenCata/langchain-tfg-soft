from enum import Enum
import tiktoken
import streamlit as st

#
# --- TOKEN TYPE ---
#
class TokenType(Enum):
    INPUT = 1
    OUTPUT = 2


# --- TOKEN COUNT FUNCTIONS ---

def posTratamiento(s: str):
    return s.replace("#OPENED_CURLY_BRACE#", "{").replace("#CLOSED_CURLY_BRACE#", "}")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        posTratamiento(text),
        disallowed_special=()
    )
    return len(tokens)

def add_tokens(text, type: TokenType):
    tokens: float = tiktoken_len(text)
    if type == TokenType.INPUT:
        st.session_state.question_in_tokens += tokens
        st.session_state.total_in_tokens += tokens
    else:
        st.session_state.question_out_tokens += tokens
        st.session_state.total_out_tokens += tokens

    st.session_state.question_cost += 0.002/1000.0 * tokens
    st.session_state.total_cost += 0.002/1000.0 * tokens