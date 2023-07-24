import os
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackManager, BaseCallbackHandler

from uuid import UUID
import time
from typing import (
    Any,
)
from langchain.schema import (
    LLMResult,
)

import app_langchain.tokens as tokens
import app_functions as app

#
# --- AZURE CHAT MODEL ---
#
def get_chat_model(streaming=True, temperature=0.0, handler=None):
    if streaming:
        return AzureChatOpenAI(
            openai_api_base = os.environ["OPENAI_API_BASE"],
            openai_api_version = os.environ["OPENAI_API_VERSION"],
            deployment_name = os.environ["DEPLOYMENT_NAME"],
            openai_api_key = os.environ["OPENAI_API_KEY"],
            openai_api_type = os.environ["OPENAI_API_TYPE"],
            temperature = temperature,
            streaming=streaming,
            callback_manager=BaseCallbackManager([handler]),
            request_timeout=6,
            verbose=True,
        )
    else:
        return AzureChatOpenAI(
            openai_api_base = os.environ["OPENAI_API_BASE"],
            openai_api_version = os.environ["OPENAI_API_VERSION"],
            deployment_name = os.environ["DEPLOYMENT_NAME"],
            openai_api_key = os.environ["OPENAI_API_KEY"],
            openai_api_type = os.environ["OPENAI_API_TYPE"],
            temperature = temperature,
            request_timeout=6,
            verbose=True,
        )


#
# --- FAKE STREAMING CALLBACK ---
#
class FakeStreamingCallbackHandlerClass(BaseCallbackHandler):
    """Fake CallbackHandler."""

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs: Any) -> Any:
        for p in prompts:
            tokens.add_tokens(text=p, type=tokens.TokenType.INPUT)

    def on_llm_new_token(self, token, **kwargs: Any):
        pass

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id = None, **kwargs: Any) -> Any:
        tokens.add_tokens(text=response.generations[0][0].message.content, type=tokens.TokenType.OUTPUT)


#
# --- STREAMING CALLBACK HANDLER ---
#
class MyStreamingCallbackHandlerClass(BaseCallbackHandler):
    """Custom CallbackHandler."""
    slow_down: bool = True
    widget = None
    incomplete_chat_model_answer: str = ""

    def set_slow_down(self, slow_down):
        self.slow_down = slow_down

    def set_widget(self, widget):
        self.widget = widget

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs: Any) -> Any:
        self.incomplete_chat_model_answer = ""
        for p in prompts:
            tokens.add_tokens(text=p, type=tokens.TokenType.INPUT)

    def on_llm_new_token(self, token, **kwargs: Any):
        # Do something with the new token
        self.incomplete_chat_model_answer = self.incomplete_chat_model_answer + token
        self.widget.chat_message("assistant").write(
            app.replace_urls_with_fqdn_and_lastpath(self.incomplete_chat_model_answer)
            )
        if self.slow_down:
            time.sleep(15/1000)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id = None, **kwargs: Any) -> Any:
        tokens.add_tokens(text=response.generations[0][0].message.content, type=tokens.TokenType.OUTPUT)