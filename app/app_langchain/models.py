import os
import re
from urllib.parse import urlparse

from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackManager, BaseCallbackHandler
from langchain.schema import (
    LLMResult,
)

from uuid import UUID
import time
from typing import (
    Any,
)

import app_langchain.tokens as tokens


#
# --- AZURE CHAT MODEL ---
#
def get_chat_model(streaming=True, temperature=0.0, handler=None):
    kwargs = {
        "openai_api_base": os.environ["OPENAI_API_BASE"],
        "openai_api_version": os.environ["OPENAI_API_VERSION"],
        "deployment_name": os.environ["DEPLOYMENT_NAME"],
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "openai_api_type": os.environ["OPENAI_API_TYPE"],
        "temperature": temperature,
        "request_timeout": 6,
        "verbose": True,
    }

    if streaming:
        kwargs.update({
            "streaming": True,
            "callback_manager": BaseCallbackManager([handler]),
        })

    return AzureChatOpenAI(**kwargs)


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
    widget = None
    incomplete_chat_model_answer: str = ""

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
            replace_urls_with_fqdn_and_lastpath(self.incomplete_chat_model_answer)
            )
        time.sleep(10/1000)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id = None, **kwargs: Any) -> Any:
        tokens.add_tokens(text=response.generations[0][0].message.content, type=tokens.TokenType.OUTPUT)


#
# --- URL RESPONSE FORMATTER ---
#
def replace_urls_with_fqdn_and_lastpath(text: str):
    # Matches URLs but not thoses like []()
    url_pattern = re.compile(r'''(?<!\])\((?:https?://)\S+\)|(?<!\()\b(?:https?://)\S+\b(?!\))''')

    # Replace all URLs with their FQDN and last path segment
    def replace_url(match):
        url = match.group(0)
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        path_segments = parsed_url.path.split('/')

        if len(path_segments)>3:
            last_segment = path_segments[-1] if path_segments[-1] else path_segments[-2]
            return f"[{scheme}://{netloc}/.../{last_segment}]({url})"
        else:
           return url

    return url_pattern.sub(replace_url, text)