import os
from time import sleep
from uuid import uuid4

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import vars
import app_langchain.tokens as tokens


embeddings = OpenAIEmbeddings(model=vars.EMBEDDING_MODEL)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = 650,
        chunk_overlap  = 25,
        length_function = tokens.tiktoken_len,
        separators = ['\n\n', '\n', ' ', '']
    )


def chunk_pdf(pages, document_id):
    chunks = []
    chunk_count = 0
    text_splitter = get_text_splitter()
    first = True
    title = ""
    filename = ""

    for page in pages:
        if first:
            if page.metadata['title'] == "":
                title = os.path.splitext(os.path.basename(page.metadata['file_path']))[0]
            else:
                title = page.metadata['title']
            filename = os.path.basename(page.metadata['file_path'])
            first = False

        pageSplitted = text_splitter.split_text(page.page_content)
        for chunkText in enumerate(pageSplitted):
            chunks.append({
                'id': str(uuid4()),
                'document_id': document_id,
                'filename': filename,
                'title': title,
                'text': chunkText[1],
                'page': page.metadata['page']+1,
                'total_pages': page.metadata['total_pages'],
                'chunk': str(chunk_count),
            })
            chunk_count = chunk_count+1
    return chunks


def embed_pdf_to_pinecone(index, chunks, progress_widget):
    batch_size = 50
    progress_text = f'Processing the document "{chunks[0]["title"]}" for Q&A. Please wait.'
    my_bar = progress_widget.progress(0, text=progress_text)
    percent_complete = 0
    step = int(batch_size/len(chunks) * 100)
    if step > 100:
        step=100

    for i in range(0, len(chunks), batch_size):
        # set end position of batch
        i_end = min(i+batch_size, len(chunks))

        # inicialize info
        embeds = []

        # get batch of meta, text and IDs
        meta = chunks[i:i_end]
        texts_batch = [x['text'] for x in meta]
        ids_batch = [x['id'] for x in meta]

        # create embeddings (try-except added to avoid RateLimitError)
        for j in range(0, len(meta)): # process everything 1 by 1 because input parameters are limited to 1
            try:
                response = embeddings.embed_query(texts_batch[j])
            except:
                done = False
                while not done:
                    sleep(1)
                    try:
                        response = embeddings.embed_query(texts_batch[j])
                        done = True
                    except:
                        pass
            embeds = embeds + [response]

        # update the meta
        meta_batch = [{
            'document_id': x['document_id'],
            'filename': x['filename'],
            'title': x['title'],
            'text': x['text'],
            'page': x['page'],
            'total_pages': x['total_pages'],
            'chunk': x['chunk'],
        } for x in meta]

        # upsert batch to Pinecone
        to_upsert = zip(ids_batch, embeds, meta_batch)
        index.upsert(vectors=list(to_upsert), namespace='uploaded-documents')

        percent_complete = percent_complete + step
        if percent_complete > 100:
            percent_complete=100
        my_bar.progress(percent_complete, text=progress_text)
    my_bar.progress(100, text=progress_text)