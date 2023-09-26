import os
from time import sleep
from uuid import uuid4

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import vars
import app_langchain.tokens as tokens


embeddings = OpenAIEmbeddings(model=vars.EMBEDDING_MODEL)


def inicialize_doc_namespace(index, namespace):
    metadata = {
            'document_md5': "0",
            'filename': "Init_doc",
            'title': "Init_doc",
            'text': " ",
            'page': 0,
            'chunk': "0",
        }
    index.upsert(vectors=[{'id':str(uuid4()), 'values':embeddings.embed_query(" "), 'metadata':metadata}], namespace=namespace)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size = 650,
        chunk_overlap  = 25,
        length_function = tokens.tiktoken_len,
        separators = ['\n\n', '\n', ' ', '']
    )


def parse_pdf_metadata(metadata):
    metadata["filename"] = metadata.pop("file_path")
    metadata["page_number"] = metadata.pop("page")
    return metadata


def chunk_doc(pages, file_extension, document_md5):
    chunks = []
    chunk_count = 0
    text_splitter = get_text_splitter()
    first = True
    title = ""
    filename = ""

    for page in pages:
        if file_extension == ".pdf":
            page.metadata = parse_pdf_metadata(page.metadata)
        if first:
            if 'title' not in page.metadata or page.metadata['title'] == "":
                title = os.path.splitext(os.path.basename(page.metadata['filename']))[0]
            else:
                title = page.metadata['title']
            filename = os.path.basename(page.metadata['filename'])
            first = False

        pageSplitted = text_splitter.split_text(page.page_content)
        for chunkText in enumerate(pageSplitted):
            data = {
                'id': str(uuid4()),
                'document_md5': document_md5,
                'filename': filename,
                'title': title,
                'text': chunkText[1],
                'page': page.metadata['page_number']+1 if 'page_number' in page.metadata else 1,
                'chunk': str(chunk_count),
            }
            chunks.append(data)
            chunk_count = chunk_count+1
    return chunks


def embed_doc_to_pinecone(chunks, progress_widget):
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
            'document_md5': x['document_md5'],
            'filename': x['filename'],
            'title': x['title'],
            'text': x['text'],
            'page': x['page'],
            'chunk': x['chunk'],
        } for x in meta]

        # upsert batch to Pinecone
        to_upsert = zip(ids_batch, embeds, meta_batch)
        vars.index.upsert(vectors=list(to_upsert), namespace='uploaded-documents')

        percent_complete = percent_complete + step
        if percent_complete > 100:
            percent_complete=100
        my_bar.progress(percent_complete, text=progress_text)
    my_bar.progress(100, text=progress_text)