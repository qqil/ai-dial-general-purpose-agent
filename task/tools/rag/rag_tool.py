import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

# TODO: provide system prompt for Generation step
_SYSTEM_PROMPT = """## Role
You are a helpful assistant for answering questions based on provided context.

## Response Generation Guidelines
- Always base your answers solely on the provided context. Do not use any external knowledge or assumptions.
- If the answer is not explicitly stated in the context, do not attempt to infer or guess the answer. Instead, respond with "The answer is not available in the provided context."
- Be concise and to the point. 
- If the context contains multiple pieces of relevant information, synthesize it into a coherent answer that directly addresses the user's question.

## Input Format
The input to your response generation will be in the following format:
CONTEXT: Relevant pieces of information extracted from the document will be provided here. This may include multiple chunks of text that are relevant to the user's question.
USER REQUEST: The user's question or query that you need to answer based on the provided context.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        #TODO:
        # 1. Set endpoint
        # 2. Set deployment_name
        # 3. Set document_cache. DocumentCache is implemented, relate to it as to centralized Dict with file_url (as key),
        #    and indexed embeddings (as value), that have some autoclean. This cache will allow us to speed up RAG search.
        # 4. Create SentenceTransformer and set is as `model` with:
        #   - model_name_or_path='all-MiniLM-L6-v2', it is self hosted lightwait embedding model.
        #     More info: https://medium.com/@rahultiwari065/unlocking-the-power-of-sentence-embeddings-with-all-minilm-l6-v2-7d6589a5f0aa
        #   - Optional! You can set it use CPU forcefully with `device='cpu'`, in case if not set up then will use GPU if it has CUDA cores
        # 5. Create RecursiveCharacterTextSplitter as `text_splitter` with:
        #   - chunk_size=500
        #   - chunk_overlap=50
        #   - length_function=len
        #   - separators=["\n\n", "\n", ". ", " ", ""]
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cpu')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        # TODO: set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        return "rag_tool"
    
    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "Tool for answering questions based on document content. " \
            "It performs semantic search to find relevant information in the document and generates answers based on that information. " \
            "Supported formats: PDF, TXT, CSV, HTML. " \
            "Use this tool when you need to answer a question that requires specific information from a document. " \
            "The tool will retrieve relevant chunks of the document and generate a response based on that information." \
            "Don't use this tool when it is required to analyze the document as a whole or when the question can be answered without referring to specific content in the document."

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema:
        #  - request is string, description: "The search query or question to search for in the document", required
        #  - file_url is string, required
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document."
                },
                "file_url": {
                    "type": "string",
                    "description": "URL of the file to perform RAG search on."
                }
            },
            "required": ["request", "file_url"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        #TODO:
        # 1. Load arguments with `json`
        # 2. Get `request` from arguments
        # 3. Get `file_url` from arguments
        # 4. Get stage from `tool_call_params`
        # 5. Append content to stage: "## Request arguments: \n"
        # 6. Append content to stage: `f"**Request**: {request}\n\r"`
        # 7. Append content to stage: `f"**File URL**: {file_url}\n\r"`
        # 8. Create `cache_document_key`, it is string from `conversation_id` and `file_url`, with such key we guarantee
        #    access to cached indexes for one particular conversation,
        # 9. Get from `document_cache` by `cache_document_key` a cache
        # 10. If cache is present then set it as `index, chunks = cached_data` (cached_data is retrieved cache from 9 step),
        #     otherwise:
        #       - Create DialFileContentExtractor and extract text by `file_url` as `text_content`
        #       - If no `text_content` then appen to stage info about it ans return the string with the error that file content is not found
        #       - Create `chunks` with `text_splitter`
        #       - Create `embeddings` with `model`
        #       - Create IndexFlatL2 with `384` dimensions as `index` (more about IndexFlatL2 https://shayan-fazeli.medium.com/faiss-a-quick-tutorial-to-efficient-similarity-search-595850e08473)
        #       - Add to `index` np.array with created embeddings as type 'float32'
        #       - Add to `document_cache`
        # 11. Prepare `query_embedding` with model. You need to encode request as type 'float32'
        # 12. Through created index make search with `query_embedding`, `k` set as 3. As response we expect tuple of
        #     `distances` and `indices`
        # 13. Now you need to iterate through `indices[0]` and and by each idx get element from `chunks`, result save as `retrieved_chunks`
        # 14. Make augmentation
        # 15. Append content to stage: "## RAG Request: \n"
        # 16. Append content to stage: `ff"```text\n\r{augmented_prompt}\n\r```\n\r"` (will be shown as markdown text)
        # 17. Append content to stage: "## Response: \n"
        # 18. Now make Generation with AsyncDial (don't forget about api_version '025-01-01-preview, provide LLM with system prompt and augmented prompt and:
        #   - stream response to stage (user in real time will be able to see what the LLM responding while Generation step)
        #   - collect all content (we need to return it as tool execution result)
        # 19. return collected content
        arguments = json.loads(tool_call_params.tool_call.function.arguments or "{}")
        request = arguments.get("request")
        file_url = arguments.get("file_url")
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}_{file_url}"
        cached_data = self.document_cache.get(cache_document_key)

        if cached_data:
            index, chunks = cached_data
        else:
            content_extractor = DialFileContentExtractor(
                endpoint=self.endpoint, 
                api_key=tool_call_params.api_key
            )
            text_content = content_extractor.extract_text(file_url)

            if not text_content:
                stage.append_content("Error: File content not found.")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.model.encode(chunks)
            
            index = faiss.IndexFlatL2(384)
            index.add(np.array(embeddings).astype('float32')) # type: ignore

            self.document_cache.set(cache_document_key, index, chunks)
        
        query_embedding = self.model.encode([request]).astype('float32')
        k = min(3, len(chunks))
        distances, indices = index.search(query_embedding, k=k) # type: ignore

        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        
        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")
        
        dial_client = AsyncDial(base_url=self.endpoint, api_key=tool_call_params.api_key)

        reponse_stream = await dial_client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt}
            ],
            stream=True
        )
        collected_response = ""
        async for chunk in reponse_stream:
            if not chunk.choices or len(chunk.choices) == 0 or not chunk.choices[0].delta:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                stage.append_content(delta.content)
                collected_response += delta.content

        return collected_response


    def __augmentation(self, request: str, chunks: list[str]) -> str:
        #TODO: make prompt augmentation
        return f"CONTEXT:\n\n{'\n\n'.join(chunks)}\n\nUSER REQUEST: {request}\n\n"
