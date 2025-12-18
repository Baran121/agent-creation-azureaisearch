from typing import Optional

import glob
import csv
import json

from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery 
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,  
    SimpleField,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration)
from openai import AsyncAzureOpenAI
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError


class SearchIndexManager:
    """
    The class for searching of context for user queries.

    :param endpoint: The search endpoint to be used.
    :param credential: The credential to be used for the search.
    :param index_name: The name of an index to get or to create.
    :param dimensions: The number of dimensions in the embedding. Set this parameter only if
                       embedding model accepts dimensions parameter.
    :param model: The embedding model to be used,
                  must be the same as one use to build the file with embeddings.
    :param embeddings_client: The embedding client.
    """
    
    MIN_DIFF_CHARACTERS_IN_LINE = 5
    MIN_LINE_LENGTH = 5
    
    def __init__(
            self,
            endpoint: str,
            credential: AsyncTokenCredential,
            index_name: str,
            dimensions: Optional[int],
            model: str,
            embeddings_client: AsyncAzureOpenAI,
        ) -> None:
        """Constructor."""
        self._dimensions = dimensions
        self._index_name = index_name
        self._embeddings_client = embeddings_client
        self._endpoint = endpoint
        self._credential = credential
        self._index = None
        self._model = model
        self._client = None

    def _get_client(self):
        """Get search client if it is absent."""
        if self._client is None:
            self._client = SearchClient(
                endpoint=self._endpoint, index_name=self._index_name, credential=self._credential)
        return self._client

    async def search(self, message: str) -> str:
        """
        Search the message in the vector store.

        :param message: The customer question.
        :return: The context for the question.
        """
        print(f"DEBUG: Searching for: {message}")
        print(f"DEBUG: Using model: {self._model}, dimensions: {self._dimensions}")

        response = await self._embeddings_client.embeddings.create(
            input=message,
            model=self._model,
            dimensions=self._dimensions

        )
        embedded_question = response.data[0].embedding
        print(f"DEBUG: Embedding created with {len(embedded_question)} dimensions")

        vector_query = VectorizedQuery(vector=embedded_question, k=5, fields="text_vector")
        print(f"DEBUG: Searching index: {self._index_name}")

        response = await self._get_client().search(
            vector_queries=[vector_query],
            select=['chunk'],
        )
        results = [result['chunk'] async for result in response]
        print(f"DEBUG: Found {len(results)} results")

        if results:
            print(f"DEBUG: First result preview: {results[0][:100]}...")

        return "\n------\n".join(results)
      

     
    async def close(self):
        """Close the closeable resources, associated with SearchIndexManager."""
        if self._client:
            await self._client.close()

