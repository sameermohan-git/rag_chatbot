import azure.functions as func
import openai
import psycopg2
import requests
import json
import traceback 

import logging
import os

from datetime import datetime
from typing import List, Tuple

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

# Import the util classes
from openai_authentication import RequiresAuth
from audit_logger import AuditLogger  
from data_parser import DataParser
from openai_configuration import Configuration

bp_http_openai = func.Blueprint()

@bp_http_openai.route(route="ask_openai_http_fn", auth_level=func.AuthLevel.ANONYMOUS)
@RequiresAuth()
def ask_openai_http_fn(req: func.HttpRequest) -> func.HttpResponse:
    question = req.params.get('question')
    if not question:
        try:
            question = req.get_json().get('question')
        except ValueError:
            traceback.print_exc()

    # Check if 'question' is still empty
    if not question:
        logging.error(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: Please provide a 'question' parameter in the query string or request body.")
        return func.HttpResponse(
            f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: Please provide a 'question' parameter in the query string or request body.",
            status_code=400
        )
    
    logging.info(f'{Configuration.OPENAI_ENVIRONMENT}: Python HTTP trigger function (ask_openai_http_fn) processed incoming question request:'+ question)

    interaction_id = None

    try:
        start_timestamp = datetime.now()

        # Initialize gpt and our embedding model
        llm = AzureChatOpenAI(deployment_name=Configuration.OPENAI_GPT_DEPLOYMENT_NAME, 
                              temperature=0, 
                              n=1,
                        openai_api_version=Configuration.OPENAI_GPT_API_VERSION,verbose=True,
                        request_timeout=30,max_retries=3)

        embeddings = OpenAIEmbeddings(deployment=Configuration.OPENAI_EMBEDDINGS_MODEL,
                                    model=Configuration.OPENAI_EMBEDDINGS_MODEL, 
                                    openai_api_base=Configuration.OPENAI_API_BASE,
                                    openai_api_type='azure',
                                    chunk_size=Configuration.APPLICATION_OPENAI_EMBEDDING_CHUNK_SIZE, 
                                    request_timeout=Configuration.APPLICATION_OPENAI_EMBEDDING_REQUEST_TIMEOUT,
                                    max_retries=Configuration.APPLICATION_OPENAI_EMBEDDING_MAX_RETRIES,
                                    openai_api_version=Configuration.OPENAI_GPT_API_EMBEDDING_VERSION)
    
        store = PGVector(
            connection_string=Configuration.PG_LANGCHAIN_CONNECTION_STRING, 
            embedding_function=embeddings, 
            collection_name=Configuration.PGVECTOR_COLLECTION_NAME,
            distance_strategy=DistanceStrategy.COSINE
        )
        logging.debug(f"{Configuration.OPENAI_ENVIRONMENT}: ask_openai_http_fn: DB connection successful!")
        retriever = store.as_retriever(search_kwargs={'k': Configuration.APPLICATION_PGVECTOR_RETRIEVER_K, 
                                                      'fetch_k': Configuration.APPLICATION_PGVECTOR_RETRIEVER_FETCH_K, 
                                                      'lambda_mult':Configuration.APPLICATION_PGVECTOR_RETRIEVER_LAMBDA_MULT})

        # Build prompt
        template = """You are AI agent. Use the following pieces of context to answer the question at the end. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer. 
        Don't mention in your response that you are anyway related to AI, and always consider yourself as AI Agent.
        If question includes PII data, always and only say without referring any context, Sorry, but Live Chat Assist cannot be used when Personal Data is shared.
        If relevant website link is mentioned in context, include it in answer for better response quality, dont leave it as placeholder."
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            return_source_documents=True,
                                            verbose=True,
                                            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT})
        
        chat_history = []
        with get_openai_callback() as cb:
            result = qa({"question": question, "chat_history": chat_history})

        sources = [doc.metadata for doc in result["source_documents"]]
        try:
            source_formatted = DataParser.parse_source_detail_response(sources)
            logging.info(f"metadata: {str(source_formatted)}")
        except Exception as e:
            # Handle the exception gracefully, e.g., log the error or return a default value
            logging.error(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {e=}, {type(e)=}")

        # Create an instance of the AuditLogger class
        audit_logger = AuditLogger()

        # Insert the record into the database and get the interaction ID
        interaction_id = audit_logger.insert_openai_interaction_audit_log(start_timestamp=start_timestamp, user_query=question, 
                                                        model_response=str(result["answer"]), metadata=str(source_formatted),
                                                        status='SUCCESS',context=None, 
                                                        prompt=None, feedback=None, conversation_history=None, 
                                                        interaction_detail=None, callback=cb)

        return func.HttpResponse(str(result["answer"]) + "\ninteraction_id: " + str(interaction_id),
            status_code=200)
    except psycopg2.errors.InsufficientPrivilege as sqlexcep:
        traceback.print_exc()
        logging.error(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {sqlexcep=}, {type(sqlexcep)=}")
        return func.HttpResponse(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {sqlexcep=}, {type(sqlexcep)=};", status_code=500) 
    except Exception as ex:
        traceback.print_exc()
        logging.error(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {ex=}, {type(ex)=}")
        if interaction_id is None:
             # Create an instance of the AuditLogger class
            audit_logger = AuditLogger(Configuration.PG_CONNECTION_STRING,Configuration.PGVECTOR_SCHEMA)
            interaction_id = audit_logger.insert_openai_interaction_audit_log(start_timestamp=start_timestamp, user_query=question,
                                                        model_response=None, metadata=None,
                                                        status='FAILURE',context=None, 
                                                        prompt=None, feedback=None, conversation_history=None, 
                                                        interaction_detail=str(ex), callback=None)
        return func.HttpResponse(f"ask_openai_http_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {ex=}, {type(ex)=};"
                                 +"\ninteraction_id: " + str(interaction_id), status_code=500) 
    finally:
            # Close the database connection in the finally block
            audit_logger.close()

@bp_http_openai.route(route="ask_openai_http_noauth_fn", auth_level=func.AuthLevel.ANONYMOUS)
def ask_openai_http_noauth_fn(req: func.HttpRequest) -> func.HttpResponse:

    question = req.params.get('question')
    if not question:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            question = req_body.get('question')


     # Check if 'question' is still empty
    if not question:
        return func.HttpResponse(
            f"{Configuration.OPENAI_ENVIRONMENT}: Please provide a 'question' parameter in the query string or request body.",
            status_code=400
        )
    
    logging.info(f'{Configuration.OPENAI_ENVIRONMENT}: Python HTTP trigger function (ask_openai_http_noauth_fn) processed incoming question request:'+ question)

    try:
        # Initialize gpt-35-turbo and our embedding model
        llm = AzureChatOpenAI(deployment_name=Configuration.OPENAI_GPT_DEPLOYMENT_NAME, temperature=0, 
                        openai_api_version=Configuration.OPENAI_GPT_API_VERSION,verbose=True)

        embeddings = OpenAIEmbeddings(deployment=Configuration.OPENAI_EMBEDDINGS_MODEL,
                                    model=Configuration.OPENAI_EMBEDDINGS_MODEL, 
                                    openai_api_base=Configuration.OPENAI_API_BASE,
                                    openai_api_type='azure',
                                    chunk_size=1, 
                                    openai_api_version=Configuration.OPENAI_GPT_API_EMBEDDING_VERSION)
        store = PGVector(
            connection_string=Configuration.PG_LANGCHAIN_CONNECTION_STRING, 
            embedding_function=embeddings, 
            collection_name=Configuration.PGVECTOR_COLLECTION_NAME,
            distance_strategy=DistanceStrategy.COSINE
        )
        logging.debug(f"ask_openai_http_noauth_fn: {Configuration.OPENAI_ENVIRONMENT}: DB connection successful!")
        retriever = store.as_retriever(search_kwargs={'k': 10, 'fetch_k': 30, 'lambda_mult':0.75})

        # Adapt if needed
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:""")

        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=retriever,
                                            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=True)
        
        chat_history = []
        result = qa({"question": question, "chat_history": chat_history})
        sources = [doc.metadata for doc in result["source_documents"]]
        
        logging.info("metadata:"+ str(sources))

        return func.HttpResponse(Configuration.OPENAI_ENVIRONMENT + str(result["answer"]) + ";\n" + str(sources),
             status_code=200)
    except Exception as error:
        logging.error(f"ask_openai_http_noauth_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {error=}, {type(error)=}")
        return func.HttpResponse(f"ask_openai_http_noauth_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {error=}, {type(error)=}", status_code=500) 
