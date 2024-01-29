import azure.functions as func

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobPrefix
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.docstore.document import Document

import nltk
import os
import tempfile
import datetime
import logging
import traceback 

from openai_authentication import RequiresAuth
from openai_configuration import Configuration
from database_operations import DatabaseOperations
from audit_logger import AuditLogger  
from azure_blob_operations import AzureBlobOperations
from status_enum import Status

# Set the path to the nltk_data directory
nltk.data.path.append("/home/site/wwwroot/nltk_data")

bp_http_openai_upload = func.Blueprint()

@bp_http_openai_upload.route(route="update_openai_datastore", auth_level=func.AuthLevel.ANONYMOUS)
@RequiresAuth()
def update_openai_datastore_fn(req: func.HttpRequest) -> func.HttpResponse:
    # Create an instance of the AuditLogger class
    audit_logger = AuditLogger()

    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(f"{Configuration.AZURE_STORAGE_ACCOUNT_URL}{Configuration.AZURE_STORAGE_CONTAINER_SAS_TOKEN}")
        container_client = blob_service_client.get_container_client(container=Configuration.AZURE_STORAGE_CONTAINER_NAME)
        latest_blob_list = container_client.list_blobs(name_starts_with=Configuration.AZURE_STORAGE_CONTAINER_LATEST_FILE_PATH)
        
        logging.info("update_openai_datastore_fn: Got list of documents from Storage")

        not_found_files = []
        filtered_files_list = []
        for blob in latest_blob_list:
            blob_name = blob.name
            if blob_name.endswith(".pdf") or blob_name.endswith(".csv"):
                if blob_name[len(Configuration.AZURE_STORAGE_CONTAINER_LATEST_FILE_PATH):] not in Configuration.APPLICATION_REFERENCE_FILES_INDEX:
                    not_found_files.append(blob_name)
                else:
                    filtered_files_list.append(blob_name)
        
        if len(filtered_files_list) < 1:
            logging.error(f"update_openai_datastore_fn: {Configuration.OPENAI_ENVIRONMENT}: None of the file names matched, so processing stopped - {filtered_files_list}")
            return func.HttpResponse(
                f"update_openai_datastore_fn: {Configuration.OPENAI_ENVIRONMENT}: None of the file names matched, so processing stopped - {filtered_files_list}",
                status_code=400
            )
        
        batch_id = audit_logger.insert_openai_upload_batch_entry()
        logging.info(f"update_openai_datastore_fn: Batch Id: {batch_id} - Got Batch Id with files to process count - {len(filtered_files_list)}")

        if len(not_found_files) > 0:
            for file in not_found_files:
                audit_logger.insert_openai_upload_file_entry(batch_id, file, None, 0, Status.FILE_NAME_MISMATCH_FAILED.name)
            logging.error(f"update_openai_datastore_fn: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id} - These files do not match with application reference list, so will not be processed. Check if name is correct:{not_found_files}")
        
        # Start the batch operation in the background
        logging.info(f"update_openai_datastore_fn: Batch Id: {batch_id} - Calling Async Batch Process.")
        batch_processing(container_client, filtered_files_list, audit_logger, batch_id)
        
        # Respond to the caller with a 202 Accepted status
        return func.HttpResponse("update_openai_datastore_fn: Batch Processing Started", status_code=202)
    
    except Exception as ex:
        traceback.print_exc()
        logging.error(f"update_openai_datastore_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {ex=}, {type(ex)=}")
        return func.HttpResponse(f"update_openai_datastore_fn: {Configuration.OPENAI_ENVIRONMENT}: An error occurred: {ex=}, {type(ex)=};", status_code=500) 

def batch_processing(container_client, filtered_files_list, audit_logger, batch_id):
    try:
        for blob in filtered_files_list:
            logging.info(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Starting processing for file - {blob}")
            upload_file_id = audit_logger.insert_openai_upload_file_entry(batch_id, blob, None, 0, Status.FILE_UPLOAD_IN_PROGRESS.name)

            # Step-1 - Generate Pages for files
            try:
                pages = process_blob_and_return_pages(container_client, blob, upload_file_id)
            except Exception as e:
                logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - An error occurred while generating pages for file {blob}: {e=}, {type(e)=}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                                Configuration.FILE_PAGE_COUNT_COLUMN: 0,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_FAILED_PAGES_GENERATED.name
                                                            })
                continue

            if len(pages) < 1:
                logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - Zero pages generated for blob - {blob}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                                Configuration.FILE_PAGE_COUNT_COLUMN: 0,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_FAILED_PAGES_GENERATED.name
                                                            })
                continue
            
            logging.info(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - {len(pages)} pages generated for blob - {blob}")
            audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                            Configuration.FILE_PAGE_COUNT_COLUMN: len(pages),
                                                            Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_IN_PROGRESS_PAGES_GENERATED.name
                                                            })
            
            # Step-2 - Delete embeddings
            try:
                db_operation = DatabaseOperations()
                delete_embeddings_count = db_operation.delete_embeddings(blob)
                logging.info(f"batch_processing: Batch Id: {batch_id}; File Id: {upload_file_id}; Blob Name: {blob} - Total embeddings deleted - {delete_embeddings_count}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, {Configuration.EMBEDDING_COUNT_OLD: delete_embeddings_count,Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_IN_PROGRESS_OLD_EMBEDDINGS_DELETED.name})
            except Exception as e:
                logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - An error occurred while deleting embeddings for file {blob}: {e=}, {type(e)=}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                                Configuration.EMBEDDING_COUNT_OLD: 0,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_FAILED_OLD_EMBEDDINGS_DELETED.name
                                                            })
                continue
        
            # Step-3 - Create New Embeddings
            try:
                generated_embeddings_count = generate_embeddings(pages, blob, upload_file_id)
                logging.info(f"batch_processing: Batch Id: {batch_id}; File Id: {upload_file_id} - Blob Name: {blob} - Completed generation of embeddings of files and inserted into DB - {generated_embeddings_count}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, {Configuration.EMBEDDING_COUNT_NEW: generated_embeddings_count,Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_IN_PROGRESS_NEW_EMBEDDINGS_GENERATED.name})
            except Exception as e:
                logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - An error occurred while generating embeddings for file {blob}: {e=}, {type(e)=}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                                Configuration.EMBEDDING_COUNT_NEW: 0,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_FAILED_NEW_EMBEDDINGS_GENERATED.name
                                                            })
                continue               
            
            # Step-4 - Move Files to Archive
            try:
                blob_operation = AzureBlobOperations()
                destination_blob = blob_operation.move_blobs(blob)
                logging.info(f"batch_processing: Batch Id: {batch_id}; File Id: {upload_file_id} - Blob Moved to {destination_blob}")
                destination_blob = audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                                {
                                                                Configuration.UPLOAD_DESTINATION_FILE_NAME_COLUMN: destination_blob,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_SUCCESS.name
                                                                })
                audit_logger.update_openai_upload_batch_entry(Status.BATCH_SUCCESS.name, batch_id)
            except Exception as e:
                logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id}; File Id: {upload_file_id} - An error occurred while generating embeddings for file {blob}: {e=}, {type(e)=}")
                audit_logger.update_openai_upload_file_entry(upload_file_id, 
                                                            {
                                                                Configuration.UPLOAD_DESTINATION_FILE_NAME_COLUMN: None,
                                                                Configuration.FILE_UPLOAD_STATUS_COLUMN: Status.FILE_UPLOAD_FAILED_ARCHIVE_BLOB.name
                                                            })
                continue       

    except Exception as ex:
        traceback.print_exc()
        logging.error(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}: Batch Id: {batch_id} - An error occurred: {ex=}, {type(ex)=}")
        audit_logger.update_openai_upload_batch_entry(Status.BATCH_FAILED.name, batch_id)
        return func.HttpResponse(f"batch_processing: {Configuration.OPENAI_ENVIRONMENT}:  Batch Id: {batch_id} - An error occurred: {ex=}, {type(ex)=};", status_code=500) 
    return func.HttpResponse(f"Batch Id: {batch_id} - Completed Batch Process!",status_code=200)


def process_blob_and_return_pages(container_client, blob, upload_file_id):
    pages = []
    logging.info(f"process_blob_and_return_pages: {Configuration.OPENAI_ENVIRONMENT}: File Id: {upload_file_id} - Starting processing for blob - {blob}")

    blob_client = container_client.get_blob_client(blob)


    if blob_client.exists():
        with tempfile.TemporaryDirectory() as temp_dir:
        
            file_path = os.path.join(temp_dir, container_client.container_name, blob)
            logging.info(f"process_blob_and_return_pages: File Id: {upload_file_id} - Reading file from storage - {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file:
                blob_data = blob_client.download_blob()
                blob_data.readinto(file)

            logging.info(f"process_blob_and_return_pages: File Id: {upload_file_id} - File Read Complete - {file_path}")

            loader = UnstructuredFileLoader(file_path, mode="paged")
            
            text_splitter_tik = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=Configuration.APPLICATION_TEXT_SPLIT_CHUNK_SIZE, 
                chunk_overlap=Configuration.APPLICATION_TEXT_SPLIT_CHUNK_OVERLAP
            )
            pages = loader.load_and_split(text_splitter_tik)
            logging.info(f"process_blob_and_return_pages: File Id: {upload_file_id} - Pages Generated - {len(pages)} for {blob}")
    return pages

def generate_embeddings(pages, blob, upload_file_id):
    generated_embeddings_count = 0
    try:
        embeddings = OpenAIEmbeddings(deployment=Configuration.OPENAI_EMBEDDINGS_MODEL,
                                    model=Configuration.OPENAI_EMBEDDINGS_MODEL, 
                                    openai_api_base=Configuration.OPENAI_API_BASE,
                                    openai_api_type='azure',
                                    chunk_size=16, 
                                    request_timeout=30,
                                    max_retries=3,
                                    openai_api_version=Configuration.OPENAI_GPT_API_EMBEDDING_VERSION)
    
        # When we have an existing PG Vector 
        metadata = {"document_reference": "Pensions OnQ Knowledgebase", "CreatedOn": datetime.datetime.now().isoformat()}
        PGVector.from_documents(
            embedding=embeddings,
            documents=pages,
            collection_name=Configuration.PGVECTOR_COLLECTION_NAME,
            distance_strategy=DistanceStrategy.COSINE,
            pre_delete_collection = False,
            connection_string=Configuration.PG_LANGCHAIN_CONNECTION_STRING,
            collection_metadata=metadata
        )
        logging.info(f"generate_embeddings: File Id: {upload_file_id} - Embeddings Generated")

        try:
            db_operation = DatabaseOperations()
            generated_embeddings_count = db_operation.count_generated_embeddings(blob)
        except Exception as e:
            # Handle the exception gracefully, e.g., log the error or return a default value
            logging.error(f"generate_embeddings: {Configuration.OPENAI_ENVIRONMENT}: An error occurred while getting count of generated embeddings for {blob}: {e=}, {type(e)=}")
            return -1
    except Exception as e:
            # Handle the exception gracefully, e.g., log the error or return a default value
            logging.error(f"generate_embeddings: {Configuration.OPENAI_ENVIRONMENT}: An error occurred while generating pages for file {blob}: {e=}, {type(e)=}")
            return -1
    return generated_embeddings_count