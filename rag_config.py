import os

class Configuration:

    OPENAI_ENVIRONMENT = os.getenv("OPENAI_ENVIRONMENT", "DEV")
    ################# AZURE Settings ##############################################################################
    AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    # Update this with the scopes you've configured in Azure AD
    #SCOPE = ["https://your-tenant-name.onmicrosoft.com/your-scope"]

    AZURE_AD_WELL_KNOWN_URL = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0/.well-known/openid-configuration"
    AZURE_AD_KEYS_URL = f"https://sts.windows.net/{AZURE_TENANT_ID}/discovery/v2.0/keys"
    AZURE_AD_ISSUER_URL = f"https://sts.windows.net/{AZURE_TENANT_ID}/"
    AZURE_AD_AUDIENCE = f"api://{AZURE_CLIENT_ID}"

    AZURE_STORAGE_CONTAINER_SAS_TOKEN = os.getenv("AZURE_STORAGE_CONTAINER_SAS_TOKEN")
    AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL","https://omaccpaidfuncst01.blob.core.windows.net")
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME","openai-kb-upload")
    AZURE_STORAGE_CONTAINER_LATEST_FILE_PATH = os.getenv("AZURE_STORAGE_CONTAINER_LATEST_FILE_PATH","latest/")
    AZURE_STORAGE_CONTAINER_ARCHIVE_FILE_PATH = os.getenv("AZURE_STORAGE_CONTAINER_ARCHIVE_FILE_PATH","archive/")
    ################# AZURE Settings ###############################################################################

    ################# OPENAI Settings ##############################################################################
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_GPT_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
    OPENAI_GPT_API_VERSION = "2023-05-15"
    OPENAI_EMBEDDINGS_MODEL = "text-embedding-ada-002"
    OPENAI_GPT_API_EMBEDDING_VERSION = "2023-03-15-preview"
    ################# OPENAI Settings ##############################################################################

    ################# DB Settings ##################################################################################
    PGVECTOR_SCHEMA = os.getenv("PGVECTOR_SCHEMA", "penopenai")
    PGVECTOR_DATABASE = os.getenv("PGVECTOR_DATABASE", "penopenai")
    PGVECTOR_USER = os.getenv("PGVECTOR_USER", "penopenai_user")
    PGVECTOR_HOST = os.getenv("PGVECTOR_HOST")
    PGVECTOR_PORT = os.getenv("PGVECTOR_PORT", "5432")
    PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "pensions_onq")
    PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD")

    PG_LANGCHAIN_CONNECTION_STRING = f"postgresql+psycopg2://{PGVECTOR_USER}:{PGVECTOR_PASSWORD}@{PGVECTOR_HOST}:{PGVECTOR_PORT}/{PGVECTOR_DATABASE}?options=-csearch_path%3D{PGVECTOR_SCHEMA},public&sslmode=require"
    PG_CONNECTION_STRING = "host='%s' dbname='%s' user='%s' password='%s' sslmode='require' port='%s'" % (PGVECTOR_HOST, PGVECTOR_DATABASE, PGVECTOR_USER, PGVECTOR_PASSWORD, PGVECTOR_PORT)
    
    EMBEDDINGS_DELETE_QUERY = "DELETE FROM penopenai.langchain_pg_embedding WHERE cmetadata->>'source' LIKE %s;"
    EMBEDDINGS_COUNT_QUERY = "SELECT COUNT(*) FROM penopenai.langchain_pg_embedding WHERE cmetadata->>'source' LIKE %s;"
    INTERACTION_AUDIT_LOG_INSERT_QUERY = '''INSERT INTO openai_interaction_audit_log (start_timestamp, end_timestamp, user_query, context, model_response, prompt_token,    
                                         completion_token, total_token, prompt, feedback, conversation_history, metadata, status, interaction_detail) 
                                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                                         RETURNING interaction_id;'''
    OPENAI_UPLOAD_BATCH_INSERT_QUERY = '''INSERT INTO penopenai.openai_upload_batch(batch_start_timestamp, batch_end_timestamp, batch_upload_status, created_by, updated_by, updated_date)
                                         VALUES (%s, %s, %s, %s, %s, %s)
                                         RETURNING openai_upload_batch_id;'''
    OPENAI_UPLOAD_BATCH_UPLOAD_QUERY = '''UPDATE penopenai.openai_upload_batch
                                                 SET batch_end_timestamp=%s, batch_upload_status=%s, updated_by=%s, updated_date=%s
                                                 WHERE openai_upload_batch_id=%s;'''
    OPENAI_UPLOAD_FILE_INSERT_QUERY = '''INSERT INTO penopenai.openai_upload_file(
                                         openai_upload_batch_id, upload_source_file_name, upload_destination_file_name, file_page_count, file_upload_status, upload_file_timestamp)
                                         VALUES (%s, %s, %s, %s, %s, %s)
                                          RETURNING openai_upload_file_id;'''
    OPENAI_UPLOAD_FILE_UPDATE_QUERY = '''UPDATE penopenai.openai_upload_file
                                         SET upload_destination_file_name=%s, file_page_count=%s, file_upload_status=%s,upload_file_timestamp=%s
                                         WHERE openai_upload_file_id=%s;'''
    ### Following columns belongs to OPENAI_UPLOAD_FILE table
    UPLOAD_DESTINATION_FILE_NAME_COLUMN = "upload_destination_file_name"
    FILE_PAGE_COUNT_COLUMN = "file_page_count"
    EMBEDDING_COUNT_NEW = "embedding_count_new"
    EMBEDDING_COUNT_OLD = "embedding_count_old"
    FILE_UPLOAD_STATUS_COLUMN = "file_upload_status"
    ################# DB Settings ####################################################################################

    ################# APPLICATION Settings ###########################################################################
    APPLICATION_REFERENCE_FILES_INDEX = os.getenv("APPLICATION_REFERENCE_FILES_INDEX",["myOMERS_reference_guide.pdf",
                "Buy-backs (OnQ).pdf", "Adjustments.pdf", 
            "Estimates (OnQ).pdf", "LiveChat_Responses.pdf", 
            "Retirements (OnQ).pdf", "Terminations (OnQ).pdf",
            "Employer information (OnQ).pdf", "Closed plan consolidation (OnQ).pdf",
            "Survivor benefits (OnQ).pdf", "Plan changes (OnQ).pdf", "Plan basics (OnQ).pdf",
            "Omission periods (OnQ).pdf", "NRA conversion (OnQ).pdf",
            "Marriage breakdown (OnQ).pdf", "Leave periods (OnQ).pdf",
            "Interplan transfers (OnQ).pdf", "General procedures (OnQ).pdf",
            "Enrolment and onboarding (OnQ).pdf", "Divestments (OnQ).pdf",
            "Disabilities (OnQ).pdf", "BRUCE job aids and procedures (OnQ).pdf", "AVCs (OnQ).pdf",
            "bruce_input.csv", "Existing Quick Text Replies.csv"])
    
    
    APPLICATION_TEXT_SPLIT_CHUNK_SIZE = os.getenv("APPLICATION_TEXT_SPLIT_CHUNK_SIZE",1000)
    APPLICATION_TEXT_SPLIT_CHUNK_OVERLAP = os.getenv("APPLICATION_TEXT_SPLIT_CHUNK_OVERLAP",200)
    APPLICATION_OPENAI_EMBEDDING_CHUNK_SIZE = os.getenv("APPLICATION_OPENAI_EMBEDDING_CHUNK_SIZE",16)
    APPLICATION_OPENAI_EMBEDDING_REQUEST_TIMEOUT = os.getenv("APPLICATION_OPENAI_EMBEDDING_REQUEST_TIMEOUT",30)
    APPLICATION_OPENAI_EMBEDDING_MAX_RETRIES = os.getenv("APPLICATION_OPENAI_EMBEDDING_MAX_RETRIES",3)
    APPLICATION_PGVECTOR_RETRIEVER_K = os.getenv("APPLICATION_PGVECTOR_RETRIEVER_K",10)
    APPLICATION_PGVECTOR_RETRIEVER_FETCH_K = os.getenv("APPLICATION_PGVECTOR_RETRIEVER_FETCH_K",30)
    APPLICATION_PGVECTOR_RETRIEVER_LAMBDA_MULT = os.getenv("APPLICATION_PGVECTOR_RETRIEVER_LAMBDA_MULT",0.75)
    ################# APPLICATION Settings #############################################################################