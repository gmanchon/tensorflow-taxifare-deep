
from google.cloud import storage


def download_blob(bucket_name, blob_from, file_to):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_from)
    blob.download_to_filename(file_to)


def upload_file(bucket_name, file_from, blob_to):

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_to)
    blob.upload_from_filename(file_from)
