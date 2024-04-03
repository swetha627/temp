from minio import Minio


# Initialize MinIO client object
minio_client = Minio(
        "100.81.217.239:30114",
    access_key="root",
    secret_key="tatacomm",
    secure=False # Change to True if MinIO is configured with SSL/TLS
)

def create_bucket(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")

def list_files(bucket_name):
    objects = minio_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        print(obj.object_name)

if __name__ == "__main__":

    
    # Create bucket
    #create_bucket("kubeflow")
    
    # List files in the bucket
    list_files("kubeflow")

