import s3fs
from datasets import load_dataset_builder
from botocore.session import Session
import os
from smart_open import open

home_dir = os.path.expanduser("~")

with open(f"{home_dir}/.aws/credentials") as f:
    lines = f.readlines()
    key = lines[1].split(" = ")[1].strip()
    secret = lines[2].split(" = ")[1].strip()

storage_options = {"key": key, "secret": secret}
s3_session = Session(profile="default")
storage_options = {"session": s3_session}

lang = input("What lang? : ")
s3_dir = s3fs.S3FileSystem(**storage_options)
builder = load_dataset_builder("allenai/madlad-400", languages=[lang])

output_dir = "s3://madlad-data"

builder.download_and_prepare(output_dir) #storage_options=storage_options, file_format="parquet")