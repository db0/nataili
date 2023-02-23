import os
import random
import subprocess
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import webdataset as wds
from tqdm import tqdm

from nataili.cache import Cache
from nataili.clip.image import ImageEmbed
from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger

parser = ArgumentParser()
parser.add_argument("--start", type=int, help="start index", required=True)
parser.add_argument("--end", type=int, help="end index", required=True)
parser.add_argument("--zfill", type=int, default=6, help="zfill for index")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--endpoint", type=str, default=None, help="alternative endpoint for s3 i.e. R2")
parser.add_argument("--access_key", type=str, default=None)
parser.add_argument("--secret_key", type=str, default=None)
parser.add_argument("--bucket", type=str, required=True, help="s3://{bucket}")
parser.add_argument("--input_path", type=str, default=None, help="s3://{bucket}/{input_path}/{input}_{start..end}.tar")
parser.add_argument(
    "--output_path", type=str, default=None, help="s3://{bucket}/{output_path}/{output}_{start..end}.tar"
)
parser.add_argument("--input", type=str, required=True, help="s3://{bucket}/{input}_{start..end}.tar")
parser.add_argument("--output", type=str, required=True, help="s3://{bucket}/{output}_{start..end}.tar")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to use")
parser.add_argument("--model", type=str, default="ViT-H-14", help="model to use")
parser.add_argument("--shard_size", type=int, default=10000, help="number of images per shard")
args = parser.parse_args()

executor = ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="SaveThread")
mm = ClipModelManager()
try:
    mm.load(args.model, args.gpu_id)
except:
    logger.error(f"Failed to load model {args.model}")
    exit(1)

s3_client = boto3.resource(
    "s3",
    endpoint_url=args.endpoint,
    aws_access_key_id=args.access_key if args.access_key else os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=args.secret_key if args.secret_key else os.environ.get("AWS_SECRET_ACCESS_KEY"),
)
bucket = s3_client.Bucket(args.bucket)


def embed_shard(shard_number: int, batch_size: int = 512):
    shard = f"{shard_number:0{args.zfill}d}"
    tar_file = f"{args.output}_{shard}.tar"
    try:
        bucket.Object(f"{args.output_path}/{tar_file}").last_modified
        logger.info(f"Shard {args.output}/{tar_file} already exists")
        return
    except:
        logger.info(f"Shard {args.output}/{tar_file} does not exist")
        pass
    if args.endpoint:
        url = f"pipe: aws s3 cp --endpoint-url {args.endpoint}"
    else:
        url = f"pipe: aws s3 cp"
    if args.input_path:
        url += f" s3://{args.bucket}/{args.input_path}/{args.input}_{shard}.tar -"
    else:
        url += f" s3://{args.bucket}/{args.input}_{shard}.tar -"
    dataset = wds.WebDataset(url).shuffle(100).decode("pil")
    cache_name = f"{args.input}_{shard}"
    cache = Cache(cache_name)
    db_files = cache.get_all()
    db_files = {file for file in db_files}
    logger.info(f"Found {len(db_files)} files in cache")
    batch = []
    progress_bar = tqdm(total=args.shard_size, unit="image", disable=False, desc=f"Shard {shard}")
    if len(db_files) != args.shard_size:
        for file in dataset:
            if file["__key__"] in db_files:
                progress_bar.total -= 1
                progress_bar.refresh()
                continue
            image = {}
            image["pil_image"] = file["webp"]
            image["filename"] = file["__key__"]
            batch.append(image)
            remaining = progress_bar.total - progress_bar.n
            if len(batch) == batch_size or remaining <= batch_size:
                image_embed = ImageEmbed(mm.loaded_models[args.model], cache)
                image_embed._batch(batch)
                del batch
                batch = []
                update_size = min(batch_size, remaining)
                progress_bar.update(update_size)
                progress_bar.refresh()
        logger.info(f"Finished embedding shard {shard}")
        logger.info(f"Waiting for cache to finish writing")
        time.sleep(10)
    export = cache.get_all_export()
    logger.info(f"Exporting shard {shard}")
    sink = wds.TarWriter(tar_file)
    export = sorted(export.items(), key=lambda x: x[0])
    export = {file: pil_hash for file, pil_hash in export}
    logger.info(f"Found {len(export)} files in cache")
    for file in export:
        pil_hash = export[file]
        embed_file = os.path.join(cache.cache_dir, pil_hash + ".npy")
        embed = np.load(embed_file)
        wds_data = {"__key__": file, "npy": embed}
        sink.write(wds_data)
    sink.close()
    logger.info(f"Finished exporting shard {shard} to {tar_file}")
    logger.info(f"Uploading shard {shard} to s3")
    #TODO: use boto3 instead of subprocess
    if args.endpoint:
        command = f"aws s3 cp --endpoint-url {args.endpoint}"
    else:
        command = f"aws s3 cp"
    if args.output_path:
        command += f" {tar_file} s3://{args.bucket}/{args.output_path}/{tar_file}"
    else:
        command += f" {tar_file} s3://{args.bucket}/{tar_file}"
    process = subprocess.check_output(command, shell=True)
    logger.info(f"Finished uploading shard {shard} to s3")


if __name__ == "__main__":
    total = args.end - args.start + 1
    logger.info(f"Embedding {total} shards")
    for shard in range(args.start, args.end + 1):
        if args.threads == 1:
            embed_shard(shard, args.batch_size)
        else:
            executor.submit(embed_shard, shard, args.batch_size)
        time.sleep(random.randint(1, 5))
    if args.threads > 1:
        executor.shutdown(wait=True)
