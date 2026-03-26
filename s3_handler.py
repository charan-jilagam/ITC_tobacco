"""
s3_handler.py  –  Download images from S3 (Tobacco product images/)
                   and upload annotated results to model_results/

Category routing:
    category_id = 2  →  YOLO detection pipeline
    category_id = 3  →  Freshness (OCR) pipeline
"""
import os
import re
import mimetypes
import logging

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from db_handler import initialize_db_connection, close_db_connection

logger = logging.getLogger(__name__)

CATEGORY_YOLO      = 2
CATEGORY_FRESHNESS = 3


class S3Handler:
    def __init__(self, s3_config, db_config):
        self.bucket_name     = s3_config['bucket_name']
        self.image_folder    = s3_config.get('image_folder_s3', 'Tobacco product images/')
        self.results_folder  = s3_config.get('results_folder_s3', 'model_results/')
        self.db_config       = db_config

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=s3_config['access_key'],
            aws_secret_access_key=s3_config['secret_key'],
            region_name=s3_config.get('region', 'ap-south-1')
        )
        logger.info(f"S3Handler ready  bucket={self.bucket_name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(name):
        """Remove unsafe characters from a filename / folder name."""
        return re.sub(r'[\\:*?"<>|]', '', name).strip()

    # ------------------------------------------------------------------
    # Download  (returns images split by category)
    # ------------------------------------------------------------------
    def download_images_from_s3(self, temp_dir, pod_id):
        """
        Fetch all images assigned to pod_id (processed_flag = 'I').
        Images are split into two lists by category_id:
            category_id = 2  →  yolo_image_paths
            category_id = 3  →  freshness_image_paths

        Returns
        -------
        yolo_image_paths : list of 8-tuples
            (filesequenceid, storename, clean_filename, local_path,
             s3_key, storeid, subcategory_id, captured_timestamp)
        freshness_image_paths : list of 8-tuples  (same structure)
        failed_files : list of 3-tuples  (filesequenceid, storename, filename)
        """
        conn, cur = initialize_db_connection(self.db_config)
        try:
            cur.execute("""
                SELECT filesequenceid,
                       storename,
                       filename,
                       storeid,
                       subcategory_id,
                       category_id,
                       captured_timestamp
                FROM tbco.file_upload
                WHERE processed_flag = 'I'
                  AND podid = %s
                  AND category_id IN (%s, %s)
                ORDER BY storeid, uploadtimestamp ASC
            """, (pod_id, CATEGORY_YOLO, CATEGORY_FRESHNESS))
            rows = cur.fetchall()
        finally:
            close_db_connection(conn, cur)

        total = len(rows)
        logger.info(f"Found {total} images (cat 2+3) for pod {pod_id}")
        if total == 0:
            return [], [], []

        yolo_image_paths      = []
        freshness_image_paths = []
        failed_files          = []

        for idx, (fid, storename, filename, storeid, subcat_id,
                  category_id, captured_ts) in enumerate(rows, 1):
            try:
                s3_key      = filename
                clean_fname = self._clean(os.path.basename(filename))
                local_path  = os.path.join(temp_dir, clean_fname)

                try:
                    self.s3.download_file(self.bucket_name, s3_key, local_path)
                    record = (fid, storename, clean_fname,
                              local_path, s3_key, storeid, subcat_id, captured_ts)

                    if category_id == CATEGORY_YOLO:
                        yolo_image_paths.append(record)
                    elif category_id == CATEGORY_FRESHNESS:
                        freshness_image_paths.append(record)

                    if idx % 10 == 0 or idx == total:
                        logger.info(f"  Downloaded {idx}/{total}")

                except ClientError as e:
                    code = e.response['Error']['Code']
                    logger.error(f"S3 error [{code}] for {s3_key}: {e}")
                    failed_files.append((fid, storename, filename))

            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                failed_files.append((fid, storename, filename))

        logger.info(
            f"Download done: {len(yolo_image_paths)} YOLO, "
            f"{len(freshness_image_paths)} Freshness, "
            f"{len(failed_files)} failed"
        )
        return yolo_image_paths, freshness_image_paths, failed_files

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_file(self, local_path, s3_key):
        """Upload a single file to S3."""
        try:
            content_type, _ = mimetypes.guess_type(local_path)
            extra = {'ContentType': content_type} if content_type else {}
            self.s3.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra)
            logger.info(f"Uploaded -> s3://{self.bucket_name}/{s3_key}")
            return f"s3://{self.bucket_name}/{s3_key}"
        except NoCredentialsError:
            logger.error("Invalid AWS credentials")
            raise
        except ClientError as e:
            logger.error(f"Upload failed for {local_path}: {e}")
            raise

    def get_results_s3_key(self, filename, iterationid):
        """Build the S3 key for a result/annotated image.

        Creates a subfolder per iteration inside model_results/:
            model_results/iteration_<iterationid>/<filename>
        """
        return f"{self.results_folder}iteration_{iterationid}/{os.path.basename(filename)}"