"""
Tobacco Products - YOLO Detection + Freshness Pipeline
=======================================================
Driver script:
    • Fetches images from tbco.file_upload
    • Batch unit = STORE (user picks how many stores per batch)
      All images belonging to the selected stores are processed together.
    • category_id = 2  →  YOLO detection
          results → tbco.product_count_master + tbco.product_count_transaction
          annotated images uploaded to S3 model_results/
    • category_id = 3  →  Freshness OCR
          results → tbco.product_freshness

    iterationid is fixed for the ENTIRE pipeline run (set once at startup).

Usage:
    python main.py <pod-id>
    python main.py pod-1
"""

import sys
import os
import logging
import tempfile
import traceback
import time

from config_loader import load_config
from db_handler import initialize_db_connection, close_db_connection
from s3_handler import S3Handler
from yolo_runner import run_yolo_on_images
from result_uploader import upload_results_to_db
from freshness_runner import run_freshness_on_images, upload_freshness_to_db

os.makedirs('outputs', exist_ok=True)

# UTF-8 handlers — prevents UnicodeEncodeError on Windows (cp1252 console)
_file_handler   = logging.FileHandler('outputs/pipeline.log', encoding='utf-8')
_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[_file_handler, _stream_handler]
)
logger = logging.getLogger(__name__)

STALE_TIMEOUT_MINUTES = 60


# ---------------------------------------------------------------------------
# Store-based helpers
# ---------------------------------------------------------------------------

def get_unprocessed_store_count(conn):
    """
    Count distinct stores that still have unprocessed images
    (processed_flag = 'P' or NULL) for category 2 or 3.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT storeid)
            FROM tbco.file_upload
            WHERE (processed_flag = 'P' OR processed_flag IS NULL)
              AND category_id IN (2, 3)
        """)
        count = cur.fetchone()[0]
        cur.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get unprocessed store count: {e}")
        return 0


def get_unprocessed_store_summary(conn):
    """
    Return list of (storeid, storename, image_count) for all stores
    that still have unprocessed images, ordered by oldest upload first.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT storeid,
                   MAX(storename)   AS storename,
                   COUNT(*)         AS image_count
            FROM tbco.file_upload
            WHERE (processed_flag = 'P' OR processed_flag IS NULL)
              AND category_id IN (2, 3)
            GROUP BY storeid
            ORDER BY MIN(uploadtimestamp) ASC
        """)
        rows = cur.fetchall()
        cur.close()
        return rows          # [(storeid, storename, image_count), ...]
    except Exception as e:
        logger.error(f"Failed to get store summary: {e}")
        return []


def reset_stale_assignments(conn, timeout_minutes):
    """Reset images stuck in 'I' state beyond timeout back to 'P'."""
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE tbco.file_upload
            SET processed_flag = 'P', podid = NULL
            WHERE processed_flag = 'I'
              AND uploadtimestamp < NOW() - INTERVAL '%s minutes'
        """ % timeout_minutes)
        reset_count = cur.rowcount
        conn.commit()
        cur.close()
        if reset_count > 0:
            logger.warning(
                f"Reset {reset_count} stale images (stuck > {timeout_minutes} min)"
            )
        return reset_count
    except Exception as e:
        logger.error(f"Failed to reset stale assignments: {e}")
        conn.rollback()
        return 0


def assign_stores_to_pod(conn, store_batch_size, pod_id):
    """
    Pick the next `store_batch_size` stores (by oldest upload) and mark
    ALL their unprocessed images as 'I', assigned to pod_id.

    Returns
    -------
    (assigned_image_count, store_ids_assigned)
    """
    try:
        cur = conn.cursor()

        # Get the next N stores with unprocessed images
        cur.execute("""
            SELECT storeid
            FROM tbco.file_upload
            WHERE (processed_flag = 'P' OR processed_flag IS NULL)
              AND category_id IN (2, 3)
            GROUP BY storeid
            ORDER BY MIN(uploadtimestamp) ASC
            LIMIT %s
        """, (store_batch_size,))

        store_ids = [row[0] for row in cur.fetchall()]
        if not store_ids:
            cur.close()
            return 0, []

        # Assign ALL images of those stores to this pod
        cur.execute("""
            UPDATE tbco.file_upload
            SET processed_flag = 'I', podid = %s
            WHERE (processed_flag = 'P' OR processed_flag IS NULL)
              AND category_id IN (2, 3)
              AND storeid = ANY(%s)
        """, (pod_id, store_ids))

        assigned_images = cur.rowcount
        conn.commit()
        cur.close()

        logger.info(
            f"Assigned {assigned_images} images from "
            f"{len(store_ids)} stores to {pod_id} | "
            f"stores: {store_ids}"
        )
        return assigned_images, store_ids

    except Exception as e:
        logger.error(f"Failed to assign stores to pod: {e}")
        conn.rollback()
        return 0, []


def get_next_iteration_ids(conn):
    """
    Get next iterationid and starting iterationtranid from product_count_master.
    Called ONCE at pipeline startup — both values are fixed for the entire run.
    """
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(MAX(iterationid), 0) FROM tbco.product_count_master"
        )
        max_iter = cur.fetchone()[0]
        cur.execute(
            "SELECT COALESCE(MAX(iterationtranid), 0) FROM tbco.product_count_master"
        )
        max_tran = cur.fetchone()[0]
        cur.close()
        return max_iter + 1, max_tran + 1
    except Exception as e:
        logger.error(f"Failed to get iteration IDs: {e}")
        return 1, 1


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_batch(pod_id, iterationid, iterationtranid, config):
    """
    Download images for pod_id → split by category →
        cat 2: YOLO  → product_count_master / product_count_transaction
        cat 3: OCR   → product_freshness
    → mark all processed 'Y'.

    Returns (success: bool, next_iterationtranid: int)
    iterationid is NEVER modified here — it belongs to the whole run.
    """
    conn = None
    cur  = None
    try:
        db_config   = config['db_config']
        s3_config   = config['s3_config']
        yolo_config = config['yolo_config']

        conn, cur = initialize_db_connection(db_config)
        s3_handler = S3Handler(s3_config, db_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Temp dir: {temp_dir}")

            # --------------------------------------------------------
            # 1. Download images — split into YOLO (cat 2) and
            #    Freshness (cat 3) lists
            # --------------------------------------------------------
            yolo_image_paths, freshness_image_paths, failed_files = \
                s3_handler.download_images_from_s3(temp_dir, pod_id)

            if not yolo_image_paths and not freshness_image_paths:
                logger.warning(f"No images downloaded for {pod_id}")
                return False, iterationtranid

            logger.info(
                f"Downloaded: {len(yolo_image_paths)} YOLO images, "
                f"{len(freshness_image_paths)} Freshness images, "
                f"{len(failed_files)} failed"
            )

            new_iterationtranid = iterationtranid

            # --------------------------------------------------------
            # 2a. YOLO pipeline (category_id = 2)
            # --------------------------------------------------------
            if yolo_image_paths:
                logger.info("--- Running YOLO on category-2 images ---")
                detection_results = run_yolo_on_images(
                    image_paths=yolo_image_paths,
                    model_path=yolo_config['model_path'],
                    conf_threshold=yolo_config.get('conf_threshold', 0.3),
                    output_folder=yolo_config.get(
                        'annotated_output_folder', 'outputs/annotated/'
                    ),
                    s3_handler=s3_handler,
                    s3_output_folder=s3_config.get(
                        'results_folder_s3', 'model_results/'
                    ),
                    iterationid=iterationid       # same iterationid for whole run
                )
                logger.info(
                    f"YOLO complete: {len(detection_results)} detection records"
                )

                image_meta = {
                    clean_fname: (storeid, subcat_id)
                    for _, _, clean_fname, _, _, storeid, subcat_id, _
                    in yolo_image_paths
                }
                image_order = [
                    clean_fname
                    for _, _, clean_fname, _, _, _, _, _ in yolo_image_paths
                ]

                new_iterationtranid = upload_results_to_db(
                    conn=conn,
                    cur=cur,
                    iterationid=iterationid,
                    iterationtranid_start=iterationtranid,
                    image_meta=image_meta,
                    image_order=image_order,
                    detection_results=detection_results
                )
            else:
                logger.info("No category-2 images in this batch — skipping YOLO.")

            # --------------------------------------------------------
            # 2b. Freshness pipeline (category_id = 3)
            # --------------------------------------------------------
            if freshness_image_paths:
                logger.info("--- Running Freshness OCR on category-3 images ---")
                freshness_results = run_freshness_on_images(freshness_image_paths, iterationid=iterationid)
                upload_freshness_to_db(conn, cur, freshness_results)
            else:
                logger.info("No category-3 images in this batch — skipping Freshness.")

            # --------------------------------------------------------
            # 3. Mark all downloaded images as processed 'Y'
            # --------------------------------------------------------
            all_processed_ids = (
                [fp[0] for fp in yolo_image_paths]
                + [fp[0] for fp in freshness_image_paths]
            )
            if all_processed_ids:
                cur.execute("""
                    UPDATE tbco.file_upload
                    SET processed_flag = 'Y'
                    WHERE filesequenceid = ANY(%s)
                """, (all_processed_ids,))
                conn.commit()
                logger.info(
                    f"Marked {len(all_processed_ids)} images as processed (Y)"
                )

            # Reset any failed downloads back to 'P' for retry
            if failed_files:
                failed_ids = [f[0] for f in failed_files]
                cur.execute("""
                    UPDATE tbco.file_upload
                    SET processed_flag = 'P', podid = NULL
                    WHERE filesequenceid = ANY(%s)
                """, (failed_ids,))
                conn.commit()
                logger.warning(
                    f"Reset {len(failed_ids)} failed images to 'P'"
                )

        return True, new_iterationtranid

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.error(traceback.format_exc())
        if conn:
            conn.rollback()
        return False, iterationtranid

    finally:
        if conn:
            close_db_connection(conn, cur)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <pod-id>")
        logger.error("Example: python main.py pod-1")
        sys.exit(1)

    pod_id = sys.argv[1]
    logger.info(f"Starting tobacco pipeline for pod: {pod_id}")

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/annotated', exist_ok=True)

    config    = load_config('config.json')
    db_config = config['db_config']

    # ------------------------------------------------------------------
    # Get iteration IDs ONCE — iterationid is fixed for the entire run.
    # iterationtranid advances per image across all batches.
    # ------------------------------------------------------------------
    conn, cur = initialize_db_connection(db_config)
    iterationid, iterationtranid = get_next_iteration_ids(conn)
    close_db_connection(conn, cur)

    logger.info("=" * 60)
    logger.info(
        f"Pipeline Run | pod={pod_id} | "
        f"iterationid={iterationid} (fixed for this run) | "
        f"iterationtranid starts at {iterationtranid}"
    )
    logger.info("=" * 60)

    store_batch_size = None   # asked once, reused for every batch
    batch_number     = 0

    while True:
        try:
            # --------------------------------------------------------
            # Check remaining work — in STORES not images
            # --------------------------------------------------------
            conn, cur = initialize_db_connection(db_config)
            reset_stale_assignments(conn, STALE_TIMEOUT_MINUTES)
            remaining_stores = get_unprocessed_store_count(conn)

            if remaining_stores == 0:
                close_db_connection(conn, cur)
                logger.info("=" * 60)
                logger.info(
                    f"All stores processed. "
                    f"Total batches: {batch_number} | "
                    f"iterationid used: {iterationid}"
                )
                logger.info("=" * 60)
                break

            # Show a per-store breakdown for visibility
            store_summary = get_unprocessed_store_summary(conn)
            close_db_connection(conn, cur)

            logger.info(
                f"Unprocessed stores remaining: {remaining_stores}"
            )
            logger.info("  Store breakdown:")
            for storeid, storename, img_count in store_summary:
                logger.info(
                    f"    storeid={storeid}  name={storename}  images={img_count}"
                )

            # --------------------------------------------------------
            # Ask batch size in stores (only once per run)
            # --------------------------------------------------------
            if store_batch_size is None:
                while True:
                    try:
                        raw = input(
                            f"\nEnter number of stores to process per batch "
                            f"(1-{remaining_stores}): "
                        ).strip()
                        store_batch_size = int(raw)
                        if 1 <= store_batch_size <= remaining_stores:
                            break
                        print(
                            f"Please enter a number between 1 "
                            f"and {remaining_stores}"
                        )
                    except ValueError:
                        print("Please enter a valid integer.")

            # --------------------------------------------------------
            # Assign next N stores to this pod
            # --------------------------------------------------------
            conn, cur = initialize_db_connection(db_config)
            assigned_images, assigned_store_ids = assign_stores_to_pod(
                conn, store_batch_size, pod_id
            )
            close_db_connection(conn, cur)

            if assigned_images == 0:
                logger.warning("No images assigned. Retrying in 10s…")
                time.sleep(10)
                continue

            batch_number += 1
            logger.info(
                f"--- Batch {batch_number} | "
                f"stores={len(assigned_store_ids)} | "
                f"images={assigned_images} | "
                f"iterationid={iterationid} ---"
            )

            # --------------------------------------------------------
            # Process the batch
            # --------------------------------------------------------
            success, new_iterationtranid = process_batch(
                pod_id, iterationid, iterationtranid, config
            )

            if success:
                logger.info(
                    f"Batch {batch_number} completed successfully | "
                    f"iterationtranid advanced "
                    f"{iterationtranid} → {new_iterationtranid - 1}"
                )
                iterationtranid = new_iterationtranid
                time.sleep(2)
            else:
                logger.error(
                    f"Batch {batch_number} failed. Retrying in 10s…"
                )
                time.sleep(10)
                continue

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            logger.error(traceback.format_exc())
            time.sleep(10)

    logger.info(f"Pipeline finished for {pod_id}")


if __name__ == "__main__":
    main()