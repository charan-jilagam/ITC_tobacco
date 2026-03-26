"""
result_uploader.py  -  Insert YOLO detection results into:
    tbco.product_count_master       (one row per IMAGE — each image gets its own iterationtranid)
    tbco.product_count_transaction  (one row per bounding-box detection, productsequenceno restarts per image)

Schema constraints honoured:
    product_count_master      PK  (iterationid, iterationtranid)
    product_count_transaction PK  (iterationid, iterationtranid, productsequenceno)
    product_count_transaction FK  -> product_count_master(iterationid, iterationtranid)
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def upload_results_to_db(conn, cur, iterationid, iterationtranid_start,
                         image_meta, image_order, detection_results):
    """
    Parameters
    ----------
    conn, cur              : active pg8000 connection / cursor
    iterationid            : int  - shared across the full pipeline run
    iterationtranid_start  : int  - starting iterationtranid for this batch;
                             incremented by 1 for each image
    image_meta             : dict  { clean_filename -> (storeid, subcategory_id) }
    image_order            : list of clean_filenames in processing order
                             (determines which iterationtranid each image gets)
    detection_results      : list of dicts from yolo_runner.run_yolo_on_images()

    Returns
    -------
    int  - next available iterationtranid (iterationtranid_start + len(images))
    """
    try:
        now = datetime.utcnow()

        # ----------------------------------------------------------------
        # Assign iterationtranid to each image in order
        # ----------------------------------------------------------------
        image_to_tranid = {
            fname: iterationtranid_start + i
            for i, fname in enumerate(image_order)
        }
        next_iterationtranid = iterationtranid_start + len(image_order)

        # ----------------------------------------------------------------
        # 1. Group detections by image filename
        # ----------------------------------------------------------------
        detections_by_image = {}
        for det in detection_results:
            fname = det.get('imagefilename', '')
            detections_by_image.setdefault(fname, []).append(det)

        # ----------------------------------------------------------------
        # 2. Insert one product_count_master row per image
        # ----------------------------------------------------------------
        for fname in image_order:
            tranid  = image_to_tranid[fname]
            storeid = image_meta.get(fname, (0, None))[0]

            cur.execute("""
                INSERT INTO tbco.product_count_master
                    (iterationid, iterationtranid, storeid, modelrun, processed_flag)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (iterationid, iterationtranid) DO NOTHING
            """, (iterationid, tranid, storeid, now, 'N'))

            logger.info(
                f"  Master row: iterationid={iterationid}"
                f"  iterationtranid={tranid}  storeid={storeid}"
                f"  image={fname}"
            )

        conn.commit()
        logger.info(
            f"Inserted {len(image_order)} master rows "
            f"(iterationtranid {iterationtranid_start} - {next_iterationtranid - 1})"
        )

        # ----------------------------------------------------------------
        # 3. Insert transaction rows
        #    productsequenceno restarts at 1 for each image
        # ----------------------------------------------------------------
        insert_sql = """
            INSERT INTO tbco.product_count_transaction
                (iterationid, iterationtranid, productsequenceno,
                 productclassid, x1, y1, x2, y2, confidence,
                 imagefilename, s3path_actual_file, s3path_annotated_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        total_rows = 0
        for fname in image_order:
            tranid = image_to_tranid[fname]
            dets   = detections_by_image.get(fname, [])

            if not dets:
                # No detections for this image — insert a zero-detection record
                cur.execute(insert_sql, (
                    iterationid, tranid, 1,
                    0, 0, 0, 0, 0, 0.0,
                    fname, '', ''
                ))
                total_rows += 1
                continue

            rows = []
            for seq_no, det in enumerate(dets, start=1):
                rows.append((
                    iterationid,
                    tranid,
                    seq_no,
                    det.get('productclassid', 0),
                    det.get('x1', 0),
                    det.get('y1', 0),
                    det.get('x2', 0),
                    det.get('y2', 0),
                    det.get('confidence', 0.0),
                    fname,
                    det.get('s3path_actual_file', ''),
                    det.get('s3path_annotated_file', '')
                ))

            cur.executemany(insert_sql, rows)
            total_rows += len(rows)

        conn.commit()
        logger.info(
            f"Inserted {total_rows} detection rows into product_count_transaction"
        )

        # ----------------------------------------------------------------
        # 4. Mark all master rows as processed
        # ----------------------------------------------------------------
        tranid_list = list(image_to_tranid.values())
        cur.execute("""
            UPDATE tbco.product_count_master
            SET processed_flag = 'Y'
            WHERE iterationid = %s
              AND iterationtranid = ANY(%s)
        """, (iterationid, tranid_list))
        conn.commit()
        logger.info(
            f"Marked {len(tranid_list)} product_count_master rows processed_flag = 'Y'"
        )

        return next_iterationtranid

    except Exception as e:
        logger.error(f"Failed to upload results to DB: {e}")
        conn.rollback()
        raise