"""
freshness_runner.py  –  Extract MFD (manufacture date) from product images
                         using EasyOCR and insert into tbco.product_freshness.

Only processes images with category_id = 3.

Input image tuple structure (8-tuple from s3_handler):
    (filesequenceid, storename, clean_filename, local_path,
     s3_key, storeid, subcategory_id, captured_timestamp)
"""

import re
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Lazy imports – only needed at runtime
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed. Freshness OCR will be skipped.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Freshness OCR will be skipped.")

# Matches dd/mm/yy or dd/mm/yyyy
_DATE_PATTERN = re.compile(r'\b(\d{2})[/\-.](\d{2})[/\-.](\d{2,4})\b')

# Shared reader instance (initialised once per process)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        logger.info("Initialising EasyOCR reader (English)…")
        _reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR reader ready.")
    return _reader


def _parse_date(day_s, month_s, year_s):
    """Return a datetime or None given string day / month / year parts."""
    try:
        day   = int(day_s)
        month = int(month_s)
        year  = int(year_s)
        if year < 100:                  # 2-digit year: assume 2000s
            year += 2000
        return datetime(year, month, day)
    except (ValueError, OverflowError):
        return None


def extract_mfd_from_image(image_path):
    """
    Run OCR on image_path and return the first date found as a datetime,
    or None if no valid date is detected.
    """
    if not EASYOCR_AVAILABLE or not CV2_AVAILABLE:
        logger.error("EasyOCR or OpenCV not available – cannot run freshness OCR.")
        return None

    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        reader  = _get_reader()
        results = reader.readtext(image)

        for (_bbox, text, _conf) in results:
            match = _DATE_PATTERN.search(text)
            if match:
                d, m, y = match.group(1), match.group(2), match.group(3)
                parsed  = _parse_date(d, m, y)
                if parsed:
                    logger.info(f"  MFD found in '{text}' → {parsed.date()}")
                    return parsed

        logger.info(f"  No MFD date found in {image_path}")
        return None

    except Exception as e:
        logger.error(f"OCR error on {image_path}: {e}")
        return None


def run_freshness_on_images(freshness_image_paths, iterationid=None):
    """
    Run OCR-based freshness extraction on a list of category-3 images.

    Parameters
    ----------
    freshness_image_paths : list of 8-tuples
        (filesequenceid, storename, clean_filename, local_path,
         s3_key, storeid, subcategory_id, captured_timestamp)
    iterationid : int | None
        The pipeline-level iteration ID to tag each freshness record with.

    Returns
    -------
    list of dicts, one per image:
        {
          filesequenceid, storeid, clean_filename,
          captured_timestamp, mfg_date   (datetime | None),
          iterationid
        }
    """
    results = []

    for (fid, storename, clean_fname, local_path,
         s3_key, storeid, subcat_id, captured_ts, upload_ts) in freshness_image_paths:

        logger.info(f"Freshness OCR: {clean_fname}")
        mfg_date = extract_mfd_from_image(local_path)

        results.append({
            'filesequenceid':     fid,
            'storeid':            storeid,
            'clean_filename':     clean_fname,
            'capture_date':       upload_ts,       # uploadtimestamp from file_upload
            'captured_timestamp': captured_ts,
            'mfg_date':           mfg_date,
            'iterationid':        iterationid,
        })

    logger.info(
        f"Freshness OCR done. "
        f"{sum(1 for r in results if r['mfg_date'])} dates found "
        f"out of {len(results)} images."
    )
    return results


def upload_freshness_to_db(conn, cur, freshness_results):
    """
    Insert freshness records into tbco.product_freshness.

    Only rows where mfg_date is not None are inserted.

    Parameters
    ----------
    conn, cur         : active pg8000 connection / cursor
    freshness_results : list of dicts from run_freshness_on_images()

    Returns
    -------
    int  – number of rows inserted
    """
    insert_sql = """
        INSERT INTO tbco.product_freshness
            (iteration_id, store_id, capture_date, mfg_date, image_file_name)
        VALUES (%s, %s, %s, %s, %s)
    """

    inserted = 0
    skipped  = 0

    for rec in freshness_results:
        if rec['mfg_date'] is None:
            logger.info(
                f"  Skipping {rec['clean_filename']} – no MFD date detected."
            )
            skipped += 1
            continue

        try:
            cur.execute(insert_sql, (
                rec['iterationid'],
                rec['storeid'],
                rec['capture_date'],         # uploadtimestamp from file_upload
                rec['mfg_date'],
                rec['clean_filename'],
            ))
            inserted += 1
            logger.info(
                f"  Inserted freshness: store={rec['storeid']} "
                f"mfg={rec['mfg_date'].date()} file={rec['clean_filename']}"
            )
        except Exception as e:
            logger.error(
                f"  Failed to insert freshness for {rec['clean_filename']}: {e}"
            )
            conn.rollback()
            continue

    conn.commit()
    logger.info(
        f"Freshness upload done: {inserted} inserted, {skipped} skipped (no date)."
    )
    return inserted