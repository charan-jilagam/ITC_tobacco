"""
yolo_runner.py  –  Run YOLO detection on a list of images and return
                   structured detection records ready for DB insertion.
"""
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy import – ultralytics is only needed at runtime
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. YOLO inference will be skipped.")


def run_yolo_on_images(
    image_paths,
    model_path,
    conf_threshold=0.3,
    output_folder='outputs/annotated/',
    s3_handler=None,
    s3_output_folder='model_results/',
    iterationid=1
):
    """
    Run YOLO on every image in image_paths.

    Parameters
    ----------
    image_paths : list of 7-tuples
        (filesequenceid, storename, clean_filename, local_path,
         s3_key, storeid, subcategory_id)
    model_path : str  – path to .pt weights file
    conf_threshold : float
    output_folder : str  – local folder for annotated images
    s3_handler : S3Handler | None  – if provided, annotated images are uploaded
    s3_output_folder : str  – S3 prefix for annotated images

    Returns
    -------
    list of dicts, one per bounding-box detection:
        {
          imagefilename, local_path, s3path_actual_file,
          s3path_annotated_file, productclassid,
          x1, y1, x2, y2, confidence, modelrun
        }
    """
    os.makedirs(output_folder, exist_ok=True)

    if not YOLO_AVAILABLE:
        logger.error("YOLO not available – returning empty results.")
        return []

    if not os.path.exists(model_path):
        logger.error(f"Model weights not found: {model_path}")
        return []

    logger.info(f"Loading YOLO model from {model_path}")
    model    = YOLO(model_path)
    modelrun = datetime.utcnow()
    results  = []

    for fid, storename, clean_fname, local_path, s3_key, storeid, subcat_id, *_ in image_paths:
        if not os.path.exists(local_path):
            logger.warning(f"Image not found locally: {local_path}")
            continue

        try:
            preds = model.predict(
                source=local_path,
                conf=conf_threshold,
                save=False,
                verbose=False
            )

            # Save annotated image locally
            annotated_name = f"annotated_{clean_fname}"
            annotated_path = os.path.join(output_folder, annotated_name)
            for result in preds:
                img = result.plot()          # returns BGR numpy array
                import cv2
                cv2.imwrite(annotated_path, img)
                break                        # one image per predict call

            # Upload annotated image to S3 under iteration subfolder
            s3_annotated_path = ''
            if s3_handler and os.path.exists(annotated_path):
                # Uses model_results/iteration_<iterationid>/<filename>
                s3_key_annotated = s3_handler.get_results_s3_key(
                    annotated_name, iterationid
                )
                try:
                    s3_annotated_path = s3_handler.upload_file(
                        annotated_path, s3_key_annotated
                    )
                except Exception as e:
                    logger.warning(f"Could not upload annotated image: {e}")

            # Parse detections
            for result in preds:
                if result.boxes is None or len(result.boxes) == 0:
                    # Insert a "no detection" record so every image is recorded
                    results.append(_make_record(
                        clean_fname, local_path, s3_key, s3_annotated_path,
                        productclassid=0,
                        x1=0, y1=0, x2=0, y2=0,
                        confidence=0.0,
                        modelrun=modelrun
                    ))
                    continue

                boxes = result.boxes
                for i in range(len(boxes)):
                    xyxy  = boxes.xyxy[i].tolist()
                    conf  = float(boxes.conf[i])
                    cls   = int(boxes.cls[i])
                    results.append(_make_record(
                        clean_fname, local_path, s3_key, s3_annotated_path,
                        productclassid=cls,
                        x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                        confidence=conf,
                        modelrun=modelrun
                    ))

            logger.info(f"  {clean_fname}: {len(preds[0].boxes) if preds[0].boxes else 0} detections")

        except Exception as e:
            logger.error(f"YOLO failed on {clean_fname}: {e}")

    logger.info(f"YOLO done. Total detection records: {len(results)}")
    return results


def _make_record(imagefilename, local_path, s3_actual, s3_annotated,
                 productclassid, x1, y1, x2, y2, confidence, modelrun):
    return {
        'imagefilename':       imagefilename,
        'local_path':          local_path,
        's3path_actual_file':  s3_actual,
        's3path_annotated_file': s3_annotated,
        'productclassid':      productclassid,
        'x1':                  round(x1, 4),
        'y1':                  round(y1, 4),
        'x2':                  round(x2, 4),
        'y2':                  round(y2, 4),
        'confidence':          round(confidence, 4),
        'modelrun':            modelrun,
    }