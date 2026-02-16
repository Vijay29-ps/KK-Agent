import os
import cv2
import time
import uuid
import math
import json
import re
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple, List

from ultralytics import YOLO
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient


# ==========================================================
# CONFIG (Server defaults)
# You can later move these to ENV variables for production
# ==========================================================
YOLO_TRACK_MODEL = "yolov8n.pt"
YOLO_CONF = 0.35
TRACK_CLASSES = [0, 1, 2, 3, 5, 7]

ROBOFLOW_API_KEY = os.getenv("ROBOBOFLOW_API_KEY", "cpaeqLTV1297wQM8mZbx")

# Snatching RF (SDK)
SNATCH_WORKSPACE = "sunfibo"
SNATCH_PROJECT = "chain_snatching-qlv3z"
SNATCH_VERSION = 3
SNATCH_RF_CONF = 0.50
SNATCH_RF_COOLDOWN_FRAMES = 25

ACC_THRESHOLD = 1.8
HISTORY = 6
MAX_PAIR_DIST_PX = 350
MAX_MOTO_DIST_PX = 520
REQUIRE_VEHICLE_NEAR = True

# Fight RF (Serverless)
FIGHT_MODEL_ID = "fight-9uyg7/1"
FIGHT_RF_CONF = 0.65
FIGHT_RF_FRAME_SKIP = 5
FIGHT_MIN_AREA_RATIO = 0.02
FIGHT_MAX_AREA_RATIO = 0.60
FIGHT_MIN_PERSON_IOU = 0.02
FIGHT_PERSIST_HITS = 3

# Weapon (optional)
ENABLE_WEAPON = False
WEAPON_MODEL_PATH = "All_weapon (1).pt"
WEAPON_CONF = 0.35

# Incident
ALERT_FRAMES = 40
EVENT_COOLDOWN_SEC = 5.0
INCIDENT_SPAM_GUARD_SEC = 2.0

# FIR OLLAMA
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama-sales.mobiusdtaas.ai/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
STATION_NAME = os.getenv("STATION_NAME", "CCTV ANALYTICS DESK")
OFFICER_DESIGNATION = os.getenv("OFFICER_DESIGNATION", "Duty Officer")


# ==========================================================
# FIR helpers
# ==========================================================
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _is_single_paragraph(text: str) -> bool:
    return "\n" not in text.strip()

def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"\.\s*", text.strip()) if s])

def _validate_polish(original: str, rewritten: str) -> bool:
    if not rewritten or len(rewritten) < 120:
        return False
    if not _is_single_paragraph(rewritten):
        return False
    if re.search(r"^\s*(subject|heading|report|fir|information report)\s*[:\-]", rewritten.strip(), re.I):
        return False
    if re.search(r"[\n\r]\s*[-•]\s+", rewritten):
        return False
    if _sentence_count(original) != _sentence_count(rewritten):
        return False

    forbidden_additions = [
        "section", "ipc", "crpc", "motor vehicles act", "mva", "violation",
        "speed", "km/h", "helmet", "fatal", "death", "injury",
        "accused", "complainant", "witness", "arrest"
    ]
    orig_l = original.lower()
    rew_l = rewritten.lower()
    for w in forbidden_additions:
        if w in rew_l and w not in orig_l:
            return False

    return True

def ollama_polish_fir(base_fir: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a formal Indian Police FIR language editor. "
        "Rewrite the given paragraph in formal FIR style. "
        "Do NOT change meaning. "
        "Do NOT change sentence order. "
        "Do NOT add or remove facts. "
        "Output exactly one paragraph."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_fir},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    prompt_hash = _sha256_text(base_fir)
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content", "").strip()

        if not _validate_polish(base_fir, text):
            raise ValueError("LLM output failed validation")

        return {
            "narration": text,
            "confidence": 0.96,
            "prompt_hash": prompt_hash,
            "output_hash": _sha256_text(text),
            "generated_at_utc": generated_at,
            "fallback_used": False,
        }
    except Exception:
        return {
            "narration": base_fir,
            "confidence": 0.90,
            "prompt_hash": prompt_hash,
            "output_hash": _sha256_text(base_fir),
            "generated_at_utc": generated_at,
            "fallback_used": True,
        }

def build_fixed_report_header(station_name: str, run_id: str) -> str:
    return (
        "INFORMATION REPORT\n"
        f"{station_name}\n\n"
        f"Subject: Incident Report generated via Automated CCTV Analysis System (Ref: {run_id})\n\n"
    )

def hash_report_short(report_text: str) -> str:
    h = _sha256_text(report_text)
    return f"{h[:12]}...{h[-12:]}"

def build_fixed_footer(officer_designation: str, local_date_ddmmyyyy: str, report_hash_short: str) -> str:
    return (
        f"\n\n{officer_designation}\n\n"
        f"Generated: {local_date_ddmmyyyy}\n"
        f"Hash: {report_hash_short}\n"
    )

def build_base_fir_one_paragraph(event_type: str,
                                 video_name: str,
                                 frame_id: int,
                                 offender_tid: int,
                                 victim_tid: Optional[int],
                                 evidence_dir: str,
                                 utc_iso: str) -> str:
    victim_part = f" and a victim track ID {victim_tid}" if victim_tid is not None else ""
    return (
        f"During automated CCTV analysis of video {video_name}, an incident classified as {event_type} was detected at frame {frame_id} on UTC time {utc_iso}. "
        f"The system associated the incident with an offender track ID {offender_tid}{victim_part} based on visual tracking and model inference outputs. "
        f"Forensic evidence images were generated and stored in folder {evidence_dir} for review and further action."
    )

def write_fir_report(run_id: str, event_row: Dict[str, Any]) -> Dict[str, Any]:
    utc_iso = event_row["utc_timestamp"]
    victim_tid = event_row.get("victim_tid", None)
    if victim_tid in ("", None, "None"):
        victim_tid_val = None
    else:
        victim_tid_val = int(victim_tid)

    base_fir = build_base_fir_one_paragraph(
        event_type=event_row["event"],
        video_name=Path(event_row["video_path"]).name,
        frame_id=int(event_row["frame_id"]),
        offender_tid=int(event_row["offender_tid"]),
        victim_tid=victim_tid_val,
        evidence_dir=str(Path(event_row["full_frame_path"]).parent),
        utc_iso=utc_iso
    )

    polished = ollama_polish_fir(base_fir)

    header = build_fixed_report_header(station_name=STATION_NAME, run_id=run_id)
    local_date = datetime.now().strftime("%d/%m/%Y")
    body = polished["narration"]
    report_text = header + body
    report_hash_short = hash_report_short(report_text)
    footer = build_fixed_footer(OFFICER_DESIGNATION, local_date, report_hash_short)
    final_text = report_text + footer

    meta = {
        "run_id": run_id,
        "station_name": STATION_NAME,
        "officer_designation": OFFICER_DESIGNATION,
        "event_row": event_row,
        "ollama": polished,
        "report_hash_full": _sha256_text(final_text),
        "report_hash_short": report_hash_short,
        "generated_local": datetime.now().isoformat(),
    }
    return {"text": final_text, "meta": meta}


# ==========================================================
# CV helpers
# ==========================================================
def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)

def save_crop(img, box, path: Path):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(path), crop)

def nearest_person(persons: List[Tuple[int, Tuple[int,int,int,int]]], from_tid: int) -> Optional[Tuple[int, float]]:
    from_box = None
    for tid, box in persons:
        if tid == from_tid:
            from_box = box
            break
    if from_box is None:
        return None
    fc = centroid(from_box)
    best_tid, best_d = None, float("inf")
    for tid, box in persons:
        if tid == from_tid:
            continue
        d = dist(fc, centroid(box))
        if d < best_d:
            best_d = d
            best_tid = tid
    if best_tid is None:
        return None
    return best_tid, best_d

def nearest_vehicle(vehicles: List[Tuple[int, Tuple[int,int,int,int], int]],
                    point: Tuple[int,int]) -> Optional[Tuple[Tuple[int,int,int,int], int, float]]:
    if not vehicles:
        return None
    preferred = [v for v in vehicles if v[2] in (3, 1)]
    pool = preferred if preferred else vehicles
    best = None
    best_d = float("inf")
    for _tid, box, cls in pool:
        d = dist(point, centroid(box))
        if d < best_d:
            best_d = d
            best = (box, cls, best_d)
    return best


# ==========================================================
# MAIN PIPELINE FUNCTION
# ==========================================================
def run_cctv_pipeline(video_path: str, run_id: str, run_dir: str) -> Dict[str, Any]:
    run_dir = Path(run_dir)

    tmp_dir = run_dir / "tmp"
    out_dir = run_dir / "output"
    evidence_dir = run_dir / "evidence"
    reports_dir = run_dir / "reports"

    for d in [tmp_dir, out_dir, evidence_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / "annotated.mp4"
    events_csv = out_dir / "events.csv"

    # models (load inside function so server can run independently)
    yolo_track = YOLO(YOLO_TRACK_MODEL)
    weapon_model = YOLO(WEAPON_MODEL_PATH) if ENABLE_WEAPON else None

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    snatch_project = rf.workspace(SNATCH_WORKSPACE).project(SNATCH_PROJECT)
    snatch_version = snatch_project.version(SNATCH_VERSION)
    snatch_model = snatch_version.model

    fight_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (W, H)
    )

    frame_id = 0
    last_event_time = 0.0
    last_incident_time = 0.0

    records: List[Dict[str, Any]] = []

    track_history = defaultdict(lambda: deque(maxlen=HISTORY))
    rf_checked_frame: Dict[int, int] = {}
    alert_state: Dict[int, int] = {}

    lock_event_label: Optional[str] = None
    lock_offender_tid: Optional[int] = None
    lock_victim_tid: Optional[int] = None
    lock_vehicle_box: Optional[Tuple[int,int,int,int]] = None

    fight_hits = 0
    fight_pair_persist: Optional[Tuple[int,int]] = None
    fight_box_persist: Optional[Tuple[int,int,int,int]] = None

    def log_row(row: Dict[str, Any]) -> None:
        records.append(row)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        now = time.time()

        # --- tracking ---
        results = yolo_track.track(
            frame,
            persist=True,
            conf=YOLO_CONF,
            classes=TRACK_CLASSES,
            verbose=False
        )

        persons: List[Tuple[int, Tuple[int,int,int,int]]] = []
        vehicles: List[Tuple[int, Tuple[int,int,int,int], int]] = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                tid_i = int(tid)
                cls_i = int(cls)
                if cls_i == 0:
                    persons.append((tid_i, (x1,y1,x2,y2)))
                else:
                    vehicles.append((tid_i, (x1,y1,x2,y2), cls_i))

        person_map = {tid: box for tid, box in persons}

        # --- fight gated ---
        if frame_id % FIGHT_RF_FRAME_SKIP == 0:
            try:
                rf_res = fight_client.infer(frame, model_id=FIGHT_MODEL_ID)
                preds = sorted(rf_res.get("predictions", []), key=lambda p: p.get("confidence", 0.0), reverse=True)

                best_box = None
                for p in preds:
                    conf = float(p.get("confidence", 0.0))
                    if conf < FIGHT_RF_CONF:
                        continue
                    x1 = int(p["x"] - p["width"] / 2)
                    y1 = int(p["y"] - p["height"] / 2)
                    x2 = int(p["x"] + p["width"] / 2)
                    y2 = int(p["y"] + p["height"] / 2)
                    x1, y1, x2, y2 = clamp_box(x1,y1,x2,y2,W,H)

                    area_ratio = ((x2-x1)*(y2-y1)) / float(W*H + 1e-9)
                    if area_ratio < FIGHT_MIN_AREA_RATIO or area_ratio > FIGHT_MAX_AREA_RATIO:
                        continue

                    best_box = (x1,y1,x2,y2)
                    break

                if best_box is not None and persons:
                    overlap_people = []
                    for tid, pbox in persons:
                        if iou(pbox, best_box) >= FIGHT_MIN_PERSON_IOU:
                            overlap_people.append((tid, pbox))
                    if len(overlap_people) >= 2:
                        overlap_people.sort(key=lambda tb: iou(tb[1], best_box), reverse=True)
                        fight_pair = (int(overlap_people[0][0]), int(overlap_people[1][0]))
                        fight_hits += 1
                        if fight_hits >= FIGHT_PERSIST_HITS:
                            fight_pair_persist = fight_pair
                            fight_box_persist = best_box
                        else:
                            fight_pair_persist = None
                            fight_box_persist = None
                    else:
                        fight_hits = 0
                        fight_pair_persist = None
                        fight_box_persist = None
                else:
                    fight_hits = 0
                    fight_pair_persist = None
                    fight_box_persist = None

            except Exception:
                fight_hits = 0
                fight_pair_persist = None
                fight_box_persist = None

        # --- snatching candidate + confirm ---
        snatch_candidate_tid = None
        snatch_victim_tid = None
        snatch_vehicle_box = None
        snatch_offender_box = None
        snatch_victim_box = None

        for tid, box in persons:
            cx, cy = centroid(box)
            track_history[tid].append((cx, cy))

            triggered = False
            if len(track_history[tid]) >= 3:
                pts = list(track_history[tid])
                (x0,y0), (x1p,y1p), (x2p,y2p) = pts[-3:]
                v1 = math.hypot(x1p-x0, y1p-y0)
                v2 = math.hypot(x2p-x1p, y2p-y1p)
                if abs(v2 - v1) > ACC_THRESHOLD:
                    triggered = True

            if not triggered:
                continue

            nv = nearest_person(persons, tid)
            if nv is None:
                continue
            victim_tid, pv_d = nv
            if pv_d > MAX_PAIR_DIST_PX:
                continue

            veh_pick = nearest_vehicle(vehicles, (cx, cy))
            if REQUIRE_VEHICLE_NEAR:
                if veh_pick is None:
                    continue
                vbox, _vcls, vd = veh_pick
                if vd > MAX_MOTO_DIST_PX:
                    continue
            else:
                vbox = veh_pick[0] if veh_pick else None

            snatch_candidate_tid = tid
            snatch_victim_tid = victim_tid
            snatch_vehicle_box = vbox
            snatch_offender_box = box
            snatch_victim_box = person_map.get(victim_tid)
            break

        confirmed_snatch = False
        if snatch_candidate_tid is not None and snatch_offender_box is not None:
            last_chk = rf_checked_frame.get(snatch_candidate_tid, -999999)
            if frame_id - last_chk > SNATCH_RF_COOLDOWN_FRAMES:
                x1,y1,x2,y2 = snatch_offender_box
                img_path = tmp_dir / f"{uuid.uuid4()}.jpg"
                cv2.imwrite(str(img_path), frame[y1:y2, x1:x2])

                try:
                    rf_json = snatch_model.predict(str(img_path), confidence=SNATCH_RF_CONF).json()
                    preds = rf_json.get("predictions", [])
                    if any(float(p.get("confidence", 0.0)) >= SNATCH_RF_CONF for p in preds):
                        confirmed_snatch = True
                except Exception:
                    confirmed_snatch = False

                rf_checked_frame[snatch_candidate_tid] = frame_id

        # --- event selection ---
        new_event_type = None
        offender_tid = None
        victim_tid = None
        offender_box = None
        victim_box = None
        vehicle_box = None

        if (now - last_event_time) > EVENT_COOLDOWN_SEC and (now - last_incident_time) > INCIDENT_SPAM_GUARD_SEC:
            if confirmed_snatch and snatch_candidate_tid is not None and snatch_offender_box is not None:
                new_event_type = "snatching"
                offender_tid = snatch_candidate_tid
                victim_tid = snatch_victim_tid
                offender_box = snatch_offender_box
                victim_box = snatch_victim_box
                vehicle_box = snatch_vehicle_box
            elif fight_pair_persist is not None:
                new_event_type = "fight"
                offender_tid = fight_pair_persist[0]
                victim_tid = fight_pair_persist[1]
                offender_box = person_map.get(offender_tid)
                victim_box = person_map.get(victim_tid)

        # --- incident artifacts ---
        if new_event_type and offender_tid is not None and offender_box is not None:
            incident_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            incident_dir = evidence_dir / f"{new_event_type}_incident_{incident_id}"
            incident_dir.mkdir(parents=True, exist_ok=True)

            full_frame_path = incident_dir / "full_frame.jpg"
            cv2.imwrite(str(full_frame_path), frame)

            offender_path = incident_dir / "offender.jpg"
            save_crop(frame, offender_box, offender_path)

            victim_path = None
            if victim_tid is not None and victim_box is not None:
                victim_path = incident_dir / "victim.jpg"
                save_crop(frame, victim_box, victim_path)

            vehicle_path = None
            if new_event_type == "snatching" and vehicle_box is not None:
                vehicle_path = incident_dir / "vehicle.jpg"
                save_crop(frame, vehicle_box, vehicle_path)

            # lock drawing
            alert_state[offender_tid] = ALERT_FRAMES
            if victim_tid is not None:
                alert_state[victim_tid] = ALERT_FRAMES

            lock_event_label = "SNATCHING" if new_event_type == "snatching" else "FIGHT"
            lock_offender_tid = offender_tid
            lock_victim_tid = victim_tid
            lock_vehicle_box = vehicle_box if new_event_type == "snatching" else None

            row = {
                "event": new_event_type,
                "frame_id": frame_id,
                "utc_timestamp": datetime.utcnow().isoformat(),
                "video_path": video_path,

                "offender_tid": offender_tid,
                "victim_tid": victim_tid if victim_tid is not None else "",

                "offender_box": str(offender_box),
                "victim_box": str(victim_box) if victim_box else "",
                "vehicle_box": str(vehicle_box) if vehicle_box else "",

                "incident_dir": str(incident_dir),
                "full_frame_path": str(full_frame_path),
                "offender_path": str(offender_path),
                "victim_path": str(victim_path) if victim_path else "",
                "vehicle_path": str(vehicle_path) if vehicle_path else "",

                "extra_json": json.dumps({
                    "confirmed_snatch": confirmed_snatch,
                    "fight_hits": fight_hits,
                    "fight_pair_persist": fight_pair_persist,
                    "fight_box_persist": fight_box_persist,
                }, ensure_ascii=False),
            }
            log_row(row)

            # FIR report
            report_bundle = write_fir_report(run_id=run_id, event_row=row)
            report_txt_path = reports_dir / f"{new_event_type}_{incident_id}.txt"
            report_json_path = reports_dir / f"{new_event_type}_{incident_id}.json"
            report_txt_path.write_text(report_bundle["text"], encoding="utf-8")
            report_json_path.write_text(json.dumps(report_bundle["meta"], indent=2, ensure_ascii=False), encoding="utf-8")

            last_event_time = now
            last_incident_time = now

        # --- draw thin boxes for all persons (optional) ---
        for tid, box in persons:
            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        involved = set(alert_state.keys())

        # locked offender/victim in red
        if lock_offender_tid is not None and lock_offender_tid in involved and lock_offender_tid in person_map:
            x1,y1,x2,y2 = person_map[lock_offender_tid]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, lock_event_label or "ALERT", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        if lock_victim_tid is not None and lock_victim_tid in involved and lock_victim_tid in person_map:
            x1,y1,x2,y2 = person_map[lock_victim_tid]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, lock_event_label or "ALERT", (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        # snatching vehicle
        if lock_event_label == "SNATCHING" and lock_vehicle_box is not None and lock_offender_tid in involved:
            vx1,vy1,vx2,vy2 = lock_vehicle_box
            cv2.rectangle(frame, (vx1,vy1), (vx2,vy2), (0,255,255), 2)
            cv2.putText(frame, "VEHICLE", (vx1, max(0, vy1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # weapon overlay
        if ENABLE_WEAPON and weapon_model is not None:
            wres = weapon_model(frame, conf=WEAPON_CONF, verbose=False)[0]
            for b in wres.boxes:
                label = weapon_model.names[int(b.cls[0])]
                if label.lower() == "person":
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,W,H)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(frame, label, (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # decrement alert
        for tid in list(alert_state.keys()):
            alert_state[tid] -= 1
            if alert_state[tid] <= 0:
                del alert_state[tid]

        # clear lock if expired
        if lock_offender_tid is not None:
            off_alive = lock_offender_tid in alert_state
            vic_alive = (lock_victim_tid in alert_state) if lock_victim_tid is not None else False
            if not off_alive and not vic_alive:
                lock_event_label = None
                lock_offender_tid = None
                lock_victim_tid = None
                lock_vehicle_box = None

        writer.write(frame)

    cap.release()
    writer.release()

    pd.DataFrame(records).to_csv(events_csv, index=False)

    # Return artifacts for frontend
    return {
        "input_video": str(video_path),
        "annotated_video": str(output_video),
        "events_csv": str(events_csv),
        "evidence_dir": str(evidence_dir),
        "reports_dir": str(reports_dir),
    }
