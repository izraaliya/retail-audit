"""FastAPI app: upload order data, get audit findings, ask the audit assistant, download the working paper."""
from __future__ import annotations

import io
import time
import uuid
from datetime import date
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent
from .audit import DataError, LABELS, MANDATORY, REQUIRED, Params, detect_kind, run_audit
from .report import build_workbook

ROOT = Path(__file__).resolve().parent.parent
STATIC, SAMPLES, REPORTS = Path(__file__).resolve().parent / "static", ROOT / "sample_data", ROOT / ".reports"
SAMPLE_FILES = ["Data_Order_Lengkap.xlsx", "Data_Master.xlsx"]
MAX_BYTES = 40 * 1024 * 1024
TTL, MAX_JOBS = 3600, 12

app = FastAPI(title="Retail Audit", description="Upload transaction data, get audit findings with evidence.", version="1.0.0")
JOBS: dict[str, dict] = {}
LOCK = Lock()


def _gc():
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["t"] > TTL]:
        JOBS.pop(k, None)
    while len(JOBS) > MAX_JOBS:
        JOBS.pop(min(JOBS, key=lambda k: JOBS[k]["t"]), None)


def _records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _parse(filename: str, data: bytes) -> list[tuple[str, str, pd.DataFrame]]:
    name = filename.lower()
    if len(data) > MAX_BYTES:
        raise HTTPException(422, f"{filename}: file lebih dari {MAX_BYTES // 1024 // 1024} MB.")
    if name.endswith((".xlsx", ".xlsm")):
        try:
            items = list(pd.read_excel(io.BytesIO(data), sheet_name=None).items())
        except Exception:
            raise HTTPException(422, f"{filename}: file Excel tidak bisa dibaca.")
        excel = True
    elif name.endswith(".xls"):
        raise HTTPException(422, f"{filename}: format .xls lama tidak didukung. Simpan ulang sebagai .xlsx atau .csv.")
    else:
        try:
            items, excel = [(filename, pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding_errors="replace"))], False
        except Exception:
            raise HTTPException(422, f"{filename}: tidak bisa dibaca sebagai CSV.")
    out = []
    for label, df in items:
        df.columns = [str(c).strip().lower() for c in df.columns]
        kind = detect_kind(df)
        if kind is None:
            if excel:
                continue
            raise HTTPException(422, f"{filename}: kolomnya tidak cocok dengan data apa pun. Lihat bagian Format kolom.")
        out.append((f"{filename} › {label}" if excel else filename, kind, df))
    return out


def _payload(res: dict, job_id: str, detected: dict) -> dict:
    fs = res["findings"]
    slim = [{**{k: (v if k != "evidence" else v[:60]) for k, v in f.items()}, "steps": agent.STEPS.get(f["rule"], [])} for f in fs]
    return {
        "job_id": job_id, "detected_files": detected, "meta": res["meta"],
        "kpi": {"orders": res["meta"]["n_orders"], "findings": len(fs), "severity": {s: sum(1 for f in fs if f["severity"] == s) for s in ("Tinggi", "Sedang", "Rendah")},
                "amount": round(sum(f["amount"] for f in fs)), "high_risk_cashiers": int((res["risk"].level == "Tinggi").sum())},
        "risk": _records(res["risk"]), "peers": _records(res["peers"]), "findings": slim,
        "assistant": {"enabled": bool(agent._key())},
    }


def _run(tables: dict, detected: dict, discount_limit: float) -> dict:
    try:
        res = run_audit(tables, Params(discount_limit=discount_limit))
    except DataError as e:
        raise HTTPException(422, str(e))
    job = uuid.uuid4().hex[:12]
    with LOCK:
        _gc()
        JOBS[job] = {"t": time.time(), "res": res}
    return _payload(res, job, detected)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/schema")
def schema():
    where = {"orders": "Data_Order_Lengkap.xlsx › Order", "items": "Data_Order_Lengkap.xlsx › Item", "refunds": "Data_Order_Lengkap.xlsx › Refund",
             "cash": "Data_Order_Lengkap.xlsx › Setoran_Kas", "employees": "Data_Master.xlsx › Karyawan", "stores": "Data_Master.xlsx › Toko",
             "skus": "Data_Master.xlsx › SKU", "opname": "Data_Master.xlsx › Opname"}
    return {"files": [{"kind": k, "label": LABELS[k], "required": k in MANDATORY, "columns": REQUIRED[k], "sample": where[k]} for k in REQUIRED],
            "sample_files": SAMPLE_FILES}


def _check(limit: float):
    if not 5 <= limit <= 90:
        raise HTTPException(422, "Batas diskon harus antara 5 dan 90 persen.")


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), discount_limit: float = Form(20.0)):
    _check(discount_limit)
    tables, detected = {}, {}
    for f in files:
        for label, kind, df in _parse(f.filename or "file", await f.read()):
            if kind in tables:
                raise HTTPException(422, f"Ada dua sumber untuk '{LABELS[kind]}' ({detected[kind]} dan {label}). Sisakan satu saja.")
            tables[kind], detected[kind] = df, label
    return JSONResponse(_run(tables, detected, discount_limit))


@app.post("/api/analyze/sample")
def analyze_sample(discount_limit: float = Form(20.0)):
    _check(discount_limit)
    tables, detected = {}, {}
    for name in SAMPLE_FILES:
        for label, kind, df in _parse(name, (SAMPLES / name).read_bytes()):
            tables[kind], detected[kind] = df, label
    return JSONResponse(_run(tables, detected, discount_limit))


class Ask(BaseModel):
    job_id: str
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    job = JOBS.get(body.job_id)
    if not job:
        raise HTTPException(404, "Hasil analisis sudah kedaluwarsa. Unggah ulang datanya.")
    return agent.ask(job["res"], body.question)


@app.get("/api/report/{job_id}.xlsx")
def report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Hasil analisis sudah kedaluwarsa. Unggah ulang datanya.")
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{job_id}.xlsx"
    if not path.exists():
        build_workbook(job["res"], path)
    return FileResponse(path, filename="Kertas_Kerja_Audit.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/sample/{name}")
def sample(name: str):
    if name not in SAMPLE_FILES:
        raise HTTPException(404, "File contoh tidak ada.")
    return FileResponse(SAMPLES / name, filename=name)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
