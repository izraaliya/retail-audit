import io
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import agent
from app.audit import DataError, detect_kind, run_audit
from app.main import app

S = Path(__file__).resolve().parent.parent / "sample_data"
client = TestClient(app)


def load():
    t = {}
    for f in ("Data_Order_Lengkap", "Data_Master"):
        for n, df in pd.read_excel(S / f"{f}.xlsx", sheet_name=None).items():
            if n != "Petunjuk":
                df.columns = [c.lower() for c in df.columns]
                t[detect_kind(df)] = df
    return t


@pytest.fixture(scope="module")
def tables():
    return load()


@pytest.fixture(scope="module")
def res(tables):
    return run_audit(tables)


PLANTED = {"K02", "K05", "K08", "K11", "K14", "K17"}


def test_all_kinds_detected(tables):
    assert set(tables) == {"orders", "items", "refunds", "cash", "employees", "stores", "skus", "opname"}


def test_planted_cashiers_found_without_false_positives(res):
    high = set(res["risk"][res["risk"].level == "Tinggi"].cashier_id)
    assert high == PLANTED
    assert set(res["risk"][res["risk"].level == "Normal"].cashier_id).isdisjoint(PLANTED)


def test_each_scheme_maps_to_its_rule(res):
    got = {(f["subject"], f["rule"]) for f in res["findings"]}
    for c, rule in [("K02", "refund_disetujui_sendiri"), ("K05", "diskon_tanpa_persetujuan"), ("K08", "selisih_kas"), ("K11", "luar_jam_buka"),
                    ("K14", "struk_ganda"), ("K17", "refund_tanpa_order")]:
        assert (c, rule) in got, (c, rule)
    assert any(f["rule"] == "susut_stok" and "Malioboro" in f["subject"] for f in res["findings"])
    assert any(f["rule"] == "integritas_data" for f in res["findings"])


def test_every_finding_has_evidence_and_valid_fields(res):
    for f in res["findings"]:
        assert f["evidence_count"] > 0 and f["evidence"] and f["severity"] in ("Tinggi", "Sedang", "Rendah") and 1 <= f["score"] <= 100
    assert [f["score"] for f in res["findings"]] == sorted((f["score"] for f in res["findings"]), reverse=True)


def test_evidence_matches_claim(res, tables):
    f = next(f for f in res["findings"] if f["rule"] == "refund_disetujui_sendiri" and f["subject"] == "K02")
    rf = tables["refunds"]
    assert f["evidence_count"] == int(((rf.cashier_id == "K02") & (rf.approved_by == "K02")).sum())


def test_clean_data_has_no_high_findings(tables):
    t = {k: v.copy() for k, v in tables.items()}
    bad = PLANTED
    o = t["orders"]
    keep = ~o.cashier_id.isin(bad)
    ids = set(o[keep].order_id)
    t["orders"] = o[keep]
    t["items"] = t["items"][t["items"].order_id.isin(ids)]
    t["refunds"] = t["refunds"][~t["refunds"].cashier_id.isin(bad)]
    t["cash"] = t["cash"][~t["cash"].cashier_id.isin(bad)]
    r = run_audit(t)
    assert not [f for f in r["findings"] if f["severity"] == "Tinggi" and f["rule"] not in ("susut_stok",)]


def test_orders_only_runs(tables):
    r = run_audit({"orders": tables["orders"]})
    assert r["meta"]["checks"] == [] and any("tidak diunggah" in w for w in r["meta"]["warnings"])


def test_errors():
    with pytest.raises(DataError):
        run_audit({})
    with pytest.raises(DataError):
        run_audit({"orders": pd.DataFrame({"order_id": ["a"], "datetime": ["2026-01-01"], "store_id": ["S"], "cashier_id": ["K"], "subtotal": [1], "discount": [0], "total": [1]})})


def test_assistant_fallback_answers(res, monkeypatch):
    monkeypatch.setattr(agent, "enabled", lambda: False)
    a = agent.ask(res, "jelaskan F001")
    assert a["source"] == "aturan" and "F001" in a["answer"] and a["steps"][0]["tool"] == "get_finding"
    assert "K05" in agent.ask(res, "ceritakan K05")["answer"]
    assert "Prioritas" in agent.ask(res, "ringkas")["answer"]
    assert agent.ask(res, "  ")["answer"].startswith("Tulis")


def test_assistant_failure_and_hang_fall_back(res, monkeypatch):
    monkeypatch.setattr(agent, "enabled", lambda: True)
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(agent, "_model_answer", lambda *a: (_ for _ in ()).throw(RuntimeError("503")))
    a = agent.ask(res, "ringkas")
    assert a["source"] == "aturan" and "tidak tersedia" in a["answer"]
    monkeypatch.setattr(agent, "_model_answer", lambda *a: time.sleep(40))
    t = time.time()
    assert "tidak merespons" in agent.ask(res, "ringkas")["answer"] and time.time() - t < 30


def test_toolbox_unknown_ids(res):
    tb = agent.Toolbox(res)
    assert "error" in tb.get_finding("F999") and "error" in tb.cashier_profile("NOPE")


# ---------------------------------------------------------------- API
def test_health_index_schema():
    assert client.get("/health").json() == {"status": "ok"}
    assert "Retail Audit" in client.get("/").text
    sc = client.get("/api/schema").json()
    assert len(sc["files"]) == 8 and sc["sample_files"] == ["Data_Order_Lengkap.xlsx", "Data_Master.xlsx"]


def test_sample_ask_and_report():
    r = client.post("/api/analyze/sample")
    assert r.status_code == 200
    d = r.json()
    assert d["kpi"]["high_risk_cashiers"] == 6 and d["findings"][0]["steps"] and len(d["peers"]) == 18
    a = client.post("/api/ask", json={"job_id": d["job_id"], "question": "ringkas hasil"})
    assert a.status_code == 200 and a.json()["answer"]
    rep = client.get(f"/api/report/{d['job_id']}.xlsx")
    assert rep.status_code == 200
    sheets = pd.read_excel(io.BytesIO(rep.content), sheet_name=None)
    assert {"Ringkasan", "Temuan", "Risiko Kasir", "Bukti"} <= set(sheets) and len(sheets["Temuan"]) == len(d["findings"])
    assert client.post("/api/ask", json={"job_id": "nope", "question": "x"}).status_code == 404
    assert client.get("/api/report/nope.xlsx").status_code == 404
    assert client.get("/sample/secret.xlsx").status_code == 404 and client.get("/sample/Data_Master.xlsx").status_code == 200


def test_csv_upload_orders_only(tables):
    buf = tables["orders"].to_csv(index=False).encode()
    r = client.post("/api/analyze", files=[("files", ("o.csv", buf, "text/csv"))])
    assert r.status_code == 200 and r.json()["detected_files"] == {"orders": "o.csv"}


def test_bad_uploads(tables):
    assert client.post("/api/analyze", files=[("files", ("x.csv", b"a,b\n1,2\n", "text/csv"))]).status_code == 422
    assert client.post("/api/analyze", files=[("files", ("x.xls", b"zzz", "application/vnd.ms-excel"))]).status_code == 422
    b = tables["orders"].to_csv(index=False).encode()
    r = client.post("/api/analyze", files=[("files", ("a.csv", b, "text/csv")), ("files", ("b.csv", b, "text/csv"))])
    assert r.status_code == 422 and "dua sumber" in r.json()["detail"].lower()
    assert client.post("/api/analyze/sample", data={"discount_limit": "1"}).status_code == 422
    assert client.post("/api/analyze", files=[("files", ("c.csv", tables["refunds"].to_csv(index=False).encode(), "text/csv"))]).status_code == 422


def test_inspect_reports_tables_without_running_audit(tables):
    r = client.post("/api/inspect", files=[("files", ("Data_Master.xlsx", (S / "Data_Master.xlsx").read_bytes(), "application/octet-stream"))])
    d = r.json()
    assert r.status_code == 200 and set(d["found"]) == {"employees", "stores", "skus", "opname"} and d["ready"] is False
    both = client.post("/api/inspect", files=[("files", ("Data_Order_Lengkap.xlsx", (S / "Data_Order_Lengkap.xlsx").read_bytes(), "application/octet-stream")),
                                              ("files", ("Data_Master.xlsx", (S / "Data_Master.xlsx").read_bytes(), "application/octet-stream"))]).json()
    assert both["ready"] is True and len(both["found"]) == 8 and both["duplicates"] == {}
    csv = tables["orders"].head(5).to_csv(index=False).encode()
    dup = client.post("/api/inspect", files=[("files", ("a.csv", csv, "text/csv")), ("files", ("b.csv", csv, "text/csv"))]).json()
    assert dup["ready"] is False and "orders" in dup["duplicates"]
    bad = client.post("/api/inspect", files=[("files", ("x.xls", b"zz", "application/vnd.ms-excel"))]).json()
    assert bad["files"][0]["error"]


def test_breakdown_and_sku_question(res, monkeypatch):
    monkeypatch.setattr(agent, "enabled", lambda: False)
    tb = agent.Toolbox(res)
    top = tb.cashier_breakdown("K17", "refund")
    assert top and top[0]["nilai_rp"] >= top[-1]["nilai_rp"] and set(top[0]) == {"sku", "produk", "nilai_rp", "jumlah"}
    assert tb.cashier_breakdown("K05", "discount")
    a = agent.ask(res, "wulan paling bahaya di SKU apa")
    assert "Produk dengan nilai refund terbesar" in a["answer"] and a["steps"][-1]["tool"] == "cashier_breakdown"


def test_model_path_prefetches_and_falls_back_to_plain_call(res, monkeypatch):
    monkeypatch.setattr(agent, "enabled", lambda: True)
    calls = []

    def fake_gen(res_, q, tb, facts, use_tools, timeout):
        calls.append(use_tools)
        if use_tools:
            raise RuntimeError("tools call failed")
        assert "kasir_K17" in facts
        return "jawaban dari model"

    monkeypatch.setattr(agent, "_gen", fake_gen)
    a = agent.ask(res, "ceritakan Wulan")
    assert a["answer"] == "jawaban dari model" and a["source"] == "asisten" and calls == [True, False]


def test_ollama_tool_loop_and_errors(res, monkeypatch):
    tb = agent.Toolbox(res)
    seq = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "cashier_breakdown", "arguments": {"cashier_id": "K17", "metric": "refund"}}}]},
        {"role": "assistant", "content": "Wulan: SKU teratas sudah dicek."},
    ]
    sent = []
    monkeypatch.setattr(agent, "_chat", lambda msgs, tools, timeout: (sent.append(len(msgs)), seq.pop(0))[1])
    out = agent._gen(res, "wulan di sku apa", tb, {"x": 1}, True, 5)
    assert out.startswith("Wulan") and sent == [1, 3] and tb.log[-1]["tool"] == "cashier_breakdown"
    monkeypatch.setattr(agent, "_chat", lambda *a: {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "hapus_semua", "arguments": {}}}]})
    monkeypatch.setattr(agent, "_chat", lambda msgs, tools, timeout: {"role": "assistant", "content": "ok"} if tools is None else {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "hapus_semua", "arguments": {}}}]})
    assert agent._gen(res, "x", tb, {}, True, 5) == "ok"


def test_ollama_down_reports_cause(res, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    monkeypatch.setattr(agent, "enabled", lambda: True)
    a = agent.ask(res, "ringkas")
    assert a["source"] == "aturan" and "penyebab teknis" in a["answer"] and "Ollama tidak terjangkau" in a["answer"]
    monkeypatch.undo()
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    agent._probe["t"] = 0
    assert agent.enabled() is False
