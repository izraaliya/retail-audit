import io
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import agent
from app.audit import DataError, RULE_AUDITS, detect_kind, run_audit, select
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
    assert set(tables) == {"orders", "items", "refunds", "cash", "employees", "stores", "skus", "opname", "customers"}


def test_planted_cashiers_found_without_false_positives(res):
    high = set(res["risk"][res["risk"].level == "Tinggi"].cashier_id)
    assert high == PLANTED
    assert set(res["risk"][res["risk"].level == "Normal"].cashier_id).isdisjoint(PLANTED)


def test_each_scheme_maps_to_its_rule(res):
    got = {(f["subject"], f["rule"]) for f in res["findings"]}
    for c, rule in [("K02", "refund_disetujui_sendiri"), ("K05", "diskon_sering"), ("K08", "selisih_kas"), ("K11", "luar_jam_buka"),
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
    assert len(sc["files"]) == 9 and sc["sample_files"] == ["Data_Order_Lengkap.xlsx", "Data_Master.xlsx"]


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
    assert client.post("/api/analyze", files=[("files", ("c.csv", tables["refunds"].to_csv(index=False).encode(), "text/csv"))]).status_code == 422


def test_inspect_reports_tables_without_running_audit(tables):
    r = client.post("/api/inspect", files=[("files", ("Data_Master.xlsx", (S / "Data_Master.xlsx").read_bytes(), "application/octet-stream"))])
    d = r.json()
    assert r.status_code == 200 and set(d["found"]) == {"employees", "stores", "skus", "opname", "customers"} and d["ready"] is False
    both = client.post("/api/inspect", files=[("files", ("Data_Order_Lengkap.xlsx", (S / "Data_Order_Lengkap.xlsx").read_bytes(), "application/octet-stream")),
                                              ("files", ("Data_Master.xlsx", (S / "Data_Master.xlsx").read_bytes(), "application/octet-stream"))]).json()
    assert both["ready"] is True and len(both["found"]) == 9 and both["duplicates"] == {}
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


def _call(name, **args):
    return {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}


def _say(text):
    return {"role": "model", "parts": [{"text": text}]}


def test_gemini_tool_loop_and_errors(res, monkeypatch):
    tb = agent.Toolbox(res)
    seq = [_call("cashier_breakdown", cashier_id="K17", metric="refund"), _say("Wulan: SKU teratas sudah dicek.")]
    sent = []
    monkeypatch.setattr(agent, "_chat", lambda contents, tools, timeout, force_text=False: (sent.append((len(contents), force_text)), seq.pop(0))[1])
    out = agent._gen(res, "wulan di sku apa", tb, {"x": 1}, True, 5)
    assert out.startswith("Wulan") and sent == [(1, False), (3, False)] and tb.log[-1]["tool"] == "cashier_breakdown"
    seen = []

    def fake(contents, tools, timeout, force_text=False):
        seen.append(contents[-1])
        return _say("ok") if force_text else _call("hapus_semua")

    monkeypatch.setattr(agent, "_chat", fake)
    assert agent._gen(res, "x", tb, {}, True, 5) == "ok"
    reply = [p["functionResponse"] for c in seen for p in c["parts"] if "functionResponse" in p][0]
    assert reply["name"] == "hapus_semua" and "tidak ada" in reply["response"]["result"]["error"]


def test_gemini_request_shape_and_empty_parameters(res, monkeypatch):
    summary = next(t for t in agent.TOOLS if t["name"] == "summary")
    assert "parameters" not in summary
    gf = next(t for t in agent.TOOLS if t["name"] == "get_finding")
    assert gf["parameters"]["required"] == ["finding_id"]
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-x")
    captured = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"candidates":[{"content":{"role":"model","parts":[{"text":"hai"}]}}]}'

    def fake_urlopen(req, timeout):
        captured.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, body=json.loads(req.data))
        return R()

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    m = agent._chat([{"role": "user", "parts": [{"text": "q"}]}], agent.TOOLS, 5, force_text=True)
    assert m["parts"][0]["text"] == "hai"
    assert captured["url"].endswith("/models/gemini-x:generateContent") and "k-test" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "k-test"
    assert captured["body"]["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
    assert len(captured["body"]["tools"][0]["functionDeclarations"]) == 5


def test_gemini_enabled_depends_on_key(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ASSISTANT_MODE"):
        monkeypatch.delenv(k, raising=False)
    assert agent.enabled() is False
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    assert agent.enabled() is True
    monkeypatch.setenv("ASSISTANT_MODE", "rules")
    assert agent.enabled() is False


@pytest.mark.parametrize("code,needle", [(403, "GEMINI_API_KEY"), (404, "GEMINI_MODEL"), (429, "kuota")])
def test_gemini_http_errors_report_cause_without_leaking_key(res, monkeypatch, code, needle):
    import urllib.error
    monkeypatch.setenv("GEMINI_API_KEY", "rahasia-123")

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)

    monkeypatch.setattr(agent.urllib.request, "urlopen", boom)
    a = agent.ask(res, "ringkas")
    assert a["source"] == "aturan" and "penyebab teknis" in a["answer"] and needle in a["answer"] and "rahasia-123" not in a["answer"]


def test_gemini_unreachable_reports_cause(res, monkeypatch):
    import urllib.error
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(urllib.error.URLError("dns")))
    a = agent.ask(res, "ringkas")
    assert a["source"] == "aturan" and "Gemini tidak terjangkau" in a["answer"]


# ---------------------------------------------------------------- two audits: cashier behaviour vs order data
def test_every_rule_belongs_to_an_audit_and_select_filters(res):
    assert {f["rule"] for f in res["findings"]} <= set(RULE_AUDITS)
    k, o = select(res, "kasir"), select(res, "order")
    assert all("kasir" in f["audits"] for f in k["findings"]) and all("order" in f["audits"] for f in o["findings"])
    assert not any(f["rule"] == "integritas_data" for f in k["findings"]) and any(f["rule"] == "integritas_data" for f in o["findings"])
    assert not any(f["rule"] == "diskon_sering" for f in o["findings"])
    for v in (k, o):
        assert [f["id"] for f in v["findings"]] == [f"F{i:03d}" for i in range(1, len(v["findings"]) + 1)]
    assert set(k["risk"][k["risk"].level == "Tinggi"].cashier_id) == PLANTED
    assert set(o["risk"][o["risk"].level == "Tinggi"].cashier_id) <= {"K14", "K17"}
    with pytest.raises(DataError):
        select(res, "lainnya")


def test_clean_sample_has_no_false_order_data_findings(res):
    new_rules = {"id_ganda", "angka_mustahil", "nilai_tidak_wajar", "harga_tidak_cocok", "refund_melebihi_order"}
    assert not new_rules & {f["rule"] for f in res["findings"]}


def test_order_audit_catches_planted_data_errors(tables):
    t = {k: v.copy() for k, v in tables.items()}
    o = t["orders"].reset_index(drop=True)
    o.loc[10, "total"] = -50000
    o.loc[11, "discount"] = o.loc[11, "subtotal"] + 5000
    o.loc[12, ["total", "subtotal"]] = o.total.median() * 200
    o.loc[12, "voided"] = 0
    t["orders"] = pd.concat([o, o.iloc[[5]]], ignore_index=True)
    it = t["items"].reset_index(drop=True)
    sku = it.sku.iloc[0]
    it.loc[it.sku == sku, "unit_price"] = it.loc[it.sku == sku, "unit_price"] * 1.5
    t["items"] = it
    rf = t["refunds"].reset_index(drop=True)
    row = rf[rf.order_id.isin(o.order_id)].index[0]
    rf.loc[row, "amount"] = float(o.set_index("order_id").subtotal[rf.loc[row, "order_id"]]) * 3 + 10000
    t["refunds"] = rf
    r = select(run_audit(t), "order")
    got = {f["rule"]: f for f in r["findings"]}
    for rule in ("id_ganda", "angka_mustahil", "nilai_tidak_wajar", "harga_tidak_cocok", "refund_melebihi_order"):
        assert rule in got, rule
        assert got[rule]["evidence_count"] >= 1 and got[rule]["subject_type"] == "data"
    assert {e["alasan"] for e in got["angka_mustahil"]["evidence"]} == {"ada nilai negatif", "diskon lebih besar dari subtotal"}
    assert got["id_ganda"]["evidence_count"] == 2
    assert all(e["sku"] == sku for e in got["harga_tidak_cocok"]["evidence"])


def test_order_audit_api_and_report(monkeypatch):
    monkeypatch.setattr(agent, "enabled", lambda: False)
    k = client.post("/api/analyze/sample").json()
    assert k["mode"] == "kasir" and all("kasir" in f["audits"] for f in k["findings"])
    r = client.post("/api/analyze/sample", data={"mode": "order"})
    o = r.json()
    assert r.status_code == 200 and o["mode"] == "order" and o["findings"] and all("order" in f["audits"] for f in o["findings"])
    assert all(f["steps"] for f in o["findings"])
    dl = client.get(f"/api/report/{o['job_id']}.xlsx")
    assert dl.status_code == 200 and "Data_Order" in dl.headers["content-disposition"]
    assert "Data_Order" not in client.get(f"/api/report/{k['job_id']}.xlsx").headers["content-disposition"]
    a = client.post("/api/ask", json={"job_id": o["job_id"], "question": "jelaskan F001"}).json()
    assert a["source"] == "aturan" and "F001" in a["answer"]
    bad = client.post("/api/analyze/sample", data={"mode": "xyz"})
    assert bad.status_code == 422 and "audit" in bad.json()["detail"].lower()


def test_sample_data_is_complete(tables):
    o, it, cu = tables["orders"], tables["items"], tables["customers"]
    for c in ("customer_id", "channel", "payment_method", "voucher_code", "jumlah_item", "ppn", "poin_didapat", "void_reason"):
        assert c in o.columns, c
    for c in ("nama_produk", "kategori", "brand", "line_total"):
        assert c in it.columns, c
    assert (it.line_total == it.qty * it.unit_price).all()
    assert set(it.nama_produk) == set(tables["skus"].name)
    assert o.customer_id.nunique() > 1000 and (o.customer_id == "GUEST").mean() > 0.2
    assert cu.customer_id.is_unique and {"no_hp", "email", "kota", "member_tier", "tanggal_daftar"} <= set(cu.columns)
    assert "discount_approved_by" not in o.columns


def test_customer_checks_find_the_planted_customers_only(res, tables):
    truth = pd.read_csv(S / "Label_Anomali.csv")
    planted = set(truth[truth.skema == "pelanggan_refund_sering"].ref)
    flagged = {f["subject"].split(" ")[0] for f in res["findings"] if f["rule"] == "pelanggan_refund_sering"}
    assert len(planted) == 3 and flagged == planted
    unk = next(f for f in res["findings"] if f["rule"] == "pelanggan_tidak_dikenal")
    assert unk["evidence_count"] == int((truth.skema == "pelanggan_tidak_dikenal").sum()) == 25
    assert all(e["customer_id"].startswith("C9") for e in unk["evidence"])
    assert all(f["audits"] == ["order"] for f in res["findings"] if f["rule"].startswith("pelanggan_"))
    no_cust = run_audit({k: v for k, v in tables.items() if k != "customers"})
    assert not {f["rule"] for f in no_cust["findings"]} & {"pelanggan_tidak_dikenal", "pelanggan_refund_sering"}
