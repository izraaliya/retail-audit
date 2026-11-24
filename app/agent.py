"""Audit assistant: an agent that investigates the findings with read-only tools and writes an evidence-based answer.

When GEMINI_API_KEY is set, a Gemini model chooses which tools to call (function calling) and every call is logged.
Without a key, or when the call fails, it answers from the same tools with fixed rules, so the page always works.
With a key, the question and the audit facts the tools return (finding ids, cashier ids and names, amounts) are sent to the Gemini API.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pandas as pd

DEFAULT_MODEL = "gemini-3.5-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_pool = ThreadPoolExecutor(max_workers=2)

STEPS = {
    "refund_tinggi": ["Buka struk asli untuk 10 refund terbesar dan cocokkan dengan bukti retur fisik.", "Tanyakan alasan refund ke pelanggan lewat nomor di struk (sampel).", "Bandingkan dengan kasir lain di toko yang sama."],
    "refund_tanpa_order": ["Cari order_id di sistem kasir lain atau arsip. Jika tidak ada, anggap refund fiktif sampai terbukti sebaliknya.", "Periksa siapa yang menyetujui dan apakah uang keluar dari laci kas.", "Cek rekaman CCTV pada jam refund."],
    "refund_disetujui_sendiri": ["Cek pengaturan hak akses: kasir tidak boleh menyetujui refund-nya sendiri.", "Minta manajer menelaah ulang refund yang disetujui sendiri.", "Cocokkan dengan bukti retur fisik."],
    "refund_tanpa_persetujuan": ["Minta bukti persetujuan tertulis atau kode otorisasi.", "Perketat sistem agar refund besar wajib disetujui."],
    "refund_sebelum_order": ["Periksa jam sistem dan proses input manual.", "Cocokkan dengan log sistem."],
    "refund_terlambat": ["Periksa kebijakan batas waktu refund dan siapa yang mengizinkan pengecualian."],
    "id_ganda": ["Cari baris-baris dengan nomor order sama dan bandingkan isinya: salinan persis atau transaksi berbeda?", "Periksa apakah sistem kasir mengizinkan nomor order dipakai ulang atau data diekspor dua kali."],
    "angka_mustahil": ["Buka struk asli dan cocokkan dengan catatan sistem kasir.", "Cek apakah nilai negatif adalah retur yang salah dicatat sebagai penjualan.", "Periksa siapa yang membuat atau mengubah struk itu."],
    "nilai_tidak_wajar": ["Cocokkan dengan faktur atau bukti pembayaran pembelian besar (grosir, pesanan khusus).", "Periksa apakah angkanya salah ketik, misalnya kelebihan nol."],
    "harga_tidak_cocok": ["Periksa apakah ada promo resmi pada periode itu yang belum masuk data master.", "Cek siapa yang boleh mengubah harga di kasir.", "Perbarui harga master bila memang sudah usang."],
    "refund_melebihi_order": ["Buka order asal dan semua refund yang terkait, lalu cocokkan dengan barang yang dikembalikan.", "Periksa apakah refund yang sama diproses dua kali.", "Cek siapa yang menyetujui refund itu."],
    "pelanggan_tidak_dikenal": ["Cari ID itu di sistem member: salah ketik, atau memang tidak pernah ada?", "Tanyakan ke kasir bagaimana ID itu dimasukkan dan apakah ada poin atau diskon member yang ikut diberikan.", "Periksa apakah ID yang sama dipakai berulang kali."],
    "pelanggan_refund_sering": ["Hubungi pelanggan lewat nomor di master untuk memastikan refund memang diminta olehnya.", "Cocokkan barang retur secara fisik dengan catatan refund.", "Bandingkan dengan kasir yang memproses: apakah hampir selalu orang yang sama?"],
    "diskon_sering": ["Bandingkan dengan promo yang resmi berlaku pada periode itu.", "Tanyakan dasar pemberian diskon ke kasir."],
    "selisih_kas": ["Hitung kas bersama saksi pada penutupan berikutnya.", "Periksa kebiasaan membuka laci tanpa transaksi.", "Cek CCTV area kasir pada shift terkait."],
    "luar_jam_buka": ["Periksa log akses toko dan alarm pada jam itu.", "Cek apakah ada kegiatan resmi di luar jam (stok opname, event).", "Cek penggunaan akun kasir oleh orang lain."],
    "struk_ganda": ["Cocokkan dengan struk fisik: apakah pelanggan benar membeli dua kali.", "Periksa apakah salah satu struk dibatalkan setelah uang diterima."],
    "void_tinggi": ["Cek alasan void tiap struk dan siapa yang menyetujui.", "Cocokkan waktu void dengan pembayaran tunai."],
    "integritas_data": ["Periksa log perubahan total atau harga di sistem kasir.", "Cek apakah ada pengeditan manual setelah struk dibuat."],
    "susut_stok": ["Lakukan hitung ulang fisik untuk SKU tersebut.", "Periksa retur ke supplier, barang rusak, dan transfer antar toko yang belum dicatat.", "Tinjau akses gudang dan CCTV."],
    "pola_tidak_biasa": ["Lihat hari-hari yang ditandai model, lalu cocokkan dengan temuan aturan.", "Gunakan sebagai petunjuk arah pemeriksaan, bukan sebagai bukti."],
}


def rp(x) -> str:
    return 'Rp ' + f'{int(x):,}'.replace(',', '.')


def _api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _model() -> str:
    return (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()


def _timeout() -> float:
    try:
        return max(5.0, float(os.getenv("AGENT_TIMEOUT_SECONDS", "90")))
    except ValueError:
        return 90.0


def enabled() -> bool:
    """True when a Gemini API key is configured (and ASSISTANT_MODE is not 'rules'). No network call, so the page load stays quick."""
    return os.getenv("ASSISTANT_MODE", "").lower() != "rules" and bool(_api_key())


def _http_error(code: int) -> str:
    if code in (400, 401, 403):
        return f"Gemini menolak permintaan (HTTP {code}); periksa GEMINI_API_KEY dan akses modelnya"
    if code == 404:
        return f"model '{_model()}' tidak ditemukan; atur GEMINI_MODEL ke model Gemini yang tersedia"
    if code == 429:
        return "kuota atau batas laju Gemini terlampaui (HTTP 429); coba lagi nanti"
    return f"Gemini membalas HTTP {code}"


def _chat(contents: list, tools: list | None, timeout: float, force_text: bool = False) -> dict:
    """One generateContent call. Returns the model's content ({"role": "model", "parts": [...]}).
    force_text keeps the tool declarations (the history contains function calls) but forbids new calls."""
    body: dict = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
        if force_text:
            body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    req = urllib.request.Request(f"{API_BASE}/models/{_model()}:generateContent", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-goog-api-key": _api_key()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(_http_error(e.code)) from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini tidak terjangkau ({e.reason})") from None
    cands = data.get("candidates") or []
    if not cands:
        fb = (data.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"Gemini tidak memberi jawaban{' (diblokir: ' + fb + ')' if fb else ''}")
    content = cands[0].get("content") or {}
    return {"role": "model", "parts": content.get("parts") or []}


def _schema(name: str, desc: str, props: dict, req: list | None = None) -> dict:
    """A Gemini function declaration. Gemini rejects an empty parameters object, so it is left out when there are no properties."""
    d: dict = {"name": name, "description": desc}
    if props:
        d["parameters"] = {"type": "object", "properties": props, **({"required": req} if req else {})}
    return d


TOOLS = [
    _schema("summary", "Ringkasan audit: jumlah temuan per tingkat, kasir berisiko tertinggi.", {}),
    _schema("list_findings", "Daftar temuan, bisa disaring.", {"severity": {"type": "string", "description": "Tinggi, Sedang, atau Rendah"}, "cashier_id": {"type": "string"}, "rule": {"type": "string"}}),
    _schema("get_finding", "Detail satu temuan beserta baris bukti.", {"finding_id": {"type": "string", "description": "contoh F003"}}, ["finding_id"]),
    _schema("cashier_profile", "Profil risiko satu kasir.", {"cashier_id": {"type": "string", "description": "contoh K05"}}, ["cashier_id"]),
    _schema("store_summary", "Jumlah temuan, tingkat, dan nilai berisiko per toko; urut dari yang paling bermasalah.", {}),
    _schema("cashier_breakdown", "SKU/produk yang paling besar nilai refund, diskon, atau void seorang kasir.", {"cashier_id": {"type": "string"}, "metric": {"type": "string", "description": "refund, discount, atau void"}}, ["cashier_id"]),
]


class Toolbox:
    """Read-only tools over one audit result. Every call is logged."""

    def __init__(self, res: dict):
        self.res, self.log = res, []
        self.by_id = {f["id"]: f for f in res["findings"]}

    def _note(self, name, args, out):
        self.log.append(dict(tool=name, args=args, hasil=out if isinstance(out, str) else f"{len(out)} baris" if isinstance(out, list) else "ok"))

    def list_findings(self, severity: str = "", cashier_id: str = "", rule: str = "") -> list:
        """List audit findings. Filter by severity (Tinggi/Sedang/Rendah), cashier_id (e.g. K05) or rule name."""
        out = [dict(id=f["id"], rule=f["rule"], subject=f["subject"], nama=f["subject_name"], toko=f["store"], severity=f["severity"], skor=f["score"],
                    judul=f["title"], nilai_rp=round(f["amount"])) for f in self.res["findings"]
               if (not severity or f["severity"].lower() == severity.lower()) and (not cashier_id or f["subject"].lower() == cashier_id.lower())
               and (not rule or f["rule"] == rule)][:25]
        self._note("list_findings", dict(severity=severity, cashier_id=cashier_id, rule=rule), out)
        return out

    def get_finding(self, finding_id: str, max_rows: int = 8) -> dict:
        """Get one finding with its explanation, comparison with peers, suggested verification steps, and the first evidence rows."""
        f = self.by_id.get(finding_id.upper())
        if not f:
            self._note("get_finding", dict(finding_id=finding_id), "tidak ditemukan")
            return {"error": "finding tidak ditemukan"}
        out = dict(id=f["id"], judul=f["title"], detail=f["detail"], severity=f["severity"], skor=f["score"], nilai_rp=round(f["amount"]),
                   nilai_kasir=f["metric"], nilai_median_rekan=f["peer"], jumlah_bukti=f["evidence_count"], contoh_bukti=f["evidence"][:max_rows],
                   langkah_verifikasi=STEPS.get(f["rule"], []))
        self._note("get_finding", dict(finding_id=finding_id), "ok")
        return out

    def cashier_profile(self, cashier_id: str) -> dict:
        """Profile of one cashier: orders, risk score, comparison metrics against all cashiers, and all of their findings."""
        r = self.res["risk"]
        row = r[r.cashier_id.str.lower() == cashier_id.lower()]
        if row.empty:
            self._note("cashier_profile", dict(cashier_id=cashier_id), "tidak ditemukan")
            return {"error": "kasir tidak ditemukan"}
        pe = self.res["peers"]
        me = pe[pe.cashier_id.str.lower() == cashier_id.lower()].iloc[0].to_dict()
        out = dict(profil=row.iloc[0].to_dict(), metrik=me, median_semua_kasir={k: round(float(pe[k].median()), 2) for k in ("refund_pct", "diskon_pct", "void_pct", "kas_kurang")},
                   temuan=[dict(id=f["id"], judul=f["title"], skor=f["score"]) for f in self.res["findings"] if f["subject"].lower() == cashier_id.lower()])
        self._note("cashier_profile", dict(cashier_id=cashier_id), "ok")
        return out

    def cashier_breakdown(self, cashier_id: str, metric: str = "refund") -> list:
        """Which products drive a cashier's numbers. metric is 'refund' (refund value by SKU), 'discount' (discount value by SKU) or 'void' (voided value by SKU). Returns the top 5 SKUs."""
        t, cid = self.res["tables"], cashier_id.upper()
        names = dict(zip(t["skus"].sku, t["skus"].name)) if "skus" in t else {}
        out: list = []
        o = t["orders"]
        if metric == "refund" and "refunds" in t and "sku" in t["refunds"].columns:
            g = t["refunds"][t["refunds"].cashier_id.str.upper() == cid].groupby("sku").amount.agg(["sum", "size"]).sort_values("sum", ascending=False).head(5)
            out = [dict(sku=k, produk=names.get(k, k), nilai_rp=round(float(r["sum"])), jumlah=int(r["size"])) for k, r in g.iterrows()]
        elif metric in ("discount", "void") and "items" in t:
            mine = o[(o.cashier_id.str.upper() == cid) & ((o.discount > 0) if metric == "discount" else (o.voided == 1))]
            it = t["items"][t["items"].order_id.isin(mine.order_id)].assign(v=lambda x: x.qty * x.unit_price)
            g = it.groupby("sku").v.agg(["sum", "size"]).sort_values("sum", ascending=False).head(5)
            out = [dict(sku=k, produk=names.get(k, k), nilai_rp=round(float(r["sum"])), jumlah=int(r["size"])) for k, r in g.iterrows()]
        self._note("cashier_breakdown", dict(cashier_id=cashier_id, metric=metric), out)
        return out

    def store_summary(self) -> list:
        """Findings per store: count, high-severity count, value at risk, and the cashiers involved. Worst store first."""
        agg: dict = {}
        for f in self.res["findings"]:
            st = f.get("store") or "(tanpa toko)"
            a = agg.setdefault(st, dict(toko=st, temuan=0, tinggi=0, nilai_rp=0, orang=set()))
            a["temuan"] += 1
            a["tinggi"] += f["severity"] == "Tinggi"
            a["nilai_rp"] += round(f["amount"])
            if f.get("subject_name"):
                a["orang"].add(f["subject_name"])
        out = sorted(({**a, "orang": sorted(a["orang"])[:5]} for a in agg.values()), key=lambda a: (-a["tinggi"], -a["nilai_rp"]))
        self._note("store_summary", {}, out)
        return out

    def summary(self) -> dict:
        """Overall audit summary: counts by severity, total value at risk, and the highest-risk cashiers."""
        fs = self.res["findings"]
        out = dict(periode=f"{self.res['meta']['date_from']} sampai {self.res['meta']['date_to']}", jumlah_order=self.res["meta"]["n_orders"],
                   temuan_per_tingkat={s: sum(1 for f in fs if f["severity"] == s) for s in ("Tinggi", "Sedang", "Rendah")},
                   nilai_berisiko_rp=round(sum(f["amount"] for f in fs)), kasir_risiko_tertinggi=self.res["risk"].head(5)[["cashier_id", "nama", "store", "risk", "rules"]].to_dict("records"))
        self._note("summary", {}, "ok")
        return out


PROMPT = """Kamu adalah asisten auditor internal untuk jaringan toko retail. Jawab dalam bahasa Indonesia, singkat dan konkret.
Aturan:
- Gunakan alat yang tersedia untuk mengambil fakta. Jangan mengarang angka, nama, atau id yang tidak ada di hasil alat.
- Sebut id temuan (misalnya F003) dan angka bukti pada jawabanmu.
- Temuan berarti "perlu diperiksa", bukan bukti kecurangan. Jangan menuduh orang. Gunakan kata "kemungkinan" atau "perlu diverifikasi".
- Akhiri dengan langkah verifikasi yang disarankan bila relevan. Jangan menambahkan tindakan yang berisiko (misalnya memecat) sebagai saran.
Pertanyaan auditor: {q}"""


def _facts(res: dict, q: str, tb: Toolbox) -> dict:
    """Run the obvious lookups up front (and log them), so the model usually needs a single round trip."""
    ql = q.lower()
    facts: dict = {"ringkasan": tb.summary()}
    for fid in re.findall(r"\bf\d{3}\b", ql)[:2]:
        facts[fid.upper()] = tb.get_finding(fid)
    cids = re.findall(r"\bk\d{2}\b", ql) + [r.cashier_id for r in res["risk"].itertuples() if r.nama.lower() in ql]
    for cid in list(dict.fromkeys(c.upper() for c in cids))[:2]:
        facts["kasir_" + cid] = tb.cashier_profile(cid)
        if any(w in ql for w in ("sku", "produk", "barang", "item")):
            facts["produk_" + cid] = {m: tb.cashier_breakdown(cid, m) for m in ("refund", "discount", "void")}
    if any(w in ql for w in ("toko", "store", "cabang", "outlet", "lokasi")):
        facts["per_toko"] = tb.store_summary()[:10]
    if len(facts) == 1:
        facts["temuan_tinggi"] = tb.list_findings(severity="Tinggi")[:8]
    return facts


def _gen(res: dict, q: str, tb: Toolbox, facts: dict, use_tools: bool, timeout: float) -> str:
    """One question to Gemini. With use_tools the model may call the toolbox for up to 3 rounds."""
    prompt = PROMPT.format(q=q) + "\n\nData yang sudah diambil:\n" + json.dumps(facts, ensure_ascii=False, default=str)[:9000]
    if use_tools:
        prompt += "\n\nPakai alat hanya bila data di atas belum cukup."
    contents: list = [{"role": "user", "parts": [{"text": prompt}]}]
    allowed = {"summary", "list_findings", "get_finding", "cashier_profile", "cashier_breakdown", "store_summary"}
    called = False
    for _ in range(3 if use_tools else 0):
        m = _chat(contents, TOOLS, timeout)
        parts = m.get("parts") or []
        calls = [p["functionCall"] for p in parts if p.get("functionCall")]
        if not calls:
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text
            break
        contents.append(m)  # keep the model turn exactly as returned (it may carry thought signatures)
        replies = []
        for fn in calls:
            name, args = fn.get("name", ""), fn.get("args") or {}
            try:
                out = getattr(tb, name)(**args) if name in allowed and isinstance(args, dict) else {"error": f"alat '{name}' tidak ada"}
            except TypeError as e:
                out = {"error": str(e)}
            raw = json.dumps(out, ensure_ascii=False, default=str)
            replies.append({"functionResponse": {"name": name, "response": {"result": json.loads(raw) if len(raw) <= 6000 else raw[:6000]}}})
        contents.append({"role": "user", "parts": replies})
        called = True
    m = _chat(contents, TOOLS if called else None, timeout, force_text=called)
    text = "".join(p.get("text", "") for p in (m.get("parts") or []) if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("model memberi jawaban kosong")
    return text


def _model_answer(res: dict, q: str, tb: Toolbox) -> str:
    """Attempt 1 lets the model call tools; attempt 2 is a single plain call on the prefetched facts."""
    facts = _facts(res, q, tb)
    try:
        return _pool.submit(_gen, res, q, tb, facts, True, _timeout()).result(timeout=_timeout() + 3)
    except Exception:
        return _pool.submit(_gen, res, q, tb, facts, False, _timeout()).result(timeout=_timeout() + 3)


def _rule_answer(res: dict, q: str, tb: Toolbox) -> str:
    ql = q.lower()
    ids = re.findall(r"\bf\d{3}\b", ql)
    cash = re.findall(r"\bk\d{2}\b", ql)
    names = [r.cashier_id for r in res["risk"].itertuples() if r.nama.lower() in ql]
    if ids:
        f = tb.get_finding(ids[0])
        if "error" in f:
            return "Temuan itu tidak ada. Coba lihat daftar temuan di halaman."
        return f"**{f['id']}: {f['judul']}** (tingkat {f['severity']}, skor {f['skor']})\n\n{f['detail']}\n\nLangkah verifikasi yang disarankan:\n" + "\n".join(f"- {s}" for s in f["langkah_verifikasi"])
    if cash or names:
        p = tb.cashier_profile((cash or names)[0])
        if "error" in p:
            return "Kasir itu tidak ditemukan."
        if any(w in ql for w in ("sku", "produk", "barang", "item")):
            cid = p["profil"]["cashier_id"]
            rules = p["profil"]["rules"]
            metric = "refund" if "refund" in rules else "discount" if "diskon" in rules else "void" if "void" in rules else "refund"
            top = tb.cashier_breakdown(cid, metric)
            if top:
                lab = {"refund": "refund", "discount": "diskon", "void": "void"}[metric]
                return f"Produk dengan nilai {lab} terbesar untuk {p['profil']['nama']} ({cid}):\n" + "\n".join(f"- {t['produk']} ({t['sku']}): {rp(t['nilai_rp'])}, {t['jumlah']} baris" for t in top)
        pr = p["profil"]
        orders = f"{pr['orders']:,}".replace(",", ".")
        lines = [f"**{pr['nama']} ({pr['cashier_id']}), {pr['store']}**: skor risiko {pr['risk']} ({pr['level']}), {orders} order."]
        if p["temuan"]:
            lines += [f"- {t['id']}: {t['judul']}" for t in p["temuan"]]
        else:
            lines.append("Tidak ada temuan untuk kasir ini.")
        m, med = p["metrik"], p["median_semua_kasir"]
        lines.append(f"Dibanding median semua kasir: refund {m['refund_pct']}% (median {med['refund_pct']}%), diskon {m['diskon_pct']}% ({med['diskon_pct']}%), void {m['void_pct']}% ({med['void_pct']}%).")
        return "\n".join(lines)
    if any(w in ql for w in ("toko", "store", "cabang", "outlet", "lokasi")):
        rows = tb.store_summary()[:5]
        if rows:
            lines = ["Toko dengan temuan terbanyak (urut dari yang paling bermasalah):"]
            lines += [f"- **{r['toko']}**: {r['temuan']} temuan ({r['tinggi']} tinggi), nilai sekitar {rp(r['nilai_rp'])}" + (f", terkait {', '.join(r['orang'])}" if r["orang"] else "") for r in rows]
            lines.append("Tanyakan nama toko atau id temuan untuk detailnya.")
            return "\n".join(lines)
    s = tb.summary()
    top = tb.list_findings(severity="Tinggi")[:5]
    lines = [f"Periode {s['periode']}, {s['jumlah_order']:,} order.".replace(",", "."),
             f"Temuan: {s['temuan_per_tingkat']['Tinggi']} tinggi, {s['temuan_per_tingkat']['Sedang']} sedang, {s['temuan_per_tingkat']['Rendah']} rendah. Nilai yang perlu ditelusuri sekitar Rp {s['nilai_berisiko_rp']:,}.".replace(",", "."),
             "Prioritas pemeriksaan:"] + [f"- {t['id']} ({t['nama']}, {t['toko']}): {t['judul']}" for t in top]
    lines.append("Tanyakan id temuan (mis. F003) atau nama kasir untuk penjelasan dan langkah verifikasi.")
    return "\n".join(lines)


def ask(res: dict, question: str) -> dict:
    tb = Toolbox(res)
    q = (question or "").strip()[:600]
    if not q:
        return dict(answer="Tulis pertanyaannya dulu.", source="aturan", steps=[])
    if enabled():
        try:
            ans = _pool.submit(_model_answer, res, q, tb).result(timeout=_timeout() * 2 + 10)
            return dict(answer=ans, source="asisten", steps=tb.log)
        except FutureTimeout:
            note, why = "Asisten lengkap tidak merespons, memakai jawaban standar.", "TimeoutError (Gemini lambat; naikkan AGENT_TIMEOUT_SECONDS atau pakai model lebih ringan)"
        except Exception as exc:
            note, why = "Asisten lengkap sedang tidak tersedia, memakai jawaban standar.", f"{type(exc).__name__}: {str(exc)[:160]}"
        tb2 = Toolbox(res)
        return dict(answer=f"{note} (penyebab teknis: {why})\n\n" + _rule_answer(res, q, tb2), source="aturan", steps=tb2.log)
    return dict(answer=_rule_answer(res, q, tb), source="aturan", steps=tb.log)
