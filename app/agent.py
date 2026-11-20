"""Audit assistant: an agent that investigates the findings with read-only tools and writes an evidence-based answer.

With an API key it lets the model choose which tools to call (function calling) and logs every call. Without a key,
or when the call fails, it answers from the same tools with fixed rules, so the page always works.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pandas as pd

DEFAULT_MODEL = "gemini-3.5-flash"
_pool = ThreadPoolExecutor(max_workers=2)

STEPS = {
    "refund_tinggi": ["Buka struk asli untuk 10 refund terbesar dan cocokkan dengan bukti retur fisik.", "Tanyakan alasan refund ke pelanggan lewat nomor di struk (sampel).", "Bandingkan dengan kasir lain di toko yang sama."],
    "refund_tanpa_order": ["Cari order_id di sistem kasir lain atau arsip. Jika tidak ada, anggap refund fiktif sampai terbukti sebaliknya.", "Periksa siapa yang menyetujui dan apakah uang keluar dari laci kas.", "Cek rekaman CCTV pada jam refund."],
    "refund_disetujui_sendiri": ["Cek pengaturan hak akses: kasir tidak boleh menyetujui refund-nya sendiri.", "Minta manajer menelaah ulang refund yang disetujui sendiri.", "Cocokkan dengan bukti retur fisik."],
    "refund_tanpa_persetujuan": ["Minta bukti persetujuan tertulis atau kode otorisasi.", "Perketat sistem agar refund besar wajib disetujui."],
    "refund_sebelum_order": ["Periksa jam sistem dan proses input manual.", "Cocokkan dengan log sistem."],
    "refund_terlambat": ["Periksa kebijakan batas waktu refund dan siapa yang mengizinkan pengecualian."],
    "diskon_tanpa_persetujuan": ["Minta bukti persetujuan manajer untuk tiap diskon.", "Periksa apakah diskon diberikan ke teman atau keluarga."],
    "diskon_di_batas": ["Cocokkan pelanggan pada struk dengan program promo yang berlaku.", "Periksa apakah diskon sengaja dipecah agar tidak perlu persetujuan."],
    "diskon_sering": ["Bandingkan dengan promo yang resmi berlaku pada periode itu.", "Tanyakan dasar pemberian diskon ke kasir."],
    "selisih_kas": ["Hitung kas bersama saksi pada penutupan berikutnya.", "Periksa kebiasaan membuka laci tanpa transaksi.", "Cek CCTV area kasir pada shift terkait."],
    "luar_jam_buka": ["Periksa log akses toko dan alarm pada jam itu.", "Cek apakah ada kegiatan resmi di luar jam (stok opname, event).", "Cek penggunaan akun kasir oleh orang lain."],
    "struk_ganda": ["Cocokkan dengan struk fisik: apakah pelanggan benar membeli dua kali.", "Periksa apakah salah satu struk dibatalkan setelah uang diterima."],
    "void_tinggi": ["Cek alasan void tiap struk dan siapa yang menyetujui.", "Cocokkan waktu void dengan pembayaran tunai."],
    "integritas_data": ["Periksa log perubahan total atau harga di sistem kasir.", "Cek apakah ada pengeditan manual setelah struk dibuat."],
    "susut_stok": ["Lakukan hitung ulang fisik untuk SKU tersebut.", "Periksa retur ke supplier, barang rusak, dan transfer antar toko yang belum dicatat.", "Tinjau akses gudang dan CCTV."],
    "pola_tidak_biasa": ["Lihat hari-hari yang ditandai model, lalu cocokkan dengan temuan aturan.", "Gunakan sebagai petunjuk arah pemeriksaan, bukan sebagai bukti."],
}


def _key() -> str | None:
    return os.getenv("INSIGHT_API_KEY") or os.getenv("GEMINI_API_KEY")


def _timeout() -> float:
    try:
        return max(5.0, float(os.getenv("AGENT_TIMEOUT_SECONDS", "45")))
    except ValueError:
        return 45.0


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


def _model_answer(res: dict, q: str, tb: Toolbox) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_key(), http_options=types.HttpOptions(timeout=int(_timeout() * 1000)))
    r = client.models.generate_content(
        model=os.getenv("INSIGHT_MODEL") or os.getenv("GEMINI_MODEL", DEFAULT_MODEL), contents=PROMPT.format(q=q),
        config=types.GenerateContentConfig(tools=[tb.summary, tb.list_findings, tb.get_finding, tb.cashier_profile], temperature=0.2))
    if not r.text:
        raise RuntimeError("empty answer")
    return r.text.strip()


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
    if _key():
        try:
            ans = _pool.submit(_model_answer, res, q, tb).result(timeout=_timeout() + 5)
            return dict(answer=ans, source="asisten", steps=tb.log)
        except FutureTimeout:
            note = "Asisten lengkap tidak merespons, memakai jawaban standar.\n\n"
        except Exception:
            note = "Asisten lengkap sedang tidak tersedia, memakai jawaban standar.\n\n"
        tb2 = Toolbox(res)
        return dict(answer=note + _rule_answer(res, q, tb2), source="aturan", steps=tb2.log)
    return dict(answer=_rule_answer(res, q, tb), source="aturan", steps=tb.log)
