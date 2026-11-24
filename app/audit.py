"""Audit engine: rule checks + an anomaly model, combined into findings and a risk score per cashier.

Every finding carries its evidence (order / refund ids and rows) so an auditor can verify it. Findings mean
"needs review", never "proven fraud".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = {
    "orders": ["order_id", "datetime", "store_id", "cashier_id", "subtotal", "discount", "total"],
    "items": ["order_id", "sku", "qty", "unit_price"],
    "refunds": ["refund_id", "order_id", "datetime", "store_id", "cashier_id", "amount"],
    "cash": ["date", "store_id", "cashier_id", "expected_cash", "actual_cash"],
    "employees": ["employee_id", "nama", "store_id", "peran"],
    "stores": ["store_id", "store_name"],
    "skus": ["sku", "name", "price", "unit_cost"],
    "opname": ["store_id", "sku", "system_qty", "counted_qty"],
    "customers": ["customer_id", "nama", "member_tier"],
}
LABELS = {"orders": "Order", "items": "Item order", "refunds": "Refund", "cash": "Setoran kas", "employees": "Karyawan",
          "stores": "Toko", "skus": "SKU", "opname": "Stok opname", "customers": "Pelanggan"}
MANDATORY = ["orders"]
AUDITS = ("kasir", "order")
# which audit a rule belongs to: "kasir" = how each cashier behaves compared with peers, "order" = errors, oddities and fraud signs in the order data itself
RULE_AUDITS = {
    "refund_tinggi": ["kasir"], "refund_disetujui_sendiri": ["kasir"], "refund_tanpa_persetujuan": ["kasir"], "diskon_sering": ["kasir"],
    "selisih_kas": ["kasir"], "luar_jam_buka": ["kasir"], "void_tinggi": ["kasir"], "pola_tidak_biasa": ["kasir"], "susut_stok": ["kasir"],
    "refund_tanpa_order": ["kasir", "order"], "refund_sebelum_order": ["kasir", "order"], "refund_terlambat": ["kasir", "order"], "struk_ganda": ["kasir", "order"],
    "integritas_data": ["order"], "id_ganda": ["order"], "angka_mustahil": ["order"], "nilai_tidak_wajar": ["order"], "harga_tidak_cocok": ["order"],
    "refund_melebihi_order": ["order"], "pelanggan_tidak_dikenal": ["order"], "pelanggan_refund_sering": ["order"],
}
DETECT_ORDER = ["orders", "refunds", "cash", "opname", "customers", "employees", "skus", "stores", "items"]
GUEST_IDS = {"", "GUEST", "WALKIN", "UMUM", "NON-MEMBER", "-", "NAN"}   # customer ids that mean "not a member"


class DataError(ValueError):
    """User-fixable data problem; message is shown to the user (Indonesian)."""


@dataclass
class Params:
    refund_window_days: int = 30
    dup_minutes: int = 5
    cash_tolerance: float = 20000      # shortage per closing above this counts
    shrink_pct: float = 15.0
    min_z: float = 3.0


def detect_kind(df: pd.DataFrame):
    cols = set(df.columns)
    for k in DETECT_ORDER:
        if set(REQUIRED[k]) <= cols:
            return k
    return None


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    mad = max(mad, s.std(ddof=0) * 0.25, 1e-9)
    return (s - med) / mad


def sev(score: float) -> str:
    return "Tinggi" if score >= 70 else "Sedang" if score >= 40 else "Rendah"


def _rp(x) -> str:
    return "Rp " + f"{int(round(x)):,}".replace(",", ".")


def _pct(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", ",") + "%"


def _rows(df: pd.DataFrame, cols: list[str], n: int = 500) -> list[dict]:
    out = df[[c for c in cols if c in df.columns]].head(n).copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M")
    return out.astype(object).where(out.notna(), None).to_dict("records")


def clean(t: dict[str, pd.DataFrame]) -> tuple[dict, list[str]]:
    w: list[str] = []
    if "orders" not in t:
        raise DataError("File order belum ada. Unggah sheet Order (order_id, datetime, store_id, cashier_id, subtotal, discount, total).")
    o = t["orders"].copy()
    o["datetime"] = pd.to_datetime(o["datetime"], errors="coerce")
    bad = int(o.datetime.isna().sum())
    if bad:
        w.append(f"Order: {bad} baris dengan tanggal tidak terbaca dilewati.")
        o = o.dropna(subset=["datetime"])
    for c in ("order_id", "store_id", "cashier_id"):
        o[c] = o[c].astype(str).str.strip()
    if "customer_id" in o.columns:
        o["customer_id"] = o.customer_id.fillna("").astype(str).str.strip()
    for c in ("subtotal", "discount", "total"):
        o[c] = pd.to_numeric(o[c], errors="coerce").fillna(0)
    for c, d in (("voided", 0),):
        if c not in o.columns:
            o[c] = d
    o["voided"] = pd.to_numeric(o.voided, errors="coerce").fillna(0).astype(int)
    if "shift" not in o.columns:
        o["shift"] = np.where(o.datetime.dt.hour < 16, "pagi", "sore")
    if o.order_id.duplicated().any():
        w.append(f"Order: {int(o.order_id.duplicated().sum())} order_id ganda, baris pertama dipakai.")
        o = o.drop_duplicates("order_id")
    if len(o) < 200:
        raise DataError(f"Data order hanya {len(o)} baris. Minimal 200 agar pola kasir bisa dibandingkan.")
    out = {"orders": o}
    for k in ("items", "refunds", "cash", "employees", "stores", "skus", "opname", "customers"):
        if k in t:
            d = t[k].copy()
            for c in ("store_id", "cashier_id", "order_id", "sku", "employee_id", "approved_by", "customer_id"):
                if c in d.columns:
                    d[c] = d[c].fillna("").astype(str).str.strip()
            if "datetime" in d.columns:
                d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
            out[k] = d
        else:
            w.append(f"{LABELS[k]} tidak diunggah: pemeriksaan yang memakainya dilewati.")
    return out, w


def run_audit(raw: dict[str, pd.DataFrame], P: Params | None = None) -> dict:
    P = P or Params()
    t, warnings = clean(raw)
    o = t["orders"]
    o_ok = o[o.voided == 0]
    store_name = dict(zip(t["stores"].store_id, t["stores"].store_name)) if "stores" in t else {}
    emp_name = dict(zip(t["employees"].employee_id, t["employees"].nama)) if "employees" in t else {}
    cashier_store = o.groupby("cashier_id").store_id.agg(lambda s: s.mode().iat[0]).to_dict()
    cashiers = sorted(o.cashier_id.unique())
    n_orders = o.groupby("cashier_id").size()
    findings: list[dict] = []

    def add(rule, subject, title, detail, score, evidence, cols, amount=0.0, metric=None, peer=None, kind="cashier"):
        findings.append(dict(
            id=f"F{len(findings) + 1:03d}", rule=rule, audits=RULE_AUDITS[rule], subject=subject, subject_type=kind,
            subject_name=emp_name.get(subject, subject) if kind == "cashier" else subject,
            store=store_name.get(cashier_store.get(subject, ""), cashier_store.get(subject, "")) if kind == "cashier" else "",
            title=title, detail=detail, score=int(round(min(100, max(1, score)))), severity=sev(score),
            amount=float(amount), metric=metric, peer=peer, evidence_count=int(len(evidence)),
            evidence=_rows(evidence, cols, 500), evidence_cols=[c for c in cols if c in evidence.columns]))

    # ---------------------------------------------------------------- R1 refund rate outlier
    rf = t.get("refunds")
    if rf is not None and len(rf):
        rcount = rf.groupby("cashier_id").size().reindex(cashiers).fillna(0)
        rate = rcount / n_orders.reindex(cashiers) * 100
        z = robust_z(rate)
        med = rate.median()
        for c in cashiers:
            if z[c] >= P.min_z and rate[c] >= 1.5 * med and rcount[c] >= 5:
                ev = rf[rf.cashier_id == c].sort_values("amount", ascending=False)
                add("refund_tinggi", c, f"Tingkat refund {_pct(rate[c])} jauh di atas rekan kerja",
                    f"{int(rcount[c])} refund dari {int(n_orders[c])} order ({_pct(rate[c])}). Median kasir lain {_pct(med)}. Total nilai refund {_rp(ev.amount.sum())}.",
                    45 + 8 * z[c], ev, ["refund_id", "order_id", "datetime", "sku", "amount", "reason", "approved_by"], ev.amount.sum(), round(rate[c], 2), round(med, 2))
        # ---------------- R2 refunds that do not match an order
        known = o.set_index("order_id")
        miss = rf[~rf.order_id.isin(known.index)]
        for c, g in miss.groupby("cashier_id"):
            add("refund_tanpa_order", c, f"{len(g)} refund tidak punya order asal",
                f"Refund merujuk order_id yang tidak ada di data order, total {_rp(g.amount.sum())}. Bisa refund fiktif atau salah input.",
                55 + min(40, 6 * len(g)), g, ["refund_id", "order_id", "datetime", "customer_id", "sku", "amount", "approved_by"], g.amount.sum(), len(g), 0)
        m = rf.merge(o[["order_id", "datetime"]].rename(columns={"datetime": "order_time"}), on="order_id")
        early = m[m.datetime < m.order_time]
        late = m[(m.datetime - m.order_time).dt.days > P.refund_window_days]
        for lab, g, why in (("refund_sebelum_order", early, "diproses sebelum order dibuat"), ("refund_terlambat", late, f"diproses lebih dari {P.refund_window_days} hari setelah order")):
            for c, gg in g.groupby("cashier_id"):
                add(lab, c, f"{len(gg)} refund {why}", f"Urutan waktu tidak wajar. Total {_rp(gg.amount.sum())}.", 40 + min(40, 8 * len(gg)), gg,
                    ["refund_id", "order_id", "datetime", "order_time", "amount"], gg.amount.sum(), len(gg), 0)
        # ---------------- R3 self-approved / unapproved refunds
        if "approved_by" in rf.columns:
            selfap = rf[(rf.approved_by == rf.cashier_id) & (rf.approved_by != "")]
            tot_r = rf.groupby("cashier_id").size()
            for c, g in selfap.groupby("cashier_id"):
                if len(g) >= 3:
                    add("refund_disetujui_sendiri", c, f"{len(g)} refund disetujui oleh kasir yang sama ({_pct(len(g) / tot_r[c] * 100, 0)} dari refund-nya)",
                        f"Pemisahan tugas dilanggar: kasir memproses sekaligus menyetujui. Nilai {_rp(g.amount.sum())}.", 60 + min(35, 3 * len(g)), g,
                        ["refund_id", "order_id", "datetime", "amount", "reason", "approved_by"], g.amount.sum(), len(g), 0)
            noap = rf[(rf.approved_by == "") & (rf.amount >= 100000)]
            for c, g in noap.groupby("cashier_id"):
                if len(g) >= 5:
                    add("refund_tanpa_persetujuan", c, f"{len(g)} refund bernilai besar tanpa persetujuan",
                        f"Refund di atas Rp 100.000 tanpa penyetuju, total {_rp(g.amount.sum())}.", 35 + min(40, 3 * len(g)), g,
                        ["refund_id", "order_id", "datetime", "amount", "approved_by"], g.amount.sum(), len(g), 0)

    # ---------------------------------------------------------------- R4 discounts
    d = o_ok[o_ok.subtotal > 0].copy()
    d["pct"] = d.discount / d.subtotal * 100
    dd = d[d.discount > 0]
    drate = (dd.groupby("cashier_id").size().reindex(cashiers).fillna(0) / n_orders.reindex(cashiers) * 100)
    z = robust_z(drate)
    for c in cashiers:
        if z[c] >= P.min_z + 1 and drate[c] > 1.5 * drate.median():
            g = dd[dd.cashier_id == c]
            add("diskon_sering", c, f"Memberi diskon pada {_pct(drate[c], 0)} order",
                f"Rekan kerja rata-rata {_pct(drate.median(), 0)}. Total diskon {_rp(g.discount.sum())}.", 40 + 6 * z[c], g,
                ["order_id", "datetime", "subtotal", "discount", "pct"], g.discount.sum(), round(drate[c], 1), round(drate.median(), 1))

    # ---------------------------------------------------------------- R5 cash variance
    kas = t.get("cash")
    if kas is not None and len(kas):
        kas = kas.assign(selisih=kas.actual_cash - kas.expected_cash)
        short = kas[kas.selisih <= -P.cash_tolerance]
        tot = kas.groupby("cashier_id").size()
        for c, g in short.groupby("cashier_id"):
            frac = len(g) / tot[c] * 100
            if len(g) >= 4 and frac >= 8:
                where = ""
                if "shift" in g.columns and g["shift"].nunique() == 1:
                    where = f" Semuanya terjadi di shift {g['shift'].iat[0]}."
                add("selisih_kas", c, f"Kas kurang di {len(g)} penutupan ({_pct(frac, 0)})",
                    f"Total kekurangan {_rp(-g.selisih.sum())}, rata-rata {_rp(-g.selisih.mean())} per kejadian.{where}", 50 + min(45, 2.5 * len(g)), g.sort_values("selisih"),
                    ["date", "store_id", "shift", "expected_cash", "actual_cash", "selisih"], -g.selisih.sum(), len(g), 0)

    # ---------------------------------------------------------------- R6 off-hours
    if "stores" in t and {"jam_buka", "jam_tutup"} <= set(t["stores"].columns):
        hrs = t["stores"].set_index("store_id")[["jam_buka", "jam_tutup"]].apply(pd.to_numeric, errors="coerce")
        oo = o.join(hrs, on="store_id")
        h = oo.datetime.dt.hour
        off = oo[(h < oo.jam_buka) | (h >= oo.jam_tutup)]
        for c, g in off.groupby("cashier_id"):
            if len(g) >= 3:
                add("luar_jam_buka", c, f"{len(g)} transaksi di luar jam buka toko",
                    f"Transaksi tercatat sebelum toko buka atau setelah tutup, nilai {_rp(g.total.sum())}.", 45 + min(50, 4 * len(g)), g,
                    ["order_id", "datetime", "store_id", "total"], g.total.sum(), len(g), 0)

    # ---------------------------------------------------------------- R7 duplicates and R8 voids
    if "items" in t:
        sig = t["items"].assign(s=lambda x: x.sku + "x" + x.qty.astype(str)).sort_values("sku").groupby("order_id").s.agg("|".join)
        dp = o_ok.join(sig.rename("sig"), on="order_id").dropna(subset=["sig"]).sort_values(["cashier_id", "datetime"])
        same = (dp.cashier_id == dp.cashier_id.shift()) & (dp.sig == dp.sig.shift()) & (dp.total == dp.total.shift()) & \
               ((dp.datetime - dp.datetime.shift()).dt.total_seconds() <= P.dup_minutes * 60)
        dups = dp[same]
        drate_ = dups.groupby("cashier_id").size().reindex(cashiers).fillna(0) / n_orders.reindex(cashiers) * 100
        zd = robust_z(drate_)
        for c, g in dups.groupby("cashier_id"):
            if len(g) >= 5 and zd[c] >= P.min_z:
                add("struk_ganda", c, f"{len(g)} struk kembar berturutan",
                    f"Struk dengan isi dan total sama diproses ulang dalam {P.dup_minutes} menit ({_pct(drate_[c], 2)} dari order, rekan kerja median {_pct(drate_.median(), 2)}). Nilai {_rp(g.total.sum())}.",
                    40 + 7 * zd[c], g, ["order_id", "datetime", "customer_id", "total", "payment_method"], g.total.sum(), round(drate_[c], 2), round(drate_.median(), 2))
    vrate = o.groupby("cashier_id").voided.mean().reindex(cashiers) * 100
    z = robust_z(vrate)
    for c in cashiers:
        nv = int(o[(o.cashier_id == c)].voided.sum())
        if z[c] >= P.min_z and nv >= 8 and vrate[c] > 1.5 * vrate.median():
            g = o[(o.cashier_id == c) & (o.voided == 1)]
            add("void_tinggi", c, f"Tingkat void {_pct(vrate[c])} jauh di atas rekan kerja",
                f"{nv} struk dibatalkan. Median kasir lain {_pct(vrate.median())}. Nilai struk yang dibatalkan {_rp(g.total.sum())}.", 40 + 7 * z[c], g,
                ["order_id", "datetime", "total", "payment_method"], g.total.sum(), round(vrate[c], 2), round(vrate.median(), 2))

    # ---------------------------------------------------------------- R9 integrity
    mism = o[(o.total - (o.subtotal - o.discount)).abs() > 1].assign(selisih=lambda x: x.total - (x.subtotal - x.discount))
    if "items" in t:
        isum = t["items"].assign(v=lambda x: x.qty * x.unit_price).groupby("order_id").v.sum()
        x = o.join(isum.rename("isum"), on="order_id").dropna(subset=["isum"])
        mm = x[(x.subtotal - x.isum).abs() > 1].assign(selisih=lambda d: d.subtotal - d.isum)
        mism = pd.concat([mism, mm]).drop_duplicates("order_id")
    if len(mism):
        top = mism.cashier_id.value_counts()
        add("integritas_data", "Integritas data", f"{len(mism)} struk dengan angka yang tidak konsisten",
            f"Total tidak sama dengan subtotal dikurangi diskon, atau subtotal tidak sama dengan jumlah item. Selisih total {_rp(mism.selisih.abs().sum())}. Terbanyak pada kasir {', '.join(f'{emp_name.get(k, k)} ({v})' for k, v in top.head(3).items())}.",
            40 + min(35, 3 * len(mism)), mism, ["order_id", "datetime", "cashier_id", "customer_id", "subtotal", "discount", "total", "selisih"], mism.selisih.abs().sum(), len(mism), 0, kind="data")

    # ---------------------------------------------------------------- order-data checks (audit "order")
    ro = raw["orders"]
    ids = ro["order_id"].astype(str).str.strip()
    dupm = ids.duplicated(keep=False)
    if dupm.any():
        g = ro[dupm].assign(order_id=ids[dupm]).sort_values("order_id").copy()
        g["datetime"] = pd.to_datetime(g["datetime"], errors="coerce")
        n_ids = int(g.order_id.nunique())
        add("id_ganda", "Data order", f"{n_ids} order_id muncul lebih dari sekali",
            f"{len(g)} baris memakai {n_ids} nomor order yang sama. Bisa input ganda atau struk yang disalin. Hanya baris pertama tiap nomor yang dipakai dalam pemeriksaan.",
            50 + min(40, 3 * n_ids), g, ["order_id", "datetime", "cashier_id", "subtotal", "discount", "total"], pd.to_numeric(g.total, errors="coerce").fillna(0).sum(), n_ids, 0, kind="data")

    reason = np.select([(o.subtotal < 0) | (o.total < 0) | (o.discount < 0), o.discount > o.subtotal + 1], ["ada nilai negatif", "diskon lebih besar dari subtotal"], "")
    imp = o.assign(alasan=reason)
    imp = imp[imp.alasan != ""]
    if len(imp):
        add("angka_mustahil", "Data order", f"{len(imp)} struk dengan angka yang tidak masuk akal",
            f"Nilai negatif atau diskon melebihi subtotal. Bisa salah input, retur yang dicatat sebagai penjualan, atau manipulasi angka. Terbanyak pada kasir "
            + ", ".join(f"{emp_name.get(k, k)} ({v})" for k, v in imp.cashier_id.value_counts().head(3).items()) + ".",
            55 + min(40, 5 * len(imp)), imp, ["order_id", "datetime", "cashier_id", "customer_id", "subtotal", "discount", "total", "alasan"], imp.total.abs().sum(), len(imp), 0, kind="data")

    pos = o_ok[o_ok.total > 0].copy()
    if len(pos):
        pos["median_toko"] = pos.groupby("store_id").total.transform("median")
        pos["z"] = np.log(pos.total).groupby(pos.store_id).transform(robust_z)
        big = pos[(pos.z >= 5) & (pos.total >= 5 * pos.median_toko)].sort_values("total", ascending=False)
        if len(big):
            add("nilai_tidak_wajar", "Data order", f"{len(big)} struk bernilai jauh di atas kebiasaan tokonya",
                f"Nilai struk setidaknya 5 kali median tokonya dan sangat jauh dari sebaran normal. Bisa pembelian besar yang sah atau salah ketik angka. Total {_rp(big.total.sum())}.",
                30 + min(45, 6 * len(big)), big, ["order_id", "datetime", "store_id", "cashier_id", "customer_id", "total", "median_toko"], big.total.sum(), len(big), 0, kind="data")

    if "items" in t and "skus" in t:
        it = t["items"].merge(t["skus"][["sku", "name", "price"]].rename(columns={"name": "produk", "price": "harga_master"}), on="sku", how="inner")
        for c in ("unit_price", "harga_master", "qty"):
            it[c] = pd.to_numeric(it[c], errors="coerce")
        it = it.dropna(subset=["unit_price", "harga_master"])
        pb = it[(it.unit_price - it.harga_master).abs() > np.maximum(1, 0.05 * it.harga_master)].assign(selisih=lambda d: d.unit_price - d.harga_master)
        if len(pb):
            top = pb.produk.value_counts().head(3)
            add("harga_tidak_cocok", "Data order", f"{len(pb)} baris item dengan harga berbeda dari harga master",
                f"Harga di struk menyimpang lebih dari 5% dari harga master SKU. Bisa promo yang tidak tercatat, harga diubah manual, atau data master usang. Terbanyak: "
                + ", ".join(f"{k} ({v})" for k, v in top.items()) + ".",
                35 + min(45, 3 * len(pb)), pb, ["order_id", "sku", "produk", "qty", "unit_price", "harga_master", "selisih"], (pb.selisih.abs() * pb.qty.fillna(1)).sum(), len(pb), 0, kind="data")

    if rf is not None and len(rf):
        rs = rf.groupby("order_id").amount.sum().rename("total_refund")
        xo = o.join(rs, on="order_id", how="inner")
        over = xo[xo.total_refund > xo.subtotal + 1].assign(kelebihan=lambda d: d.total_refund - d.subtotal)
        if len(over):
            add("refund_melebihi_order", "Data order", f"{len(over)} order di-refund lebih besar dari nilainya",
                f"Total refund lebih besar dari nilai belanja order asalnya (sebelum diskon), selisih {_rp(over.kelebihan.sum())}. Mustahil terjadi pada refund yang wajar: bisa refund ganda, salah input, atau refund fiktif.",
                55 + min(40, 6 * len(over)), over, ["order_id", "datetime", "cashier_id", "subtotal", "total_refund", "kelebihan"], over.kelebihan.sum(), len(over), 0, kind="data")

    cu = t.get("customers")
    if "customer_id" in o.columns and cu is not None and len(cu):
        names = dict(zip(cu.customer_id, cu.nama))
        member = ~o.customer_id.str.upper().isin(GUEST_IDS)
        unk = o[member & ~o.customer_id.isin(set(cu.customer_id))]
        if len(unk):
            add("pelanggan_tidak_dikenal", "Data order", f"{len(unk)} order memakai ID pelanggan yang tidak terdaftar",
                f"ID pelanggan di struk tidak ada di master pelanggan ({unk.customer_id.nunique()} ID berbeda, tersebar di {unk.cashier_id.nunique()} kasir). Bisa salah ketik, pelanggan fiktif untuk mengejar poin atau diskon, atau master belum diperbarui. Nilai {_rp(unk.total.sum())}.",
                35 + min(45, 2 * len(unk)), unk, ["order_id", "datetime", "store_id", "cashier_id", "customer_id", "total"], unk.total.sum(), len(unk), 0, kind="data")
        if rf is not None and len(rf):
            rc = rf.drop(columns=["customer_id"], errors="ignore").merge(o[["order_id", "customer_id"]], on="order_id", how="inner")
            rc = rc[~rc.customer_id.str.upper().isin(GUEST_IDS)]
            cnt = rc.groupby("customer_id").size()
            n_cust_orders = o[member].groupby("customer_id").size()
            overall = float(len(rc) / max(int(member.sum()), 1) * 100)
            for cid, n_ in cnt[cnt >= 6].sort_values(ascending=False).items():
                rate = n_ / max(int(n_cust_orders.get(cid, n_)), 1) * 100
                if rate < max(20.0, 5 * overall):
                    continue
                g = rc[rc.customer_id == cid].sort_values("datetime")
                by = g.cashier_id.value_counts()
                top_c, top_n = by.index[0], int(by.iloc[0])
                label = f"{cid} ({names[cid]})" if cid in names else cid
                add("pelanggan_refund_sering", label, f"Pelanggan {label} melakukan {int(n_)} refund",
                    f"{int(n_)} refund dari {int(n_cust_orders.get(cid, n_))} order pelanggan ini ({_pct(rate, 0)}), padahal rata-rata semua pelanggan hanya {_pct(overall)}. {top_n} refund diproses kasir {emp_name.get(top_c, top_c)} ({_pct(top_n / n_ * 100, 0)}). Total nilai {_rp(g.amount.sum())}. Pola ini bisa berarti pelanggan dan kasir bekerja sama, atau pelanggan yang menyalahgunakan kebijakan refund.",
                    45 + min(40, 3 * int(n_)) + (10 if top_n / n_ >= 0.7 else 0), g, ["refund_id", "order_id", "datetime", "cashier_id", "amount", "reason", "approved_by"],
                    g.amount.sum(), round(rate, 1), round(overall, 1), kind="data")

    # ---------------------------------------------------------------- R10 stock shrinkage
    op = t.get("opname")
    if op is not None and len(op):
        sk = t["skus"].set_index("sku") if "skus" in t else None
        op = op.assign(selisih=op.counted_qty - op.system_qty)
        op["pct"] = np.where(op.system_qty > 0, -op.selisih / op.system_qty * 100, 0)
        op["nilai"] = op.selisih.clip(upper=0).abs() * (op.sku.map(sk.unit_cost) if sk is not None else 0)
        op["produk"] = op.sku.map(sk.name) if sk is not None else op.sku
        hot = op[(op.pct >= P.shrink_pct) & (-op.selisih >= 5)]
        for st, g in hot.groupby("store_id"):
            add("susut_stok", store_name.get(st, st), f"{len(g)} SKU dengan susut stok di atas {P.shrink_pct:.0f}%",
                f"Stok fisik lebih sedikit dari sistem, nilai {_rp(g.nilai.sum())}. Susut yang terpusat pada produk bernilai tinggi menandakan kemungkinan kehilangan.",
                45 + min(50, 12 * len(g)), g.sort_values("pct", ascending=False), ["store_id", "sku", "produk", "system_qty", "counted_qty", "selisih", "pct", "nilai"],
                g.nilai.sum(), len(g), 0, kind="store")

    # ---------------------------------------------------------------- anomaly model on cashier-day behaviour
    feats = _features(o, rf, kas, t, P)
    ml = _anomaly(feats)
    for c, info in ml["cashiers"].items():
        if info["flag_share"] >= 0.25 and info["flag_days"] >= 6:
            ev = pd.DataFrame(info["top_days"])
            add("pola_tidak_biasa", c, f"Perilaku harian berbeda dari pola normal di {info['flag_days']} hari ({_pct(info['flag_share'] * 100, 0)})",
                "Model mendeteksi kombinasi angka yang tidak biasa. Ciri yang paling menonjol: " + ", ".join(info["drivers"]) + ".",
                35 + 60 * info["flag_share"], ev, ["tanggal", "skor", "penyebab"], 0, round(info["flag_share"] * 100, 1), None)

    # ---------------------------------------------------------------- risk per cashier (one table per audit)
    def make_risk(fl: list[dict]) -> pd.DataFrame:
        fdf = pd.DataFrame(fl) if fl else pd.DataFrame(columns=["subject", "score", "subject_type", "rule", "amount"])
        rows = []
        for c in cashiers:
            fs = fdf[(fdf.subject == c) & (fdf.subject_type == "cashier")] if len(fdf) else fdf
            scores = sorted(fs.score.tolist(), reverse=True)
            risk = 0.0
            for i, s_ in enumerate(scores):
                risk += s_ * (0.6 ** i)
            risk = min(100.0, risk)
            rows.append(dict(cashier_id=c, nama=emp_name.get(c, c), store=store_name.get(cashier_store[c], cashier_store[c]), orders=int(n_orders[c]),
                             findings=len(fs), risk=round(risk), level=sev(risk) if risk >= 40 else "Normal",
                             amount=float(fs.amount.sum()) if len(fs) else 0.0, rules=", ".join(sorted(set(fs.rule))) if len(fs) else ""))
        return pd.DataFrame(rows).sort_values(["risk", "amount"], ascending=False).reset_index(drop=True)

    risk_df = make_risk(findings)
    risk_by = {m: make_risk([f for f in findings if m in f["audits"]]) for m in AUDITS}
    findings.sort(key=lambda f: (-f["score"], -f["amount"]))
    for i, f in enumerate(findings, 1):
        f["id"] = f"F{i:03d}"

    peers = pd.DataFrame({
        "cashier_id": cashiers, "nama": [emp_name.get(c, c) for c in cashiers],
        "refund_pct": [round(float(((rf.cashier_id == c).sum() / n_orders[c] * 100) if rf is not None else 0), 2) for c in cashiers],
        "diskon_pct": [round(float(drate.get(c, 0)), 1) for c in cashiers],
        "void_pct": [round(float(vrate[c]), 2) for c in cashiers],
        "kas_kurang": [int((short.cashier_id == c).sum()) if kas is not None and len(kas) else 0 for c in cashiers],
    })
    meta = dict(n_orders=int(len(o)), n_cashiers=len(cashiers), n_stores=int(o.store_id.nunique()), date_from=str(o.datetime.min().date()),
                date_to=str(o.datetime.max().date()), revenue=float(o_ok.total.sum()), warnings=warnings,
                params=dict(refund_window_days=P.refund_window_days),
                checks=[k for k, v in (("refunds", rf), ("cash", kas), ("stores", t.get("stores")), ("items", t.get("items")), ("opname", op)) if v is not None])
    return dict(findings=findings, risk=risk_df, risk_by=risk_by, peers=peers, meta=meta, ml=ml, tables=t, feats=feats)


def select(res: dict, mode: str) -> dict:
    """The view of one audit: only its findings (renumbered F001...) and the cashier risk table computed from them."""
    if mode not in AUDITS:
        raise DataError("Jenis audit tidak dikenal. Pilih audit kasir atau audit data order.")
    fs = [dict(f) for f in res["findings"] if mode in f["audits"]]
    for i, f in enumerate(fs, 1):
        f["id"] = f"F{i:03d}"
    return {**res, "findings": fs, "risk": res["risk_by"][mode], "mode": mode}


# ---------------------------------------------------------------- features + model
FEATURES = ["orders", "avg_total", "refund_rate", "discount_share", "void_share", "offhours_share", "cash_var", "approx_dups"]
FEATURE_LABELS = {"orders": "jumlah order", "avg_total": "rata-rata nilai struk", "refund_rate": "tingkat refund", "discount_share": "porsi diskon",
                  "void_share": "porsi void", "offhours_share": "porsi di luar jam buka", "cash_var": "selisih kas", "approx_dups": "struk kembar"}


def _features(o, rf, kas, t, P) -> pd.DataFrame:
    x = o.assign(date=o.datetime.dt.date)
    g = x.groupby(["cashier_id", "date"])
    f = pd.DataFrame({
        "orders": g.size(),
        "avg_total": g.total.mean(),
        "discount_share": g.apply(lambda d: (d.discount > 0).mean(), include_groups=False),
        "void_share": g.voided.mean(),
    })
    if "stores" in t and {"jam_buka", "jam_tutup"} <= set(t["stores"].columns):
        hrs = t["stores"].set_index("store_id")[["jam_buka", "jam_tutup"]].apply(pd.to_numeric, errors="coerce")
        xo = x.join(hrs, on="store_id")
        h = xo.datetime.dt.hour
        f["offhours_share"] = ((h < xo.jam_buka) | (h >= xo.jam_tutup)).groupby([xo.cashier_id, xo.date]).mean()
    else:
        f["offhours_share"] = 0.0
    if rf is not None and len(rf):
        r = rf.assign(date=rf.datetime.dt.date).groupby(["cashier_id", "date"]).size()
        f["refund_rate"] = (r.reindex(f.index).fillna(0) / f.orders)
    else:
        f["refund_rate"] = 0.0
    if kas is not None and len(kas):
        k = kas.assign(selisih=kas.actual_cash - kas.expected_cash, date=pd.to_datetime(kas["date"]).dt.date).groupby(["cashier_id", "date"]).selisih.sum()
        f["cash_var"] = k.reindex(f.index).fillna(0)
    else:
        f["cash_var"] = 0.0
    s = x.sort_values(["cashier_id", "datetime"])
    dup = (s.cashier_id == s.cashier_id.shift()) & (s.total == s.total.shift()) & ((s.datetime - s.datetime.shift()).dt.total_seconds() <= P.dup_minutes * 60)
    f["approx_dups"] = dup.groupby([s.cashier_id, s.date]).mean().reindex(f.index).fillna(0)
    return f.reset_index().fillna(0)


def _anomaly(f: pd.DataFrame) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    X = RobustScaler().fit_transform(f[FEATURES])
    X = np.clip(X, -15, 15)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=7).fit(X)
    score = -model.score_samples(X)            # higher = more unusual
    thr = np.quantile(score, 0.93)
    f = f.assign(skor=score, flag=score >= thr)
    med, mad = f[FEATURES].median(), (f[FEATURES] - f[FEATURES].median()).abs().median() * 1.4826 + 1e-9
    out = {}
    for c, g in f.groupby("cashier_id"):
        fl = g[g.flag]
        zs = ((fl[FEATURES] - med) / mad) if len(fl) else None
        drivers = []
        if zs is not None:
            mean_z = zs.mean().sort_values(ascending=False)
            drivers = [FEATURE_LABELS[k] for k, v in mean_z.items() if v > 1.5][:3]
        days = []
        for r in fl.sort_values("skor", ascending=False).head(15).itertuples():
            zr = ((pd.Series(r._asdict())[FEATURES].astype(float) - med) / mad).sort_values(ascending=False)
            days.append(dict(tanggal=str(r.date), skor=round(float(r.skor), 3), penyebab=", ".join([FEATURE_LABELS[k] for k, v in zr.items() if v > 1.5][:2]) or "-"))
        out[c] = dict(flag_days=int(len(fl)), flag_share=float(len(fl) / len(g)), mean_score=float(g.skor.mean()), drivers=drivers or ["kombinasi beberapa angka"], top_days=days)
    return dict(cashiers=out, threshold=float(thr), n_days=int(len(f)))
