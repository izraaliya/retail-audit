"""Generate a complete DUMMY order dataset for a beauty retail chain, with fraud / error patterns planted on purpose.

No public dataset has cashier, discount, void, refund-approval and cash-closing fields together (real POS fraud data is
confidential), so this one is synthetic. Writes to sample_data/:
  Data_Order_Lengkap.xlsx   Order, Item, Refund, Setoran_Kas
  Data_Master.xlsx          Karyawan, Toko, SKU, Opname
  Label_Anomali.csv         ground truth of what was planted (for evaluating the detector; NOT uploaded to the app)
Run: python scripts/generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl.styles import Alignment, Font, PatternFill

rng = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent.parent / "sample_data"
OUT.mkdir(exist_ok=True)
END, DAYS = pd.Timestamp("2026-11-19"), 60
dates = pd.date_range(END - pd.Timedelta(days=DAYS - 1), END)

stores = pd.DataFrame([
    ("ST01", "Jakarta - Grand Mall", 1.45, 10, 22), ("ST02", "Jakarta - SCBD Office", 1.15, 10, 21),
    ("ST03", "Surabaya - Galaxy Mall", 1.10, 10, 22), ("ST04", "Bandung - Dago Street", 0.95, 10, 21),
    ("ST05", "Yogyakarta - Malioboro", 0.85, 9, 21), ("ST06", "Medan - Sun Plaza", 0.80, 10, 22)],
    columns=["store_id", "store_name", "scale", "jam_buka", "jam_tutup"])

sku_rows = [
    ("SK001", "Gentle Foaming Cleanser 100ml", "Skincare", 65000, 28000, 5.0), ("SK002", "Hydrating Toner 150ml", "Skincare", 89000, 38000, 3.4),
    ("SK003", "Niacinamide 10% Serum 30ml", "Skincare", 119000, 47000, 4.6), ("SK004", "Vitamin C Brightening Serum 20ml", "Skincare", 139000, 55000, 3.2),
    ("SK005", "Ceramide Barrier Moisturizer 50g", "Skincare", 109000, 44000, 4.1), ("SK006", "Daily Sunscreen SPF50 40ml", "Skincare", 99000, 40000, 6.2),
    ("SK007", "Clay Mask Purifying 100g", "Skincare", 85000, 33000, 1.9), ("SK008", "Eye Cream Peptide 15ml", "Skincare", 149000, 58000, 1.3),
    ("SK009", "Acne Patch (36 pcs)", "Skincare", 49000, 17000, 4.8), ("SK010", "Body Lotion Shea 250ml", "Skincare", 69000, 28000, 3.3),
    ("MU001", "Lip Tint Velvet", "Makeup", 79000, 30000, 7.0), ("MU002", "Matte Lip Cream", "Makeup", 89000, 34000, 5.4),
    ("MU003", "Cushion Foundation SPF35", "Makeup", 189000, 74000, 4.2), ("MU004", "Concealer Full Cover", "Makeup", 99000, 38000, 3.3),
    ("MU005", "Loose Setting Powder", "Makeup", 119000, 46000, 2.7), ("MU006", "Blush Cream Pot", "Makeup", 85000, 32000, 3.1),
    ("MU007", "Waterproof Mascara Volume", "Makeup", 89000, 33000, 4.9), ("MU008", "Eyeliner Pen Ultra Black", "Makeup", 59000, 21000, 5.2),
    ("MU009", "Eyeshadow Palette Nude 9", "Makeup", 149000, 57000, 1.9), ("MU010", "Brush Set Essential (5)", "Makeup", 129000, 48000, 1.1),
]
skus = pd.DataFrame(sku_rows, columns=["sku", "name", "category", "price", "unit_cost", "popularity"])
pop = (skus.popularity / skus.popularity.sum()).to_numpy()

# employees: 3 cashiers per store (+1 manager). Some are the planted "bad actors".
emp = []
first = ["Rina", "Budi", "Sari", "Andi", "Dewi", "Fajar", "Maya", "Rizky", "Putri", "Agus", "Lina", "Dimas", "Nadia", "Eko", "Tika", "Yoga", "Wulan", "Hendra"]
n = 0
for _, st in stores.iterrows():
    emp.append((f"M{st.store_id[2:]}", f"Manajer {st.store_name.split(' - ')[0]}", st.store_id, "Manajer"))
    for _ in range(3):
        emp.append((f"K{n + 1:02d}", first[n], st.store_id, "Kasir")); n += 1
employees = pd.DataFrame(emp, columns=["employee_id", "nama", "store_id", "peran"])
cashiers = employees[employees.peran == "Kasir"]
mgr = employees[employees.peran == "Manajer"].set_index("store_id").employee_id.to_dict()

# ---- planted behaviour (cashier -> scheme). Rates are per order unless stated.
BAD = {
    "K02": {"refund_abuse": 0.050, "self_approve": 0.60, "label": "Refund berlebihan dan disetujui sendiri"},
    "K05": {"discount_abuse": 0.30, "just_below": True, "label": "Diskon berlebihan, banyak tepat di bawah batas persetujuan"},
    "K08": {"cash_short": True, "label": "Kas selalu kurang saat shift sore"},
    "K11": {"offhours": 0.045, "void": 0.07, "label": "Transaksi di luar jam buka dan void tinggi"},
    "K14": {"dup": 0.03, "label": "Struk ganda berturutan"},
    "K17": {"refund_fictive": 0.020, "label": "Refund tanpa order asli"},
}
DISC_LIMIT = 20          # discount % above this needs manager approval
PAY = ["cash", "debit", "qris", "kartu_kredit"]
PAY_P = [0.28, 0.27, 0.35, 0.10]

orders, items, labels = [], [], []
oid = 0
base_orders = 70
dow_mult = np.array([0.85, 0.88, 0.92, 1.0, 1.15, 1.35, 1.40])
for d in dates:
    for _, st in stores.iterrows():
        cs = cashiers[cashiers.store_id == st.store_id].employee_id.tolist()
        n_ord = int(rng.poisson(base_orders * st.scale * dow_mult[d.dayofweek]))
        last_by_cashier = {}
        for _ in range(n_ord):
            oid += 1
            c = cs[rng.integers(0, 3)]
            shift = "pagi" if rng.random() < 0.5 else "sore"
            hour = int(rng.integers(st.jam_buka, 16)) if shift == "pagi" else int(rng.integers(16, st.jam_tutup))
            minute, sec = int(rng.integers(0, 60)), int(rng.integers(0, 60))
            bad = BAD.get(c, {})
            off = False
            if bad.get("offhours") and rng.random() < bad["offhours"]:
                hour, off = int(rng.choice([6, 7, 23, 0])), True
            k = int(rng.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.14, 0.06]))
            chosen = rng.choice(len(skus), size=k, replace=False, p=pop)
            disc_pct = 0.0
            if rng.random() < 0.18:
                disc_pct = float(rng.choice([5, 10, 15], p=[0.5, 0.35, 0.15]))
            approved = ""
            reason = ""
            if bad.get("discount_abuse") and rng.random() < bad["discount_abuse"]:
                disc_pct = float(rng.choice([DISC_LIMIT - 1, DISC_LIMIT, 25, 30], p=[0.45, 0.25, 0.2, 0.1])) if bad.get("just_below") else 25.0
                reason = "diskon_abuse"
            if disc_pct > DISC_LIMIT:
                approved = mgr[st.store_id] if rng.random() < 0.75 else ""
            voided = bool(bad.get("void") and rng.random() < bad["void"]) or rng.random() < 0.006
            sub = 0
            lines = []
            for ix in chosen:
                q = int(rng.choice([1, 2, 3], p=[0.82, 0.14, 0.04]))
                p_ = int(skus.price[ix])
                lines.append((oid, skus.sku[ix], q, p_))
                sub += q * p_
            disc = round(sub * disc_pct / 100)
            total = sub - disc
            pay = str(rng.choice(PAY, p=PAY_P))
            ts = d + pd.Timedelta(hours=hour, minutes=minute, seconds=sec)
            orders.append((f"ORD{oid:07d}", ts, st.store_id, c, shift, pay, sub, disc, total, int(voided), approved))
            items.extend(lines)
            if reason:
                labels.append((f"ORD{oid:07d}", "order", c, "diskon_berlebihan"))
            if off:
                labels.append((f"ORD{oid:07d}", "order", c, "luar_jam_buka"))
            if bad.get("dup") and rng.random() < bad["dup"]:
                oid += 1
                ts2 = ts + pd.Timedelta(seconds=int(rng.integers(20, 150)))
                orders.append((f"ORD{oid:07d}", ts2, st.store_id, c, shift, pay, sub, disc, total, 0, approved))
                items.extend([(oid, a[1], a[2], a[3]) for a in lines])
                labels.append((f"ORD{oid:07d}", "order", c, "struk_ganda"))
            last_by_cashier[c] = oid

ord_df = pd.DataFrame(orders, columns=["order_id", "datetime", "store_id", "cashier_id", "shift", "payment_method", "subtotal", "discount", "total", "voided", "discount_approved_by"])
item_df = pd.DataFrame(items, columns=["_o", "sku", "qty", "unit_price"])
item_df.insert(0, "order_id", item_df._o.map(lambda x: f"ORD{x:07d}"))
item_df = item_df.drop(columns="_o")

# ---- refunds: ~3% of orders; some planted
RSN = ["changed_mind", "wrong_shade", "damaged_leaked", "allergic_reaction", "wrong_product"]
good = ord_df[(ord_df.voided == 0) & (ord_df.discount_approved_by == "")].sample(frac=0.03, random_state=7)
refunds = []
rid = 0
items_by_order = item_df.groupby("order_id")
def add_refund(o_id, ts, store, cashier, approver, sku, qty, price, reason, resell, order_ref=True):
    global rid
    rid += 1
    refunds.append((f"RFD{rid:06d}", o_id, ts, store, cashier, approver, sku, qty, qty * price, reason, resell))
    return f"RFD{rid:06d}"
for r in good.itertuples():
    it = items_by_order.get_group(r.order_id).iloc[0]
    ts = r.datetime + pd.Timedelta(days=int(rng.integers(0, 8)), hours=int(rng.integers(0, 4)))
    if ts > END + pd.Timedelta(days=1):
        continue
    cas = r.cashier_id
    appr = mgr[r.store_id] if rng.random() < 0.9 else ""
    add_refund(r.order_id, ts, r.store_id, cas, appr, it.sku, 1, it.unit_price, str(rng.choice(RSN, p=[0.4, 0.2, 0.1, 0.1, 0.2])), int(rng.random() < 0.4))
# planted: refund abuse (self-approved, high volume) for K02; fictive refunds for K17
for cid, spec in BAD.items():
    st = cashiers.set_index("employee_id").store_id[cid]
    pool = ord_df[(ord_df.cashier_id == cid) & (ord_df.voided == 0)]
    if "refund_abuse" in spec:
        for r in pool.sample(int(len(pool) * spec["refund_abuse"]), random_state=1).itertuples():
            it = items_by_order.get_group(r.order_id).iloc[0]
            self_ap = rng.random() < spec["self_approve"]
            rf = add_refund(r.order_id, r.datetime + pd.Timedelta(hours=int(rng.integers(1, 40))), st, cid, cid if self_ap else mgr[st],
                            it.sku, int(it.qty), it.unit_price, "changed_mind", 0)
            labels.append((rf, "refund", cid, "refund_berlebihan"))
    if "refund_fictive" in spec:
        for _ in range(int(len(pool) * spec["refund_fictive"])):
            s_ = skus.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
            d_ = dates[int(rng.integers(5, DAYS))] + pd.Timedelta(hours=int(rng.integers(11, 20)))
            rf = add_refund(f"ORD9{int(rng.integers(100000, 999999))}", d_, st, cid, mgr[st], s_.sku, 1, int(s_.price), "wrong_product", 0)
            labels.append((rf, "refund", cid, "refund_tanpa_order"))
refund_df = pd.DataFrame(refunds, columns=["refund_id", "order_id", "datetime", "store_id", "cashier_id", "approved_by", "sku", "qty", "amount", "reason", "resellable"])

# ---- order integrity errors (a few): total != subtotal - discount
bad_ix = ord_df.sample(14, random_state=3).index
ord_df.loc[bad_ix, "total"] = ord_df.loc[bad_ix, "total"] - 25000
for o in ord_df.loc[bad_ix, "order_id"]:
    labels.append((o, "order", ord_df.set_index("order_id").cashier_id[o], "total_tidak_cocok"))

# ---- cash closing per store / day / shift / cashier (cash sales vs counted)
cash = ord_df[(ord_df.payment_method == "cash") & (ord_df.voided == 0)].copy()
cash["date"] = cash.datetime.dt.date
kas = cash.groupby(["date", "store_id", "shift", "cashier_id"]).total.sum().reset_index().rename(columns={"total": "expected_cash"})
def counted(r):
    diff = int(rng.choice([0, 0, 0, -500, 500, -1000, 1000]))
    if r.cashier_id == "K08" and r["shift"] == "sore" and rng.random() < 0.7:
        diff = -int(rng.integers(40, 160)) * 1000
    return r.expected_cash + diff
kas["actual_cash"] = kas.apply(counted, axis=1)
for r in kas.itertuples():
    if r.cashier_id == "K08" and r.actual_cash - r.expected_cash <= -40000:
        labels.append((f"KAS-{r.date}-{r.store_id}-{r.shift}", "kas", "K08", "selisih_kas"))

# ---- stock count (opname): system vs physical, a few shrinkage hotspots
op = []
for _, st in stores.iterrows():
    for _, s in skus.iterrows():
        sys_q = int(rng.integers(20, 90))
        loss = int(rng.binomial(sys_q, 0.01))
        if st.store_id == "ST05" and s.sku in ("MU003", "SK008", "MU009"):
            loss = int(sys_q * rng.uniform(0.25, 0.40)); labels.append((f"OPN-{st.store_id}-{s.sku}", "opname", "", "susut_stok"))
        op.append((END.date(), st.store_id, s.sku, sys_q, sys_q - loss))
opname = pd.DataFrame(op, columns=["date", "store_id", "sku", "system_qty", "counted_qty"])

# ---- write
def write(path, sheets, guide):
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss", date_format="yyyy-mm-dd") as xw:
        pd.DataFrame(guide, columns=["Sheet", "Kolom", "Arti"]).to_excel(xw, sheet_name="Petunjuk", index=False)
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k, index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font, c.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F2A44")
            for col in ws.columns:
                w = max(len(str(x.value)) if x.value is not None else 0 for x in list(col)[:150])
                ws.column_dimensions[col[0].column_letter].width = min(max(w + 2, 10), 70 if ws.title == "Petunjuk" else 28)
            if ws.title == "Petunjuk":
                for row in ws.iter_rows(min_row=2):
                    for c in row:
                        c.alignment = Alignment(wrap_text=True, vertical="top")

G1 = [("Order", "order_id, datetime, store_id, cashier_id", "Satu baris per struk. datetime = tanggal dan jam transaksi"),
      ("Order", "shift, payment_method", "pagi / sore; cash, debit, qris, kartu_kredit"),
      ("Order", "subtotal, discount, total", "Total harus = subtotal - discount"),
      ("Order", "voided, discount_approved_by", "voided 1 bila dibatalkan; ID atasan yang menyetujui diskon (kosong bila tidak ada)"),
      ("Item", "order_id, sku, qty, unit_price", "Satu baris per produk dalam struk"),
      ("Refund", "refund_id, order_id, datetime, store_id, cashier_id", "Refund dan order asalnya"),
      ("Refund", "approved_by, sku, qty, amount, reason, resellable", "Penyetuju refund, produk, nilai, alasan, bisa dijual lagi (1/0)"),
      ("Setoran_Kas", "date, store_id, shift, cashier_id, expected_cash, actual_cash", "Penjualan tunai menurut sistem dan uang yang disetor")]
G2 = [("Karyawan", "employee_id, nama, store_id, peran", "Kasir atau Manajer"),
      ("Toko", "store_id, store_name, jam_buka, jam_tutup", "Jam operasional (jam, 24 jam)"),
      ("SKU", "sku, name, category, price, unit_cost", "Master produk"),
      ("Opname", "date, store_id, sku, system_qty, counted_qty", "Stok menurut sistem dan hasil hitung fisik")]
write(OUT / "Data_Order_Lengkap.xlsx", {"Order": ord_df, "Item": item_df, "Refund": refund_df, "Setoran_Kas": kas}, G1)
write(OUT / "Data_Master.xlsx", {"Karyawan": employees, "Toko": stores.drop(columns="scale"), "SKU": skus.drop(columns="popularity"), "Opname": opname}, G2)
pd.DataFrame(labels, columns=["ref", "tipe", "kasir", "skema"]).to_csv(OUT / "Label_Anomali.csv", index=False)
print(f"orders={len(ord_df):,} items={len(item_df):,} refunds={len(refund_df):,} kas={len(kas):,} opname={len(opname)} labels={len(labels)}")
print({c: v["label"] for c, v in BAD.items()})
