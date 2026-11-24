"""Excel audit working paper: summary, findings, cashier risk, and the evidence behind every finding."""
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

from .agent import STEPS

F = "Arial"
HEAD = Font(name=F, size=10, bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="1F2A44")
NORM, BOLD = Font(name=F, size=10), Font(name=F, size=10, bold=True)
thin = Side(style="thin", color="D9D9D9")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")
SEV = {"Tinggi": "F4B6B6", "Sedang": "FFE3A3", "Rendah": "DCE6F1"}


def _head(ws, row, labels, widths=None):
    for i, t in enumerate(labels, 1):
        c = ws.cell(row=row, column=i, value=t)
        c.font, c.fill, c.border, c.alignment = HEAD, FILL, BORDER, Alignment(wrap_text=True, vertical="center")
        if widths:
            ws.column_dimensions[L(i)].width = widths[i - 1]


def _put(ws, r, c, v, fmt=None, wrap=False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.font, cell.border = NORM, BORDER
    if fmt:
        cell.number_format = fmt
    if wrap:
        cell.alignment = WRAP
    return cell


def build_workbook(res: dict, path):
    fs, risk, meta = res["findings"], res["risk"], res["meta"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Ringkasan"
    title = "Kertas Kerja Audit Data Order" if res.get("mode") == "order" else "Kertas Kerja Audit Kasir"
    ws["A1"], ws["A1"].font = title, Font(name=F, size=14, bold=True)
    ws["A2"], ws["A2"].font = f"Periode {meta['date_from']} sampai {meta['date_to']}  |  {meta['n_orders']:,} order  |  {meta['n_stores']} toko  |  {meta['n_cashiers']} kasir".replace(",", "."), Font(name=F, size=10, italic=True)
    cnt = {s: sum(1 for f in fs if f["severity"] == s) for s in ("Tinggi", "Sedang", "Rendah")}
    rows = [("Temuan tingkat Tinggi", cnt["Tinggi"]), ("Temuan tingkat Sedang", cnt["Sedang"]), ("Temuan tingkat Rendah", cnt["Rendah"]),
            ("Nilai yang perlu ditelusuri (Rp)", round(sum(f["amount"] for f in fs))), ("Kasir berisiko Tinggi", int((risk.level == "Tinggi").sum()))]
    for i, (k, v) in enumerate(rows, 4):
        _put(ws, i, 1, k).font = BOLD
        _put(ws, i, 2, v, "#,##0")
    ws["A10"], ws["A10"].font = "Catatan: temuan berarti perlu diperiksa, bukan bukti kecurangan. Verifikasi dengan dokumen dan bukti fisik sebelum menyimpulkan.", Font(name=F, size=10, italic=True)
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 38, 18

    wf = wb.create_sheet("Temuan")
    _head(wf, 1, ["ID", "Tingkat", "Skor", "Subjek", "Toko", "Judul", "Penjelasan", "Nilai (Rp)", "Jumlah bukti", "Langkah verifikasi"], [7, 10, 7, 22, 24, 48, 70, 15, 10, 70])
    for i, f in enumerate(fs, 2):
        vals = [f["id"], f["severity"], f["score"], f["subject_name"], f["store"], f["title"], f["detail"], round(f["amount"]), f["evidence_count"], " | ".join(STEPS.get(f["rule"], []))]
        for j, v in enumerate(vals, 1):
            _put(wf, i, j, v, "#,##0" if j == 8 else None, wrap=j in (6, 7, 10))
    for s, color in SEV.items():
        wf.conditional_formatting.add(f"B2:B{len(fs) + 1}", CellIsRule(operator="equal", formula=[f'"{s}"'], fill=PatternFill("solid", bgColor=color, fgColor=color)))
    wf.freeze_panes = "A2"
    wf.auto_filter.ref = f"A1:J{len(fs) + 1}"

    wr = wb.create_sheet("Risiko Kasir")
    _head(wr, 1, ["ID Kasir", "Nama", "Toko", "Order", "Jumlah temuan", "Skor risiko", "Tingkat", "Nilai (Rp)", "Aturan terpicu"], [10, 16, 26, 10, 12, 11, 11, 15, 70])
    for i, r in enumerate(risk.itertuples(index=False), 2):
        for j, v in enumerate([r.cashier_id, r.nama, r.store, r.orders, r.findings, r.risk, r.level, round(r.amount), r.rules], 1):
            _put(wr, i, j, v, "#,##0" if j in (4, 8) else None)
    wr.freeze_panes = "A2"

    we = wb.create_sheet("Bukti")
    row = 1
    for f in fs:
        c = we.cell(row=row, column=1, value=f"{f['id']}  {f['title']}  ({f['subject_name']})")
        c.font = BOLD
        row += 1
        cols = f["evidence_cols"]
        _head(we, row, cols)
        row += 1
        for ev in f["evidence"][:200]:
            for j, k in enumerate(cols, 1):
                _put(we, row, j, ev.get(k))
            row += 1
        if f["evidence_count"] > 200:
            we.cell(row=row, column=1, value=f"... dan {f['evidence_count'] - 200} baris lain").font = Font(name=F, size=10, italic=True)
            row += 1
        row += 1
    for i in range(1, 12):
        we.column_dimensions[L(i)].width = 20
    wb.save(path)
