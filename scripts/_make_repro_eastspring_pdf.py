"""Build a minimal synthetic digital PDF replicating the user-reported
Eastspring Investments payment-instruction letter (test-input/1000099855.jpg):
a business letter whose debit table has a wide "Account Name" column that
wraps a fund's full name onto 2-3 lines in several rows, immediately next
to a narrow "Account Number" column that never wraps.

Row 2 ("...IDR HIGH GRADE"), row 5 ("...ASIA PACIFIC USD") and row 6
("...MIXED ASSET FUND") are the three rows whose Account Name text wraps
onto more than one line -- these are the ones the production run
(test-output/1000099857.jpg) mis-redacted: the fund-name mock value landed
mid-cell, breaking the wrapped name into unredacted fragments, while the
real account number one column over was never touched at all.

Usage (from repo root):
    python scripts/_make_repro_eastspring_pdf.py
"""
import fitz

doc = fitz.open()
page = doc.new_page(width=612, height=792)  # US Letter, points

FONT = "helv"
LEFT = 50


def text(x, y, s, size=9, font=FONT):
    page.insert_text((x, y), s, fontsize=size, fontname=font)


# --- Letterhead ---------------------------------------------------------
text(LEFT, 55, "eastspring investments", size=13)
text(430, 50, "PT. Eastspring Investments Indonesia", size=8)
text(430, 61, "Prudential Tower, 23th Floor", size=8)
text(430, 72, "Jl. Jend. Sudirman Kav. 79, 12910", size=8)
text(430, 83, "+ 62 21 2924 5515 / 44", size=8)
text(430, 94, "+ 62 21 2924 5566", size=8)

# --- Addressee -----------------------------------------------------------
text(LEFT, 130, "To:", size=9)
text(LEFT, 143, "STANDARD CHARTERED BANK, JAKARTA BRANCH", size=9)
text(LEFT, 155, "3rd floor, Word Trade Center II,", size=9)
text(LEFT, 167, "Jl.Jend. Sudirman Kav.29-31,", size=9)
text(LEFT, 179, "DKI Jakarta 12930, Indonesia", size=9)

text(LEFT, 202, "Attn.:", size=9)
text(200, 202, "Syarifudin / Ambarwati Susilarini", size=9)
text(LEFT, 220, "Date :", size=9)
text(200, 220, "8 December 2025", size=9)
text(LEFT, 240, "Subject : Payment Transaction Switching Fee Bareksa -November 2025", size=9)

text(LEFT, 262, "Please transfer on", size=9)
text(200, 262, "9 December 2025", size=9)
text(310, 262, "through SKN (full amount), with detail as follows :", size=9)

text(LEFT, 285, "Debit From :", size=9)

# --- Table -----------------------------------------------------------
TABLE_TOP = 300
col_w = [200, 75, 78, 75, 84]
col_x = [LEFT]
for w in col_w:
    col_x.append(col_x[-1] + w)
table_right = col_x[-1]

HEADERS = ["Account Name", "Account Number", "Redemption/\nSwitching Fee", "VAT 11%", "Net Amount (IDR)"]

rows_data = [
    (["Reksa Dana Eastspring Investments Value", "Discovery"],
     ["306-0886641-2"], ["IDR 11,092.48"], ["IDR 1,220.17"], ["IDR 12,312.65"]),
    (["Reksa Dana Eastspring Investments IDR", "High Grade"],
     ["306-0879870-0"], ["IDR 336,543.07"], ["IDR 37,019.74"], ["IDR 373,562.81"]),
    (["REKSA DANA EASTSPRING IDR FIXED", "INCOME FUND"],
     ["306-0986525-8"], ["IDR 39,788,693.23"], ["IDR 4,376,756.26"], ["IDR 44,165,449.49"]),
    (["REKSA DANA INDEKS EASTSPRING IDX ESG", "LEADERS PLUS"],
     ["306-8155729-7"], ["IDR 2,159,395.89"], ["IDR 237,533.55"], ["IDR 2,396,929.44"]),
    (["REKSA DANA SYARIAH EASTSPRING", "SYARIAH EQUITY ISLAMIC ASIA PACIFIC", "USD"],
     ["0107436477 (USD)"], ["IDR 1,487,027.93"], ["IDR 163,573.07"], ["IDR 1,650,601.00"]),
    (["REKSA DANA SYARIAH EASTSPRING", "SYARIAH MIXED ASSET FUND"],
     ["306-8186638-9"], ["IDR 306,211.76"], ["IDR 33,683.29"], ["IDR 339,895.05"]),
]

LINE_H = 11
ROW_PAD = 6
row_heights = [22]
for cells in rows_data:
    n_lines = max(len(c) for c in cells)
    row_heights.append(n_lines * LINE_H + ROW_PAD)

row_y = [TABLE_TOP]
for h in row_heights:
    row_y.append(row_y[-1] + h)
table_bottom = row_y[-1]

for x in col_x:
    page.draw_line((x, TABLE_TOP), (x, table_bottom), width=0.75)
for y in row_y:
    page.draw_line((LEFT, y), (table_right, y), width=0.75)


def put(col, row, lines, fontsize=8.5):
    x = col_x[col] + 5
    y_top = row_y[row] + 13
    for i, line in enumerate(lines):
        for sub in line.split("\n"):
            page.insert_text((x, y_top), sub, fontsize=fontsize, fontname=FONT)
            y_top += LINE_H


for i, h in enumerate(HEADERS):
    put(i, 0, [h], fontsize=8.5)

for r, cells in enumerate(rows_data, start=1):
    for c, lines in enumerate(cells):
        put(c, r, lines)

# --- Credit section -------------------------------------------------------
credit_top = table_bottom + 25
text(LEFT, credit_top, "Credit to :", size=9)
text(LEFT, credit_top + 15, "Account Name :", size=9)
text(180, credit_top + 15, "PT. BAREKSA PORTAL INVESTASI", size=9)
text(LEFT, credit_top + 27, "Account Number:", size=9)
text(180, credit_top + 27, "105-030-9980", size=9)
text(LEFT, credit_top + 39, "Bank :", size=9)
text(180, credit_top + 39, "BCA KCP Jembatan Lima", size=9)
text(LEFT, credit_top + 51, "Swift Code:", size=9)
text(180, credit_top + 51, "CENAIDJA", size=9)

closing_top = credit_top + 80
text(LEFT, closing_top, "Thank you for your assistance in this matter.", size=9)
text(LEFT, closing_top + 20, "PT. Eastspring Investments Indonesia", size=9)

out_path = "repro_eastspring_table.pdf"
doc.save(out_path)
print("wrote", out_path)
