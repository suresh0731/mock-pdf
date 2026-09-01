"""Build a synthetic digital PDF replicating the second user-reported
Eastspring payment-instruction letter (test-input/1000099900.jpg) as
closely as possible: same addressee (Standard Chartered Bank / Menara
Standard Chartered Bank Lt 5), same six debit-table rows/account numbers,
same 2-line Account Name wraps.

This is the file the user says was actually uploaded to the server as a
*digital* PDF (the JPG is only a visual reference of the same content) —
the server-side run on it produced badly scrambled/merged redactions for
rows 1 and 2 (a genuine two-line wrap where the row's account-number/
amounts line is the wrap's *second* line, not its first).

Usage (from repo root):
    python scripts/_make_repro_eastspring_v2_pdf.py
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
text(LEFT, 130, "To", size=9)
text(LEFT, 148, "Standard Chartered Bank", size=9)
text(LEFT, 160, "Menara Standard Chartered Bank Lt 5", size=9)
text(LEFT, 172, "Jl. Prof. Dr. Satrio No. 164", size=9)
text(LEFT, 184, "Jakarta", size=9)
text(LEFT, 196, "Attn : Lea Deciana", size=9)
text(LEFT, 208, "Telp : 021-25551642", size=9)
text(LEFT, 220, "Fax : 021 - 2555 0002", size=9)

text(LEFT, 245, "Date :", size=9)
text(200, 245, "10 December 2025", size=9)
text(LEFT, 267, "Subject : Payment Redemption/Switching Fee Maybank - November 2025", size=9)

text(LEFT, 289, "Please transfer on", size=9)
text(200, 289, "11 December 2025", size=9)
text(310, 289, "through SKN (full amount), with detail as follows :", size=9)

text(LEFT, 312, "Debit From :", size=9)

# --- Table -----------------------------------------------------------
TABLE_TOP = 327
col_w = [200, 75, 78, 75, 84]
col_x = [LEFT]
for w in col_w:
    col_x.append(col_x[-1] + w)
table_right = col_x[-1]

HEADERS = ["Account Name", "Account Number", "Redemption/\nSwitching Fee", "VAT 11 %", "Net Amount (IDR)"]

# Each Account Name cell wraps onto exactly 2 lines, with the account
# number/amounts row visually aligned with the wrap's *second* line (not
# the first) -- this is the exact geometry from the real scanned source
# (test-input/1000099900.jpg): line 1 is name-only, line 2 carries both
# the name's tail *and* the account number/fees on the same baseline.
rows_data = [
    (["Reksa Dana Saham Eastspring", "Investments Alpha Navigator"],
     ["306-0866797-5"], ["IDR 114,497,818"], ["IDR 12,594,760"], ["IDR 127,092,578.07"]),
    (["Reksa Dana Eastspring Investments", "Yield Discovery"],
     ["306-0889854-3"], ["IDR 6,796,432"], ["IDR 747,607.52"], ["IDR 7,544,039.53"]),
    (["Reksa Dana Eastspring Investments IDR", "High Grade"],
     ["306-0879870-0"], ["IDR 166,722,554"], ["IDR 18,339,480.92"], ["IDR 185,062,034.78"]),
    (["REKSA DANA INDEKS EASTSPRING IDX", "ESG LEADERS PLUS"],
     ["306-8155729-7"], ["IDR 3,738,106"], ["IDR 411,191.68"], ["IDR 4,149,297.89"]),
    (["REKSA DANA SYARIAH EASTSPRING", "SYARIAH GREATER CHINA EQUITY USD"],
     ["0104923105 (USD)"], ["IDR 1,597,749"], ["IDR 175,752.35"], ["IDR 1,773,501.00"]),
    (["REKSA DANA SYARIAH EASTSPRING", "SYARIAH FIXED INCOME USD"],
     ["306-8147753-6 (USD", "Cash Account)"], ["IDR 21,063,808"], ["IDR 2,317,018.89"], ["IDR 23,380,827.00"]),
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
    for line in lines:
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
text(180, credit_top + 15, "Fee Agent Penjual", size=9)
text(LEFT, credit_top + 27, "Account Number:", size=9)
text(180, credit_top + 27, "200.301.4553", size=9)
text(LEFT, credit_top + 39, "Bank :", size=9)
text(180, credit_top + 39, "Maybank", size=9)

closing_top = credit_top + 65
text(
    LEFT,
    closing_top,
    "Should the payment is done before date 9th, please to include in current tax period, otherwise please proceed on next tax period.",
    size=9,
)
text(LEFT, closing_top + 20, "Thank you for your assistance in this matter.", size=9)
text(LEFT, closing_top + 40, "PT. Eastspring Investments Indonesia", size=9)

out_path = "repro_eastspring_v2_table.pdf"
doc.save(out_path)
print("wrote", out_path)
