"""Build a minimal synthetic digital PDF replicating the user-reported
9-column debit/credit remittance table: a debit-side ``Bank`` column that
wraps "Standard Chartered" / "Bank" onto two lines in every row, alongside
a wider credit-side ``Bank`` column that wraps a full address block onto
several lines in every row too -- to see whether the two same-role
"bank_name" columns wrapping in lockstep on the same physical line
interferes with whether the narrow column's second line survives.

Row 3 additionally uses a single-line, ALL-CAPS, en-dash account name
(mirroring the user's real document) instead of the two-line
Title-Case/hyphen-suffix name rows 1-2 use.
"""
import fitz

doc = fitz.open()
page = doc.new_page(width=1400, height=500)

FONT = "helv"
FS = 8
LEFT = 30
TOP = 60

# name | number | bank(narrow) | amount | name2 | number2 | bank2(wide, address) | remarks | value date
col_w = [190, 110, 90, 90, 190, 110, 160, 60, 80]
col_x = [LEFT]
for w in col_w:
    col_x.append(col_x[-1] + w)

row_h = [22, 60, 60, 60]  # header, row1, row2, row3 (tall enough for the 5-line address cell)
row_y = [TOP]
for h in row_h:
    row_y.append(row_y[-1] + h)

table_right = col_x[-1]
table_bottom = row_y[-1]

for x in col_x:
    page.draw_line((x, TOP), (x, table_bottom), width=0.75)
for y in row_y:
    page.draw_line((LEFT, y), (table_right, y), width=0.75)


def put(col, row, lines, fontsize=FS):
    x = col_x[col] + 5
    y_top = row_y[row] + 12
    for i, line in enumerate(lines):
        page.insert_text((x, y_top + i * 11), line, fontsize=fontsize, fontname=FONT)


HEADERS = [
    "Account Name", "Account Number", "Bank", "Amount",
    "Account Name", "Account Number", "Bank", "Remarks", "Value Date",
]
for i, h in enumerate(HEADERS):
    put(i, 0, [h])

ADDRESS_BLOCK = [
    "Standard Chartered Bank",
    "Singapore",
    "Address : 7 Changi Business",
    "Park Crescent Level 3",
    "Singapore 486028",
]

rows_data = [
    (
        ["PT Prudential Life Assurance", "- RGEM"],
        ["306-8110482-9 (IDR)"],
        ["Standard Chartered", "Bank"],
        ["IDR 75,045,626.61"],
        ["PT Prudential Life Assurance -", "RGEM/DGEM - MULTI"],
        ["0106236334 (USD)"],
        ADDRESS_BLOCK,
        ["BEN"],
        ["7-Jan-2026"],
    ),
    (
        ["PT Prudential Life Assurance", "- RGLV"],
        ["306-8110488-8 (IDR)"],
        ["Standard Chartered", "Bank"],
        ["IDR 880,918,682.87"],
        ["PT Prudential Life Assurance -", "RGLV/DGLV - MULTI"],
        ["0106235931 (USD)"],
        ADDRESS_BLOCK,
        ["BEN"],
        ["7-Jan-2026"],
    ),
    (
        ["PT PRUDENTIAL LIFE ASSURANCE \u2013 PDTI"],
        ["306-8185122-5 (USD)"],
        ["Standard Chartered", "Bank"],
        ["USD 29,480.70"],
        ["PT Prudential Life Assurance -", "PDGT - MULTI"],
        ["0105781592 (USD)"],
        ADDRESS_BLOCK,
        ["BEN"],
        ["7-Jan-2026"],
    ),
]

for r, cells in enumerate(rows_data, start=1):
    for c, lines in enumerate(cells):
        put(c, r, lines)

out_path = "repro_table.pdf"
doc.save(out_path)
print("wrote", out_path)
