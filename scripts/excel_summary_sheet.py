"""
Builds the "Summary" sheet for the Excel report.

Layout (two-level header):

                          |      All       |    Shift A     |    Shift B     |    Shift C
 App        | Measure     | Total | OK|NOK | Total | OK|NOK | Total | OK|NOK | Total | OK|NOK
 -----------+-------------+-------------------------------------------------------------------
 app_1      | Results     |   .   | . | .  |  ...
            | VINs        |   .   | . | .  |  ...

How to use it (plain steps):
  1. Put your numbers in a dictionary like SAMPLE_DATA below.
  2. Call add_summary_sheet(workbook, data).
  3. Save the workbook.

Needs:  pip install openpyxl
Try it: python scripts/excel_summary_sheet.py   (writes summary_demo.xlsx)
"""

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# The groups shown on top, in order. Change the names here if needed.
GROUPS = ["All", "Shift A", "Shift B", "Shift C"]
# The three columns under every group.
SUB_COLUMNS = ["Total", "OK", "NOK"]
# The two rows under every app.
MEASURES = ["Results", "VINs"]

# data[app][measure][group] = (total, ok, nok)
SAMPLE_DATA = {
    "atoc_final_radiator": {
        "Results": {"All": (10, 8, 2), "Shift A": (4, 3, 1), "Shift B": (3, 3, 0), "Shift C": (3, 2, 1)},
        "VINs": {"All": (9, 8, 1), "Shift A": (4, 4, 0), "Shift B": (3, 2, 1), "Shift C": (2, 2, 0)},
    },
    "atoc_hose_star": {
        "Results": {"All": (0, 0, 0), "Shift A": (0, 0, 0), "Shift B": (0, 0, 0), "Shift C": (0, 0, 0)},
        "VINs": {"All": (0, 0, 0), "Shift A": (0, 0, 0), "Shift B": (0, 0, 0), "Shift C": (0, 0, 0)},
    },
    "new_brake_hose": {
        "Results": {"All": (5, 4, 1), "Shift A": (2, 2, 0), "Shift B": (2, 1, 1), "Shift C": (1, 1, 0)},
        "VINs": {"All": (5, 5, 0), "Shift A": (2, 2, 0), "Shift B": (2, 2, 0), "Shift C": (1, 1, 0)},
    },
}

# Styles
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
HEADER_FONT = Font(bold=True, color="FFFFFF")
BOLD = Font(bold=True)
GREEN = Font(color="2E7D32")
RED = Font(color="C62828")
GREY = Font(color="9E9E9E")
CENTER = Alignment(horizontal="center", vertical="center")
THIN = Side(style="thin", color="BFBFBF")
THICK = Side(style="medium", color="000000")


def add_summary_sheet(workbook, data, sheet_name="Summary"):
    """Create the Summary sheet in `workbook` from `data` and return it."""
    ws = workbook.create_sheet(sheet_name)

    # ---- Header row 1 and 2 ----
    ws.cell(row=1, column=1, value="App")
    ws.cell(row=1, column=2, value="Measure")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=2)

    col = 3  # first number column
    for group in GROUPS:
        # group name spans 3 columns
        ws.cell(row=1, column=col, value=group)
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + 2)
        # Total / OK / NOK under it
        for i, name in enumerate(SUB_COLUMNS):
            ws.cell(row=2, column=col + i, value=name)
        col += 3
    last_col = col - 1

    for r in (1, 2):
        for c in range(1, last_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.alignment = CENTER
            cell.fill = HEADER_FILL if r == 1 else SUBHEADER_FILL
            cell.font = HEADER_FONT if r == 1 else BOLD

    # ---- Data rows: each app takes 2 rows (Results, VINs) ----
    row = 3
    for app, measures in data.items():
        app_first_row = row
        for measure in MEASURES:
            ws.cell(row=row, column=2, value=measure).alignment = CENTER
            values = measures.get(measure, {})
            col = 3
            for group in GROUPS:
                total, ok, nok = values.get(group, (0, 0, 0))
                for i, number in enumerate((total, ok, nok)):
                    cell = ws.cell(row=row, column=col + i, value=number)
                    cell.alignment = CENTER
                    if number == 0:
                        cell.font = GREY
                    elif i == 1:      # OK column
                        cell.font = GREEN
                    elif i == 2:      # NOK column
                        cell.font = RED
                col += 3
            row += 1

        # App name once, merged over its rows
        ws.cell(row=app_first_row, column=1, value=app)
        ws.merge_cells(start_row=app_first_row, start_column=1, end_row=row - 1, end_column=1)
        ws.cell(row=app_first_row, column=1).alignment = CENTER
        ws.cell(row=app_first_row, column=1).font = BOLD

    last_row = row - 1

    # ---- Borders: thin everywhere, thick line before each group and under each app ----
    group_start_cols = [3 + 3 * i for i in range(len(GROUPS))]
    for r in range(1, last_row + 1):
        for c in range(1, last_col + 1):
            left = THICK if c in group_start_cols or c == 3 else THIN
            # thick line under the last row of each app
            bottom = THICK if (r >= 3 and (r - 3) % len(MEASURES) == len(MEASURES) - 1) else THIN
            ws.cell(row=r, column=c).border = Border(left=left, right=THIN, top=THIN, bottom=bottom)

    # ---- Widths, frozen panes ----
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 12
    for c in range(3, last_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 8
    ws.freeze_panes = "C3"   # keep header + App/Measure visible when scrolling

    return ws


if __name__ == "__main__":
    wb = Workbook()
    wb.remove(wb.active)  # drop the default empty sheet
    add_summary_sheet(wb, SAMPLE_DATA)
    wb.save("summary_demo.xlsx")
    print("Saved summary_demo.xlsx")
