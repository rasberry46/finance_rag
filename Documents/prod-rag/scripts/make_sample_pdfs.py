"""
make_sample_pdfs.py
===================
Generates realistic financial sample PDFs so the pipeline has real documents to
ingest (tables + prose). Run once: python -m scripts.make_sample_pdfs
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.lib.styles import getSampleStyleSheet

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
OUT.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()


def _table(data):
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    return t


def acme_10k():
    doc = SimpleDocTemplate(str(OUT / "acme_10k_fy2024.pdf"), pagesize=letter)
    s = []
    s.append(Paragraph("ACME Cloud Inc. — FY2024 Annual Report (Excerpt)", styles["Title"]))
    s.append(Spacer(1, 12))
    s.append(Paragraph("Item 7. Segment Revenue (USD millions)", styles["Heading2"]))
    s.append(_table([
        ["Segment", "Q1", "Q2", "Q3", "Q4", "FY Total"],
        ["Small Business", "1200", "1350", "1410", "1600", "5560"],
        ["Mid Market", "800", "820", "910", "975", "3505"],
        ["Enterprise", "2100", "2050", "2200", "2400", "8750"],
        ["Total", "4100", "4220", "4520", "4975", "17815"],
    ]))
    s.append(Spacer(1, 12))
    s.append(Paragraph(
        "Small Business revenue grew steadily through FY2024, driven by higher "
        "subscription attach rates and net revenue retention exceeding 110 percent "
        "in the second half. Enterprise revenue dipped in Q2 due to deal slippage "
        "before recovering strongly in the second half of the year.", styles["BodyText"]))
    s.append(Spacer(1, 12))
    s.append(Paragraph("Item 8. Deferred Revenue and Revenue Recognition", styles["Heading2"]))
    s.append(Paragraph(
        "Deferred revenue represents customer payments received before the related "
        "performance obligations are satisfied. The Company recognizes revenue under "
        "ASC 606 as control of the promised services transfers to the customer, "
        "typically ratably over the subscription term. Deferred revenue at fiscal "
        "year end was 2,340 million, up from 1,980 million in the prior year.",
        styles["BodyText"]))
    s.append(Spacer(1, 12))
    s.append(Paragraph("Item 9. Operating Expenses vs Budget (USD millions)", styles["Heading2"]))
    s.append(_table([
        ["Line Item", "Budget", "Actual", "Variance"],
        ["Sales & Marketing", "3200", "3560", "360"],
        ["R&D", "2800", "2710", "-90"],
        ["G&A", "1100", "1180", "80"],
    ]))
    doc.build(s)


def saas_metrics_memo():
    doc = SimpleDocTemplate(str(OUT / "saas_metrics_memo_q4.pdf"), pagesize=letter)
    s = []
    s.append(Paragraph("Internal Memo: Q4 SaaS Metrics Review", styles["Title"]))
    s.append(Spacer(1, 12))
    s.append(Paragraph(
        "This memo summarizes key SaaS metrics for Q4. Annual Recurring Revenue "
        "(ARR) reached 17.8 billion on an annualized basis. Monthly Recurring "
        "Revenue (MRR) grew 4 percent month over month in December. Net Revenue "
        "Retention (NRR) was 112 percent, and Gross Revenue Retention was 94 percent.",
        styles["BodyText"]))
    s.append(Spacer(1, 12))
    s.append(Paragraph("Key SaaS Metrics", styles["Heading2"]))
    s.append(_table([
        ["Metric", "Q3", "Q4", "Target"],
        ["ARR (USD B)", "16.9", "17.8", "17.5"],
        ["NRR (%)", "109", "112", "110"],
        ["GRR (%)", "93", "94", "92"],
        ["CAC Payback (months)", "14", "13", "12"],
        ["Magic Number", "0.9", "1.1", "1.0"],
    ]))
    s.append(Spacer(1, 12))
    s.append(Paragraph(
        "Commission accruals are booked monthly based on closed-won bookings and "
        "are amortized over the expected customer life in accordance with ASC 340-40. "
        "The Magic Number improving above 1.0 indicates efficient go-to-market spend.",
        styles["BodyText"]))
    doc.build(s)


if __name__ == "__main__":
    acme_10k()
    saas_metrics_memo()
    for p in sorted(OUT.glob("*.pdf")):
        print("wrote", p.name, p.stat().st_size, "bytes")
