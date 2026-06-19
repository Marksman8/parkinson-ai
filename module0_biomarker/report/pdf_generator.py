"""
PDF Report Generator
Creates a downloadable clinical PDF report for the neurologist.
Uses reportlab for PDF generation.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_pdf_report(
    session_id: str,
    doctor_input: str,
    top_genes: List[Dict],
    ode_params: List[Dict],
    graph_path: Optional[str],
    save_path: str = "data/outputs/biomarker_report.pdf",
) -> str:
    """
    Generate a clean clinical PDF report.
    Returns path to saved PDF.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        return _generate_reportlab(
            session_id, doctor_input, top_genes,
            ode_params, graph_path, save_path,
        )
    except ImportError:
        logger.warning("reportlab not installed — generating text report instead.")
        return _generate_text_report(
            session_id, doctor_input, top_genes,
            ode_params, save_path.replace(".pdf", ".txt"),
        )


def _generate_reportlab(
    session_id, doctor_input, top_genes,
    ode_params, graph_path, save_path,
) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, Image,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=18, leading=22, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#3a3a5e"),
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12, leading=16, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#1e4f7a"),
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#2a2a2a"),
    )
    gene_title_style = ParagraphStyle(
        "GeneTitle",
        parent=styles["Normal"],
        fontSize=11, leading=14, spaceBefore=8, spaceAfter=2,
        textColor=colors.HexColor("#1e4f7a"),
        fontName="Helvetica-Bold",
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8.5, leading=12, spaceAfter=2,
        textColor=colors.HexColor("#5a5a5a"),
    )

    # ── Header ───────────────────────────────────────────────────────
    story.append(Paragraph(
        "Parkinson's Disease Biomarker Discovery Report",
        title_style,
    ))
    story.append(Paragraph(
        f"Automated KEGG–STRING–Centrality Analysis  ·  {datetime.now().strftime('%B %d, %Y')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#2d6fa3"), spaceAfter=12))

    # ── Session info ─────────────────────────────────────────────────
    story.append(Paragraph("Session Information", section_style))
    info_data = [
        ["Session ID",    session_id],
        ["Generated",     datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Clinical Input", doctor_input[:300] + ("..." if len(doctor_input) > 300 else "")],
    ]
    info_table = Table(info_data, colWidths=[4*cm, 13*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2fb")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#c0d8ec")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("PADDING",    (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f5f9fd"), colors.white]),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Top genes table ───────────────────────────────────────────────
    story.append(Paragraph("Top Hub Genes — Degree Centrality Analysis", section_style))
    story.append(Paragraph(
        "The following genes were identified as the most connected hub nodes in the "
        "Parkinson's disease protein–protein interaction network (STRING database, "
        "confidence ≥ 0.70), ranked by composite hub score (degree + betweenness + closeness centrality).",
        body_style,
    ))
    story.append(Spacer(1, 0.3*cm))

    gene_headers = [["Rank", "Gene", "Degree", "C_D", "C_B", "Hub Score"]]
    gene_rows = []
    for g in top_genes:
        gene_rows.append([
            str(g.get("rank", "—")),
            g["gene"],
            str(g.get("degree", "—")),
            f"{g.get('degree_c', 0):.4f}",
            f"{g.get('betweenness_c', 0):.4f}",
            f"{g.get('hub_score', 0):.4f}",
        ])
    gene_table = Table(
        gene_headers + gene_rows,
        colWidths=[1.5*cm, 3*cm, 2*cm, 2.8*cm, 2.8*cm, 3*cm],
    )
    gene_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2d6fa3")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#c0d8ec")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f9fd"), colors.white]),
        ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
        ("PADDING",      (0, 0), (-1, -1), 6),
    ]))
    story.append(gene_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Gene explanations ────────────────────────────────────────────
    story.append(Paragraph("Clinical Gene Summaries", section_style))
    from .gene_info import GENE_EXPLANATIONS
    for g in top_genes:
        gene = g["gene"]
        info = GENE_EXPLANATIONS.get(gene, {})
        story.append(Paragraph(
            f"{gene}  —  {info.get('full_name', gene)}",
            gene_title_style,
        ))
        if info:
            story.append(Paragraph(
                f"<b>Role:</b> {info.get('role', '')}",
                body_style,
            ))
            story.append(Paragraph(
                f"<b>In Parkinson's Disease:</b> {info.get('in_pd', '')}",
                body_style,
            ))
            story.append(Paragraph(
                f"<b>Clinical Note:</b> {info.get('clinical_tip', '')}",
                body_style,
            ))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#d0e0ec"), spaceAfter=4,
        ))

    # ── Network graph ────────────────────────────────────────────────
    if graph_path and Path(graph_path).exists():
        story.append(Paragraph("PPI Network Visualization", section_style))
        story.append(Paragraph(
            "Hub genes (coloured, large nodes) are highlighted in the "
            "Parkinson's disease protein–protein interaction network. "
            "Edge weight reflects STRING interaction confidence.",
            body_style,
        ))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(
            graph_path,
            width=16*cm, height=12*cm,
            kind="proportional",
        ))
        story.append(Spacer(1, 0.4*cm))

    # ── ODE parameters ───────────────────────────────────────────────
    if ode_params:
        story.append(Paragraph(
            "ODE Parameters Ready for Simulation (Module 1 Calibration)",
            section_style,
        ))
        story.append(Paragraph(
            "The following parameters are calibrated from peer-reviewed literature "
            "and are loaded automatically into the Module 2 simulation engine.",
            body_style,
        ))
        story.append(Spacer(1, 0.2*cm))

        param_headers = [["Gene", "Parameter", "Value", "Unit", "Source"]]
        param_rows = [
            [
                p.get("gene", ""),
                p.get("name", ""),
                str(p.get("value", "")),
                p.get("unit", ""),
                p.get("source", ""),
            ]
            for p in ode_params[:25]
        ]
        param_table = Table(
            param_headers + param_rows,
            colWidths=[2.5*cm, 3*cm, 2.5*cm, 2.5*cm, 6.5*cm],
        )
        param_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#c07a18")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ALIGN",        (2, 1), (3, -1), "CENTER"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#d4993a")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#fffdf5"), colors.white]),
            ("PADDING",      (0, 0), (-1, -1), 5),
        ]))
        story.append(param_table)

    # ── Footer ───────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#2d6fa3"), spaceAfter=6,
    ))
    story.append(Paragraph(
        "⚠ This report is generated by an AI research tool. "
        "All findings should be verified by a qualified neurologist or clinical geneticist "
        "before any clinical decision is made.",
        small_style,
    ))
    story.append(Paragraph(
        f"PD Biomarker Discovery Tool  ·  Session {session_id}  ·  "
        f"github.com/Marksman8/parkinsons-ai",
        small_style,
    ))

    doc.build(story)
    logger.info(f"PDF report saved to {save_path}")
    return save_path


def _generate_text_report(
    session_id, doctor_input, top_genes, ode_params, save_path,
) -> str:
    """Fallback plain-text report when reportlab is not available."""
    lines = [
        "=" * 70,
        "PARKINSON'S DISEASE BIOMARKER DISCOVERY REPORT",
        f"Session: {session_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "CLINICAL INPUT:",
        doctor_input[:500],
        "",
        "TOP HUB GENES:",
    ]
    for g in top_genes:
        lines.append(
            f"  {g.get('rank','?')}. {g['gene']}  "
            f"degree_C={g.get('degree_c',0):.4f}  "
            f"hub_score={g.get('hub_score',0):.4f}"
        )
    if ode_params:
        lines += ["", "ODE PARAMETERS:"]
        for p in ode_params[:20]:
            lines.append(
                f"  {p['gene']} · {p['name']} = {p['value']} {p['unit']}  "
                f"({p['source']})"
            )
    lines += [
        "",
        "DISCLAIMER: AI research tool only. Verify with qualified clinician.",
        "=" * 70,
    ]
    Path(save_path).write_text("\n".join(lines))
    logger.info(f"Text report saved to {save_path}")
    return save_path
