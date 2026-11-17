"""
Small helper layer over python-docx that reproduces the IEEE conference
template (A4, Times New Roman, 2 columns, small-caps headings, three-rule
tables, native Word equations via LaTeX -> MathML -> OMML).
"""
from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Mm
from lxml import etree

MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
_XSLT = etree.XSLT(etree.parse(str(MML2OMML))) if MML2OMML.exists() else None

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
COL_W_IN = 3.45


def _set_cols(section, n, space_in=0.24):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(space_in * 1440)))


class IEEEDoc:
    def __init__(self):
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Mm(210), Mm(297)
        s.top_margin, s.bottom_margin = Mm(19), Mm(25)
        s.left_margin = s.right_margin = Mm(15.9)
        _set_cols(s, 1)
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.font.size = Pt(10)
        pf = st.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        self.sec_no = 0
        self.sub_no = 0
        self.eq_no = 0
        self.fig_no = 0
        self.tab_no = 0

    # ---------------------------------------------------------------- basics
    def _run(self, p, text, bold=False, italic=False, size=None, smallcaps=False, sup=False):
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if smallcaps:
            r.font.small_caps = True
        if sup:
            r.font.superscript = True
        return r

    def rich(self, p, text, size=None, base_italic=False, base_bold=False):
        """Minimal markup: **bold**, *italic*, ^{sup}, _{sub}."""
        import re
        tokens = re.split(r"(\*\*.+?\*\*|\*.+?\*|\^\{.+?\}|_\{.+?\})", text)
        for t in tokens:
            if not t:
                continue
            if t.startswith("**"):
                self._run(p, t[2:-2], bold=True, italic=base_italic, size=size)
            elif t.startswith("*"):
                self._run(p, t[1:-1], italic=not base_italic, bold=base_bold, size=size)
            elif t.startswith("^{"):
                self._run(p, t[2:-1], size=size, sup=True, italic=base_italic, bold=base_bold)
            elif t.startswith("_{"):
                r = self._run(p, t[2:-1], size=size, italic=base_italic, bold=base_bold)
                r.font.subscript = True
            else:
                self._run(p, t, size=size, italic=base_italic, bold=base_bold)
        return p

    # ------------------------------------------------------------ title block
    def title(self, text):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)
        self._run(p, text, size=24)

    def authors(self, authors):
        """authors: list of lists of lines; first line is the name."""
        t = self.doc.add_table(rows=1, cols=len(authors))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, lines in zip(t.rows[0].cells, authors):
            cell.paragraphs[0].text = ""
            for i, line in enumerate(lines):
                p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                italic = line.startswith("Dept.") or line.startswith("Dr. ")
                self._run(p, line, size=10 if i == 0 else 9, italic=italic)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(6)
        # switch to two columns for the body
        sec = self.doc.add_section(WD_SECTION.CONTINUOUS)
        _set_cols(sec, 2)

    def abstract(self, text, keywords):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(0.14)
        p.paragraph_format.space_after = Pt(6)
        self._run(p, "Abstract", bold=True, italic=True, size=9)
        self._run(p, "\u2014", bold=True, size=9)
        self.rich(p, text, size=9, base_bold=True)
        k = self.doc.add_paragraph()
        k.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        k.paragraph_format.first_line_indent = Inches(0.14)
        k.paragraph_format.space_after = Pt(6)
        self._run(k, "Keywords\u2014", bold=True, italic=True, size=9)
        self._run(k, keywords, bold=True, italic=True, size=9)

    # -------------------------------------------------------------- headings
    def section(self, text, numbered=True):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        if numbered:
            self.sec_no += 1
            self.sub_no = 0
            self._run(p, f"{ROMAN[self.sec_no - 1]}. ")
        self._run(p, text, smallcaps=True)

    def subsection(self, text):
        self.sub_no += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        self._run(p, f"{chr(64 + self.sub_no)}. {text}", italic=True)

    # -------------------------------------------------------------- body
    def para(self, text, indent=True, size=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=0):
        p = self.doc.add_paragraph()
        p.alignment = align
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.14)
        p.paragraph_format.space_after = Pt(after)
        self.rich(p, text, size=size)
        return p

    def bullet(self, label, text):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(0.14)
        self._run(p, label, italic=True)
        self.rich(p, " " + text)
        return p

    # -------------------------------------------------------------- equations
    def _omml(self, latex):
        import latex2mathml.converter
        mml = latex2mathml.converter.convert(latex)
        omml = _XSLT(etree.fromstring(mml.encode()))
        return parse_xml(etree.tostring(omml.getroot()))

    def equation(self, latex):
        self.eq_no += 1
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Inches(COL_W_IN / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Inches(COL_W_IN), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        p._p.append(self._omml(latex))
        p.add_run(f"\t({self.eq_no})")
        return self.eq_no

    def inline_math(self, p, latex):
        p._p.append(self._omml(latex))

    # -------------------------------------------------------------- figures
    def figure(self, path, caption, width_in=COL_W_IN):
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(width_in))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.paragraph_format.space_before = Pt(3)
        c.paragraph_format.space_after = Pt(8)
        self._run(c, f"Fig. {self.fig_no}.  ", size=8)
        self.rich(c, caption, size=8)
        return self.fig_no

    # -------------------------------------------------------------- tables
    def table(self, caption, header, rows, col_widths=None, note=None, bold_rows=(), align=None):
        self.tab_no += 1
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(6)
        cp.paragraph_format.keep_with_next = True
        self._run(cp, f"TABLE {ROMAN[self.tab_no - 1]}.", size=8)
        cp2 = self.doc.add_paragraph()
        cp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp2.paragraph_format.space_after = Pt(3)
        cp2.paragraph_format.keep_with_next = True
        self._run(cp2, caption.upper(), size=8, smallcaps=False)
        for r in cp2.runs:
            r.font.size = Pt(7)

        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        if not col_widths:
            first = 1.3
            rest = (3.35 - first) / (len(header) - 1)
            col_widths = [first] + [rest] * (len(header) - 1)
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(col_widths) * 1440)))
        tw.set(qn("w:type"), "dxa")
        old = tblPr.find(qn("w:tblW"))
        if old is not None:
            tblPr.remove(old)
        tblPr.append(tw)
        for j, w in enumerate(col_widths):
            t.columns[j].width = Inches(w)
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.paragraphs[0].text = ""
                p = cell.paragraphs[0]
                a = align[j] if align else ("L" if j == 0 else "R")
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if a == "L" else WD_ALIGN_PARAGRAPH.RIGHT
                self.rich(p, str(val), size=8, base_bold=(i == 0 or (i - 1) in bold_rows))
                if col_widths:
                    cell.width = Inches(col_widths[j])
                # cell margins
                tcPr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for side in ("top", "bottom"):
                    e = OxmlElement(f"w:{side}")
                    e.set(qn("w:w"), "10"); e.set(qn("w:type"), "dxa")
                    mar.append(e)
                tcPr.append(mar)
        # three-rule style: top, below header, bottom
        def border(cell, side, sz):
            tcPr = cell._tc.get_or_add_tcPr()
            b = tcPr.find(qn("w:tcBorders"))
            if b is None:
                b = OxmlElement("w:tcBorders"); tcPr.append(b)
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
            b.append(e)
        for cell in t.rows[0].cells:
            border(cell, "top", 8); border(cell, "bottom", 4)
        for cell in t.rows[-1].cells:
            border(cell, "bottom", 8)
        if note:
            n = self.doc.add_paragraph()
            n.paragraph_format.space_after = Pt(6)
            self.rich(n, note, size=7)
        else:
            self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # -------------------------------------------------------------- refs
    def references(self, refs):
        self.section("References", numbered=False)
        for i, r in enumerate(refs, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
            self._run(p, f"[{i}]\t", size=8)
            self.rich(p, r, size=8)

    def save(self, path):
        self.doc.save(path)
