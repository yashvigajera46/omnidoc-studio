import io
import os
import re
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from google import genai
from pydantic import BaseModel

# ReportLab modules for professional PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas

load_dotenv()

app = FastAPI(title="OmniDoc AI Studio")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    extracted = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted += text + "\n"
    return extracted.strip()

# Custom canvas to draw clean page numbers without browser URL/date
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Bottom page number
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 40, 25, page_text)
        
        # Subtle bottom footer brand
        self.drawString(40, 25, "OmniDoc Studio • Generated Study Guide")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(40, 38, A4[0] - 40, 38)
        self.restoreState()

@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    provider: str = Form("gemini-flash"),
    task: str = Form("summary"),
    tone: str = Form("friendly"),
    detail_level: str = Form("balanced"),
    language: str = Form("English"),
    question: str = Form(None)
):
    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=400, content={"detail": "Uploaded file is empty."})

        doc_text = extract_text_from_pdf(content)
        if not doc_text:
            return JSONResponse(status_code=400, content={"detail": "Could not extract readable text from PDF."})

        context_slice = doc_text[:14000]

        tones_map = {
            "friendly": "Explain in warm, everyday, friendly language. Avoid overly dense academic jargon.",
            "eli10": "Explain like I am 10 years old with relatable, simple analogies.",
            "quick": "Use short, crisp bullet points only. Highlight high-yield points.",
            "exam": "Focus on high-probability exam questions, important formulas, and clear answers."
        }

        detail_map = {
            "quick": "Keep the response brief and high-level.",
            "balanced": "Provide a well-structured, clear explanation.",
            "detailed": "Provide an in-depth, step-by-step breakdown."
        }

        prompts = {
            "summary": f"Provide a simple, clear summary of this document followed by 5 quick bullet takeaways:\n\n{context_slice}",
            "cheat_sheet": f"Create an easy-to-read 1-page Cheat Sheet with core concepts, formulas, pinouts, and rules neatly formatted in Markdown tables:\n\n{context_slice}",
            "mcq": f"Generate 5 Multiple-Choice Questions (MCQs) based on this text. Provide 4 options (A, B, C, D) for each, followed by an Answer Key with brief, friendly explanations:\n\n{context_slice}",
            "flashcards": f"Create 5 easy study flashcards (Question: ... and Answer: ...) covering the main points:\n\n{context_slice}",
            "glossary": f"Extract important words and technical terms into a clean Markdown table with easy-to-understand definitions:\n\n{context_slice}",
            "exam": f"Generate 5 realistic exam questions based on this document with clear, complete model answers:\n\n{context_slice}",
            "eli10": f"Explain everything in this document using simple, everyday analogies so anyone can understand it easily:\n\n{context_slice}",
            "qa": f"Answer the following question clearly and simply using ONLY the document text below.\nQuestion: {question}\n\nContext:\n{context_slice}"
        }

        selected_prompt = prompts.get(task, prompts["summary"])
        system_instruction = (
            f"You are a friendly and clear tutor. {tones_map.get(tone, '')} {detail_map.get(detail_level, '')} "
            f"Respond in {language}. Format output using clean Markdown tables, bold headers, and bullet points."
        )

        full_prompt = f"{system_instruction}\n\n{selected_prompt}"

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=400, content={"detail": "Missing GEMINI_API_KEY in your .env file."})

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=full_prompt
        )
        return JSONResponse(content={"result": response.text})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server Error: {str(e)}"})

# Request model for PDF generation
class PDFExportRequest(BaseModel):
    markdown_text: str

@app.post("/api/export-pdf")
async def export_pdf(data: PDFExportRequest):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=45
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            'DocSub',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=14
        )
        h1_style = ParagraphStyle(
            'H1',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor('#334155'),
            spaceAfter=6
        )
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4
        )
        cell_style = ParagraphStyle(
            'Cell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#1e293b')
        )
        cell_bold = ParagraphStyle(
            'CellBold',
            parent=cell_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#0f172a')
        )

        story = []

        # Header Title Banner
        story.append(Paragraph("OmniDoc Studio • Study Notes", title_style))
        story.append(Paragraph("Structured Study Guide & Quick Reference", subtitle_style))

        lines = data.markdown_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Markdown Tables Detection
            if line.startswith('|') and line.endswith('|'):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i].strip())
                    i += 1

                # Filter out markdown separator line (|---|---|)
                cleaned_rows = [r for r in table_lines if not re.match(r'^\|[\s\-:|]+\|$', r)]
                table_data = []
                for row_idx, r in enumerate(cleaned_rows):
                    cols = [c.strip() for c in r.strip('|').split('|')]
                    formatted_cols = []
                    for c in cols:
                        c_clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c)
                        if row_idx == 0:
                            formatted_cols.append(Paragraph(c_clean, cell_bold))
                        else:
                            formatted_cols.append(Paragraph(c_clean, cell_style))
                    table_data.append(formatted_cols)

                if table_data:
                    num_cols = max(len(row) for row in table_data)
                    avail_width = A4[0] - 80
                    col_width = avail_width / num_cols if num_cols > 0 else avail_width
                    
                    t = Table(table_data, colWidths=[col_width] * num_cols)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 8))
                continue

            # Headings
            if line.startswith('#'):
                clean_h = line.lstrip('#').strip()
                story.append(Paragraph(clean_h, h1_style))
            # Bullets
            elif line.startswith(('-', '*', '•')):
                clean_b = line.lstrip('-*• ').strip()
                clean_b = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_b)
                story.append(Paragraph(f"• {clean_b}", bullet_style))
            # Regular text
            else:
                clean_t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                story.append(Paragraph(clean_t, body_style))

            i += 1

        doc.build(story, canvasmaker=NumberedCanvas)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=OmniDoc_Study_Notes.pdf"}
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"PDF Generation Error: {str(e)}"})