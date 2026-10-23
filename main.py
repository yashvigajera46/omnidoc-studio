import os
import time
import io
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI(title="Omnidoc Studio")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Omnidoc Studio is Live</h1>")


def call_gemini_with_retry(client, prompt: str, max_retries: int = 4, base_delay: float = 2.0):
    delay = base_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            return response
        except Exception as e:
            last_exception = e
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                time.sleep(delay)
                delay *= 2
                continue
            raise e
    raise last_exception


@app.post("/api/analyze")
@app.post("/generate-notes")
@app.post("/generate")
async def analyze_content(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    provider: str = Form("gemini-flash"),
    task: str = Form("summary"),
    tone: str = Form("friendly"),
    detail_level: str = Form("balanced"),
    language: str = Form("English"),
    question: Optional[str] = Form(None)
):
    try:
        content_text = ""
        if file and file.filename:
            file_bytes = await file.read()
            filename_lower = file.filename.lower()
            if filename_lower.endswith(".txt") or filename_lower.endswith(".md"):
                content_text = file_bytes.decode("utf-8", errors="ignore")
            elif filename_lower.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    content_text = "".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    content_text = file_bytes.decode("latin-1", errors="ignore")
            else:
                content_text = file_bytes.decode("utf-8", errors="ignore")

        if raw_text and raw_text.strip():
            content_text += ("\n\n" if content_text else "") + raw_text.strip()

        if not content_text.strip():
            return JSONResponse(status_code=400, content={"detail": "Please provide either uploaded text or file content."})

        context_slice = content_text[:15000]

        prompts = {
            "summary": f"Provide comprehensive, well-structured study notes in {language} with a {tone} tone and {detail_level} depth:\n\n{context_slice}",
            "flashcards": f"Create 5 easy study flashcards (Question: ... and Answer: ...) covering the main points in {language}:\n\n{context_slice}",
            "quiz": f"Create a 5-question multiple choice study quiz with answer key in {language}:\n\n{context_slice}",
            "qa": f"Answer this question based on the document text: {question}\n\nDocument:\n{context_slice}"
        }

        full_prompt = prompts.get(task, prompts["summary"])

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=400, content={"detail": "Missing GEMINI_API_KEY in your environment."})

        client = genai.Client(api_key=api_key.strip())
        api_response = call_gemini_with_retry(client, full_prompt)

        # Return both keys to support whatever key your frontend JS expects
        return {
            "notes": api_response.text,
            "result": api_response.text
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Server Error: {str(e)}"})


class PDFExportRequest(BaseModel):
    content: str


@app.post("/api/export-pdf")
@app.post("/export-pdf")
async def export_pdf(data: PDFExportRequest):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        y = 750
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, "Omnidoc Studio - Study Notes")
        y -= 30
        p.setFont("Helvetica", 10)

        for line in data.content.split("\n"):
            if y < 50:
                p.showPage()
                p.setFont("Helvetica", 10)
                y = 750
            p.drawString(50, y, line[:95])
            y -= 15

        p.save()
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=study_notes.pdf"}
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"PDF Generation Error: {str(e)}"})