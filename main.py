from fastapi import FastAPI, HTTPException, File, UploadFile
from app_workflow import run_workflow
import os
import tempfile
import docx  # Add this to handle .docx files

app = FastAPI()

@app.post("/run-workflow/")
async def run_workflow_endpoint(srs_document: UploadFile = File(...), project_name: str = "default_project"):
    """
    Run the StateGraph workflow for Milestones 1 and 2.
    """
    try:
        # Sanitize the project name
        sanitized_project_name = project_name.replace(" ", "_").replace("-", "_")

        # Validate file format
        if not srs_document.filename.endswith(".docx"):
            raise HTTPException(status_code=400, detail="Only .docx files are supported.")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(await srs_document.read())
            temp_file_path = temp_file.name

        # Extract text from the .docx file
        doc = docx.Document(temp_file_path)
        srs_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        # Call the workflow function with the extracted text and sanitized project name
        result = await run_workflow(srs_text=srs_text, project_name=sanitized_project_name)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return {"message": "Workflow executed successfully.", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))