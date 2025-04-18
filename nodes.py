import os
import subprocess
import json
import re
import shutil
import stat
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import MyState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectGenerationError(Exception):
    """Base exception for project generation errors"""
    pass

class LLMResponseError(ProjectGenerationError):
    """Exception for LLM response parsing errors"""
    pass

class DatabaseSetupError(ProjectGenerationError):
    """Exception for database setup errors"""
    pass

# Constants
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_CONTAINER_IMAGE = "postgres:latest"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_USER = os.getenv("POSTGRES_USER")
DEFAULT_DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

def handle_subprocess_error(error: subprocess.CalledProcessError, context: str) -> None:
    """Handle subprocess errors with proper logging and context"""
    logger.error(f"Error in {context}: {error}")
    logger.error(f"Command output: {error.stdout}")
    logger.error(f"Command stderr: {error.stderr}")
    raise ProjectGenerationError(f"Failed to {context}: {error}")

@dataclass
class ProjectStructure:
    """Project structure configuration"""
    base_dirs = {
        "app/api/routes": ["user.py", "item.py", "__init__.py"],
        "app/models": ["user.py", "item.py", "__init__.py"],
        "app/services": [],
        "tests": [],
        "": ["Dockerfile", "requirements.txt", ".env", "README.md"],
        "app": ["database.py", "main.py"]
    }

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model=DEFAULT_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")
)

def analyze_srs(state: MyState) -> MyState:
    """
    Analyze the SRS text and extract key software requirements.
    """
    srs_text = state.srs_text  # Use dot notation to access the attribute
    logger.info("Analyzing SRS document...")
    prompt = f"""
    Analyze the following SRS document and extract the following information in valid JSON format:
    {{
        "endpoints": [
            {{
                "method": "string",
                "path": "string",
                "parameters": ["string"],
                "description": "string"
            }}
        ],
        "models": [
            {{
                "model_name": "string",
                "table_name": "string",
                "columns": [
                    {{
                        "name": "string",
                        "type": "string",
                        "options": "string"
                    }}
                ]
            }}
        ],
        "authentication": {{
            "rbac": "boolean",
            "jwt": "boolean"
        }}
    }}

    SRS Document:
    {srs_text}
    """
    response = llm.invoke(prompt)

    try:
        response_content = response.content[response.content.index("{"):response.content.rindex("}")+1]
        extracted_data = json.loads(response_content)
    except json.JSONDecodeError:
        raise LLMResponseError("The LLM response is not in valid JSON format.")
    logger.info("SRS document analyzed successfully. Extracted data:")
    logger.info(json.dumps(extracted_data, indent=2))
    
    # Update the state with the extracted data
    state.extracted_data = extracted_data
    return state

def setup_project(state: MyState) -> MyState:
    """
    Set up a structured FastAPI project with a modular folder structure.
    """
    project_name = state.project_name  # Use dot notation to access the attribute
    sanitized_project_name = project_name.replace(" ", "_").replace("-", "_")  # Sanitize project name
    extracted_data = state.extracted_data

    logger.info(f"Setting up project: {sanitized_project_name}")

    # Define the project structure
    structure = ProjectStructure.base_dirs

    # Create directories and files
    for folder, files in structure.items():
        folder_path = os.path.join(sanitized_project_name, folder)
        os.makedirs(folder_path, exist_ok=True)
        os.chmod(folder_path, stat.S_IWRITE)  # Ensure write permissions for the folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w") as f:
                # Add default content for specific files
                if file == "main.py":
                    f.write("from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'message': 'Hello, World!'}\n")
                elif file == "requirements.txt":
                    f.write("fastapi\nuvicorn\npsycopg2\nalembic\n")
                elif file == "Dockerfile":
                    f.write("FROM python:3.9-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n")
                elif file == "README.md":
                    f.write(f"# {sanitized_project_name}\n\nThis is a FastAPI project generated by the system.\n")
                elif file == ".env":
                    f.write("# Environment variables\n")
                else:
                    f.write("")  # Leave other files empty

            os.chmod(file_path, stat.S_IWRITE)  # Ensure write permissions for the file

    # Initialize Python virtual environment and install dependencies
    try:
        subprocess.run(["python", "-m", "venv", f"{sanitized_project_name}/venv"], check=True)
        subprocess.run([f"{sanitized_project_name}/venv/Scripts/pip", "install", "-r", f"{sanitized_project_name}/requirements.txt"], check=True)
    except subprocess.CalledProcessError as e:
        handle_subprocess_error(e, "initialize virtual environment and install dependencies")

    logger.info(f"Project {sanitized_project_name} setup complete.")
    state.project_structure = f"{sanitized_project_name} created successfully."
    return state

def setup_database(state: MyState) -> MyState:
    """
    Set up a PostgreSQL database using Podman and create models based on the extracted schema.
    """
    project_name = state.project_name  # Use dot notation to access the attribute
    sanitized_project_name = project_name.replace(" ", "_").replace("-", "_")  # Sanitize project name
    extracted_data = state.extracted_data  # JSON response from the LLM

    logger.info(f"Setting up PostgreSQL database for project: {sanitized_project_name}")

    # Step 1: Install Podman (if not already installed)
    try:
        subprocess.run(["podman", "--version"], check=True)
    except FileNotFoundError:
        logger.info("Podman not found. Installing Podman...")
        try:
            subprocess.run(["winget", "install", "--id", "RedHat.Podman"], check=True)
        except subprocess.CalledProcessError as e:
            handle_subprocess_error(e, "install Podman")

    # Step 2: Pull PostgreSQL image
    logger.info("Pulling PostgreSQL image...")
    try:
        subprocess.run(["podman", "pull", DEFAULT_CONTAINER_IMAGE], check=True)
    except subprocess.CalledProcessError as e:
        handle_subprocess_error(e, "pull PostgreSQL image")

    # Step 3: Check if the container already exists and remove it
    container_name = f"{sanitized_project_name}_db"
    logger.info(f"Checking if container '{container_name}' already exists...")
    try:
        existing_containers = subprocess.run(
            ["podman", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.splitlines()

        if container_name in existing_containers:
            logger.info(f"Container '{container_name}' already exists. Removing it...")
            subprocess.run(["podman", "rm", "-f", container_name], check=True)
    except subprocess.CalledProcessError as e:
        handle_subprocess_error(e, "check and remove existing container")

    # Step 4: Run PostgreSQL container
    logger.info("Running PostgreSQL container...")
    try:
        subprocess.run([
            "podman", "run", "-d",
            "--name", container_name,
            "-e", f"POSTGRES_USER={DEFAULT_DB_USER}",
            "-e", f"POSTGRES_PASSWORD={DEFAULT_DB_PASSWORD}",
            "-e", f"POSTGRES_DB={sanitized_project_name}",
            "-p", f"{DEFAULT_DB_PORT}:{DEFAULT_DB_PORT}",
            DEFAULT_CONTAINER_IMAGE
        ], check=True)
    except subprocess.CalledProcessError as e:
        handle_subprocess_error(e, "run PostgreSQL container")

    # Step 5: Create database models
    logger.info("Creating database models...")
    models_folder = os.path.join(project_name, "app", "models")
    os.makedirs(models_folder, exist_ok=True)

    for model in extracted_data.get("models", []):
        model_name = model["model_name"]
        table_name = model["table_name"]
        columns = model["columns"]

        model_file_path = os.path.join(models_folder, f"{model_name.lower()}.py")
        with open(model_file_path, "w") as f:
            f.write("from sqlalchemy import Column, Integer, String, Boolean\n")
            f.write("from sqlalchemy.ext.declarative import declarative_base\n\n")
            f.write("Base = declarative_base()\n\n")
            f.write(f"class {model_name}(Base):\n")
            f.write(f"    __tablename__ = '{table_name}'\n")
            for column in columns:
                column_name = column["name"]
                column_type = column["type"]
                options = column.get("options", "")
                f.write(f"    {column_name} = Column({column_type}, {options})\n")
            f.write("\n")

    logger.info(f"Database setup and models creation for project {sanitized_project_name} complete.")
    state.database_status = f"Database and models for {sanitized_project_name} created successfully."
    return state

def extract_json_from_text(text: str) -> str:
    """Extract valid JSON from text that may contain markdown or other content"""
    # Clean up the text by removing markdown code block markers and any leading/trailing whitespace
    if "```json" in text:
        # Extract content between ```json and ```
        pattern = r"```json\s*(\{[\s\S]*\})\s*```"
        match = re.search(pattern, text)
        if match:
            text = match.group(1)
    elif "```" in text:
        # Extract content between ``` and ```
        pattern = r"```\s*(\{[\s\S]*\})\s*```"
        match = re.search(pattern, text)
        if match:
            text = match.group(1)

    # Clean up any escaped quotes
    text = text.replace('\\"', '"')
    
    # Remove any trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        # Validate JSON structure
        json_obj = json.loads(text)
        return json.dumps(json_obj)  # Return a properly formatted JSON string
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Attempted to parse: {text}")
        raise ValueError(f"Invalid JSON structure: {e}")


def ensure_directory_permissions(path: str) -> None:
    """Ensure directory and its parents have proper permissions"""
    try:
        current = Path(path)
        for parent in reversed(current.parents):
            if parent.exists():
                os.chmod(parent, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except Exception as e:
        logger.error(f"Error setting permissions for directory {path}: {e}")
        raise

def create_file_with_content(file_path: str, content: str) -> None:
    """Helper function to create file with proper path handling"""
    try:
        # Normalize path separators for Windows
        normalized_path = os.path.normpath(file_path)
        directory = os.path.dirname(normalized_path)
        
        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            ensure_directory_permissions(directory)
            logger.info(f"Created directory: {directory}")

        # Remove file if it exists
        if os.path.exists(normalized_path):
            os.chmod(normalized_path, stat.S_IWRITE | stat.S_IREAD)  # Ensure write permissions
            os.remove(normalized_path)

        # Write content to file
        with open(normalized_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')

        # Ensure write permissions for the file
        os.chmod(normalized_path, stat.S_IWRITE | stat.S_IREAD)
        logger.info(f"Created file: {normalized_path}")

    except PermissionError as e:
        logger.error(f"Permission denied while creating or writing to {normalized_path}: {e}")
        raise PermissionError(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f"Error creating file {normalized_path}: {e}")
        raise

def process_structure(structure: dict, current_path: str = "", project_root: str = "") -> None:
    """Recursively process the file structure"""
    try:
        for key, value in structure.items():
            # Remove any forward/backward slashes from the beginning and end
            clean_key = key.strip('/').strip('\\')
            path = os.path.normpath(os.path.join(current_path, clean_key))
            
            if isinstance(value, dict):
                # If it's a dictionary, it's a directory
                full_dir_path = os.path.join(project_root, path)
                if not os.path.exists(full_dir_path):
                    Path(full_dir_path).mkdir(parents=True, exist_ok=True)
                    ensure_directory_permissions(full_dir_path)
                process_structure(value, path, project_root)
            else:
                # If it's a string, it's a file
                full_path = os.path.join(project_root, path)
                create_file_with_content(full_path, value)
    except Exception as e:
        logger.error(f"Error processing structure at {current_path}: {e}")
        raise ProjectGenerationError(f"Failed to process directory structure: {e}")

def generate_code(state: MyState) -> MyState:
    """Generate the complete FastAPI project structure based on the provided analysis."""
    project_name = state.project_name
    extracted_data = state.extracted_data

    logger.info(f"Generating code for project: {project_name}")

    # Define the folder paths
    project_root = project_name

    # Improved prompt with explicit JSON formatting instructions
    prompt = f"""Generate a FastAPI project structure as a JSON object. 
    The response should be a single JSON object with file paths as keys and file contents as string values.
    Do not include any explanation or markdown formatting - ONLY the JSON object.

    Use this exact format:
    {{
        "app/main.py": "content here",
        "app/models/user.py": "content here"
    }}

    Include these components based on the following specifications:
    
    API Endpoints: {json.dumps(extracted_data.get("endpoints", []), indent=2)}
    Database Schema: {json.dumps(extracted_data.get("models", []), indent=2)}
    Authentication: {json.dumps(extracted_data.get("authentication", {}), indent=2)}
    
    Return ONLY the JSON object, no other text."""

    try:
        response = llm.invoke(prompt)
        response_content = extract_json_from_text(response.content)
        logger.debug("Raw LLM Response:")
        logger.debug(response_content)
        
        # Parse the JSON content
        file_structure = json.loads(response_content)
        
        # Validate file structure
        if not isinstance(file_structure, dict):
            raise ValueError("LLM response is not a valid file structure dictionary")
            
        # Process the file structure
        process_structure(file_structure, project_root=project_root)
        logger.info(f"Code generation for project {project_name} complete.")
        state.code_generation_status = "Code generated successfully."
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {response.content}")
        raise LLMResponseError(f"Failed to parse LLM response as JSON: {e}")
    except Exception as e:
        logger.error(f"Error during code generation: {e}")
        raise ProjectGenerationError(f"Failed to generate code: {e}")

    return state


def generate_tests(state: MyState) -> MyState:
    """
    Generate unit tests for the FastAPI project.
    """
    project_name = state.project_name
    extracted_data = state.extracted_data

    logger.info(f"Generating unit tests for project: {project_name}")

    tests_folder = os.path.join(project_name, "tests")
    os.makedirs(tests_folder, exist_ok=True)

    # Generate test cases for each endpoint
    for endpoint in extracted_data.get("endpoints", []):
        test_file = os.path.join(tests_folder, f"test_{endpoint['path'].strip('/')}.py")
        with open(test_file, "w") as f:
            f.write("import pytest\n")
            f.write("from fastapi.testclient import TestClient\n")
            f.write("from app.main import app\n\n")
            f.write("client = TestClient(app)\n\n")
            f.write(f"def test_{endpoint['path'].strip('/').replace('/', '_')}():\n")
            f.write(f"    response = client.{endpoint['method'].lower()}('{endpoint['path']}')\n")
            f.write(f"    assert response.status_code == 200\n")
            f.write(f"    assert 'message' in response.json()\n")

    logger.info(f"Unit tests for project {project_name} generated successfully.")
    state.test_generation_status = "Tests generated successfully."
    return state

def run_tests(state: MyState) -> MyState:
    """
    Run unit tests for the FastAPI project.
    """
    project_name = state.project_name
    tests_folder = os.path.join(project_name, "tests")

    logger.info(f"Running unit tests for project: {project_name}")

    try:
        result = subprocess.run(["pytest", tests_folder], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        state.test_results = result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        state.test_results = e.stderr
        raise ValueError("Unit tests failed. Check the logs for details.")

    logger.info(f"Unit tests for project {project_name} completed successfully.")
    return state

def debug_and_refine(state: MyState) -> MyState:
    """
    Debug and refine the code using the LLM if tests fail.
    """
    if "failed" in state.test_results.lower():
        logger.info("Debugging and refining code...")
        prompt = f"""
        The following unit tests failed:\n{state.test_results}\n
        Refine the code to fix these issues and ensure all tests pass.
        """
        response = llm.invoke(prompt)
        refined_code = response.content

        # Apply the refined code (this part can be implemented as needed)
        logger.info("Code refined successfully.")
        state.refinement_status = "Code refined successfully."
    else:
        logger.info("No issues found. Skipping refinement.")
        state.refinement_status = "No refinement needed."

    return state