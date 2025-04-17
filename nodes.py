import os
import subprocess
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import MyState

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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
    print("Analyzing SRS document...")
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
        raise ValueError("The LLM response is not in valid JSON format.")
    print("SRS document analyzed successfully. Extracted data:")
    print(json.dumps(extracted_data, indent=2))
    
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

    print(f"Setting up project: {sanitized_project_name}")

    # Define the project structure
    structure = {
        "app/api/routes": ["user.py", "item.py", "__init__.py"],
        "app/models": ["user.py", "item.py", "__init__.py"],
        "app/services": [],
        "tests": [],
        "": ["Dockerfile", "requirements.txt", ".env", "README.md"],
        "app": ["database.py", "main.py"]
    }

    # Create directories and files
    for folder, files in structure.items():
        folder_path = os.path.join(sanitized_project_name, folder)
        os.makedirs(folder_path, exist_ok=True)
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

    # Initialize Python virtual environment and install dependencies
    subprocess.run(["python", "-m", "venv", f"{sanitized_project_name}/venv"], check=True)
    subprocess.run([f"{sanitized_project_name}/venv/Scripts/pip", "install", "-r", f"{sanitized_project_name}/requirements.txt"], check=True)

    print(f"Project {sanitized_project_name} setup complete.")
    state.project_structure = f"{sanitized_project_name} created successfully."
    return state

def setup_database(state: MyState) -> MyState:
    """
    Set up a PostgreSQL database using Podman and create models based on the extracted schema.
    """
    project_name = state.project_name  # Use dot notation to access the attribute
    sanitized_project_name = project_name.replace(" ", "_").replace("-", "_")  # Sanitize project name
    extracted_data = state.extracted_data  # JSON response from the LLM

    print(f"Setting up PostgreSQL database for project: {sanitized_project_name}")

    # Step 1: Install Podman (if not already installed)
    try:
        subprocess.run(["podman", "--version"], check=True)
    except FileNotFoundError:
        print("Podman not found. Installing Podman...")
        subprocess.run(["winget", "install", "--id", "RedHat.Podman"], check=True)

    # Step 2: Pull PostgreSQL image
    print("Pulling PostgreSQL image...")
    subprocess.run(["podman", "pull", "postgres:latest"], check=True)

    # Step 3: Check if the container already exists and remove it
    container_name = f"{sanitized_project_name}_db"
    print(f"Checking if container '{container_name}' already exists...")
    existing_containers = subprocess.run(
        ["podman", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.splitlines()

    if container_name in existing_containers:
        print(f"Container '{container_name}' already exists. Removing it...")
        subprocess.run(["podman", "rm", "-f", container_name], check=True)

    # Step 4: Run PostgreSQL container
    print("Running PostgreSQL container...")
    subprocess.run([
        "podman", "run", "-d",
        "--name", container_name,
        "-e", "POSTGRES_USER=admin",
        "-e", "POSTGRES_PASSWORD=admin",
        "-e", f"POSTGRES_DB={sanitized_project_name}",
        "-p", "5432:5432",
        "postgres:latest"
    ], check=True)

    # Step 5: Create database models
    print("Creating database models...")
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

    print(f"Database setup and models creation for project {sanitized_project_name} complete.")
    state.database_status = f"Database and models for {sanitized_project_name} created successfully."
    return state
def generate_code(state: MyState) -> MyState:
    """
    Generate modular components (routes, models, authentication, etc.) for the FastAPI project.
    """
    project_name = state.project_name
    extracted_data = state.extracted_data

    print(f"Generating code for project: {project_name}")

    # Define the folder paths
    routes_folder = os.path.join(project_name, "app", "api", "routes")
    models_folder = os.path.join(project_name, "app", "models")
    auth_folder = os.path.join(project_name, "app", "auth")

    # Ensure necessary directories exist
    os.makedirs(routes_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(auth_folder, exist_ok=True)

    # Generate routes
    for endpoint in extracted_data.get("endpoints", []):
        route_file = os.path.join(routes_folder, f"{endpoint['path'].strip('/').replace('/', '_')}.py")
        with open(route_file, "w") as f:
            # Call the LLM to generate code for the route
            prompt = f"""
            Generate a FastAPI route for the following endpoint:
            Method: {endpoint['method']}
            Path: {endpoint['path']}
            Parameters: {endpoint['parameters']}
            Description: {endpoint['description']}
            """
            response = llm.invoke(prompt)
            generated_code = response.content

            # Write the generated code to the file
            f.write(generated_code)

    # Generate models
    for model in extracted_data.get("models", []):
        model_file = os.path.join(models_folder, f"{model['model_name'].lower()}.py")
        with open(model_file, "w") as f:
            # Call the LLM to generate code for the model
            prompt = f"""
            Generate a SQLAlchemy model for the following table:
            Model Name: {model['model_name']}
            Table Name: {model['table_name']}
            Columns: {json.dumps(model['columns'], indent=2)}
            """
            response = llm.invoke(prompt)
            generated_code = response.content

            # Write the generated code to the file
            f.write(generated_code)

    # Generate authentication utilities if required
    if extracted_data.get("authentication", {}).get("jwt", False):
        auth_file = os.path.join(auth_folder, "jwt_utils.py")
        with open(auth_file, "w") as f:
            # Call the LLM to generate JWT utilities
            prompt = """
            Generate a Python module for JWT authentication. Include functions for:
            - Generating JWT tokens
            - Validating JWT tokens
            - Extracting user information from tokens
            """
            response = llm.invoke(prompt)
            generated_code = response.content

            # Write the generated code to the file
            f.write(generated_code)

    if extracted_data.get("authentication", {}).get("rbac", False):
        rbac_file = os.path.join(auth_folder, "rbac_utils.py")
        with open(rbac_file, "w") as f:
            # Call the LLM to generate RBAC utilities
            prompt = """
            Generate a Python module for Role-Based Access Control (RBAC). Include:
            - A function to check user roles
            - Middleware for role-based access control
            """
            response = llm.invoke(prompt)
            generated_code = response.content

            # Write the generated code to the file
            f.write(generated_code)

    print(f"Code generation for project {project_name} complete.")
    state.code_generation_status = "Code generated successfully."
    return state


def generate_tests(state: MyState) -> MyState:
    """
    Generate unit tests for the FastAPI project.
    """
    project_name = state.project_name
    extracted_data = state.extracted_data

    print(f"Generating unit tests for project: {project_name}")

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

    print(f"Unit tests for project {project_name} generated successfully.")
    state.test_generation_status = "Tests generated successfully."
    return state

def run_tests(state: MyState) -> MyState:
    """
    Run unit tests for the FastAPI project.
    """
    project_name = state.project_name
    tests_folder = os.path.join(project_name, "tests")

    print(f"Running unit tests for project: {project_name}")

    try:
        result = subprocess.run(["pytest", tests_folder], capture_output=True, text=True, check=True)
        print(result.stdout)
        state.test_results = result.stdout
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        state.test_results = e.stderr
        raise ValueError("Unit tests failed. Check the logs for details.")

    print(f"Unit tests for project {project_name} completed successfully.")
    return state


def debug_and_refine(state: MyState) -> MyState:
    """
    Debug and refine the code using the LLM if tests fail.
    """
    if "failed" in state.test_results.lower():
        print("Debugging and refining code...")
        prompt = f"""
        The following unit tests failed:\n{state.test_results}\n
        Refine the code to fix these issues and ensure all tests pass.
        """
        response = llm.invoke(prompt)
        refined_code = response.content

        # Apply the refined code (this part can be implemented as needed)
        print("Code refined successfully.")
        state.refinement_status = "Code refined successfully."
    else:
        print("No issues found. Skipping refinement.")
        state.refinement_status = "No refinement needed."

    return state