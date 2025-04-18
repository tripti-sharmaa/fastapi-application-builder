# FastAPI Application Generator

A sophisticated AI-powered tool that automatically generates production-ready FastAPI applications from Software Requirements Specification (SRS) documents using LangGraph and LLM technology.

## 🌟 Features

- **Automated Analysis**: Intelligently analyzes SRS documents to extract API endpoints, data models, and authentication requirements
- **Complete Project Generation**: Creates a fully structured FastAPI project with:
  - Modular architecture
  - Database integration (PostgreSQL)
  - Authentication and authorization
  - API endpoints with documentation
  - Unit tests
- **Smart Code Generation**: Uses LLM to generate production-quality code
- **Test-Driven Development**: Automatically generates and runs unit tests
- **Autonomous Debugging**: Self-correcting code generation with automated debugging and refinement
- **Database Integration**: Automated PostgreSQL setup using Podman
- **Best Practices**: Follows FastAPI and Python best practices

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Podman (for PostgreSQL containerization)
- FastAPI
- PostgreSQL

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd fastapi-application-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=5432
```

## 📝 Usage

1. Prepare your SRS document in .docx format

2. Use the FastAPI endpoint to generate your application:
```bash
curl -X POST "http://localhost:8000/run-workflow/" \
     -H "Content-Type: multipart/form-data" \
     -F "srs_document=@path/to/your/srs.docx" \
     -F "project_name=your_project_name"
```

## 🏗️ Generated Project Structure

```
project_root/
│── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── user.py
│   │   │   ├── item.py
│   │   │   └── __init__.py
│   ├── models/
│   │   ├── user.py
│   │   ├── item.py
│   │   └── __init__.py
│   ├── services/
│   ├── database.py
│   ├── main.py
│── tests/
│── Dockerfile
│── requirements.txt
│── .env
│── README.md
```

## 🔄 Workflow Process

1. **SRS Analysis**: Analyzes the provided SRS document to extract:
   - API endpoints and parameters
   - Data models and relationships
   - Authentication requirements
   - Business logic rules

2. **Project Setup**: Creates a structured FastAPI project with all necessary directories and files

3. **Database Setup**: Configures PostgreSQL database with:
   - Automated container setup using Podman
   - Database models creation
   - Migration scripts

4. **Code Generation**: Generates:
   - API routes and endpoints
   - Data models
   - Service layer
   - Authentication middleware
   - Configuration files

5. **Testing**: Creates and runs:
   - Unit tests for all endpoints
   - Integration tests
   - Automated test execution

6. **Refinement**: Performs:
   - Automated debugging
   - Code optimization
   - Test-driven refinements

## 🛠️ Technical Details

### Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **LangGraph**: AI workflow orchestration
- **Groq**: LLM integration for code generation
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation
- **Pytest**: Testing framework
- **Podman**: Container management

### Key Components

- `app_workflow.py`: Orchestrates the generation workflow
- `nodes.py`: Contains the core generation logic
- `state.py`: Manages workflow state
- `main.py`: FastAPI application entry point

## 📚 API Documentation

### POST /run-workflow/

Generate a new FastAPI application from an SRS document.

**Parameters**:
- `srs_document`: `.docx` file containing the SRS
- `project_name`: Name for the generated project

**Response**:
```json
{
    "message": "Workflow executed successfully.",
    "result": {
        "project_path": "path/to/generated/project",
        "status": "success"
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

[Your License Here]

## ✨ Acknowledgments

- FastAPI community
- LangChain developers
- Groq team for LLM support