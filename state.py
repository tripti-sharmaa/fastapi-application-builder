from dataclasses import dataclass

@dataclass
class MyState:
    srs_text: str = None  # The extracted text from the SRS document
    project_name: str = None  # The name of the project
    extracted_data: dict = None  # The JSON response from the LLM
    project_structure: str = None  # The project structure details