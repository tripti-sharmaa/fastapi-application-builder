from langgraph.graph import StateGraph
from state import MyState
from nodes import analyze_srs, setup_project, setup_database, generate_code, generate_tests, run_tests, debug_and_refine

async def run_workflow(srs_text: str, project_name: str):
    """
    Run the StateGraph workflow for Milestones 1, 2, 3, and 4.
    """
    stategraph_workflow = StateGraph(state_schema=MyState)

    # Add nodes to the workflow
    stategraph_workflow.add_node("Analyze SRS", analyze_srs)
    stategraph_workflow.add_node("Setup Project", setup_project)
    stategraph_workflow.add_node("Setup Database", setup_database)
    stategraph_workflow.add_node("Generate Code", generate_code)
    stategraph_workflow.add_node("Generate Tests", generate_tests)
    stategraph_workflow.add_node("Run Tests", run_tests)
    stategraph_workflow.add_node("Debug and Refine", debug_and_refine)

    # Define the workflow edges
    stategraph_workflow.add_edge("Analyze SRS", "Setup Project")
    stategraph_workflow.add_edge("Setup Project", "Setup Database")
    stategraph_workflow.add_edge("Setup Database", "Generate Code")
    stategraph_workflow.add_edge("Generate Code", "Generate Tests")
    stategraph_workflow.add_edge("Generate Tests", "Run Tests")
    stategraph_workflow.add_edge("Run Tests", "Debug and Refine")

    stategraph_workflow.set_entry_point("Analyze SRS")
    compiled_graph = stategraph_workflow.compile()

    initial_state = MyState(srs_text=srs_text, project_name=project_name)
    result = compiled_graph.invoke(initial_state)

    return result