from graph.state import PipelineState

def route_evaluation(state : PipelineState) -> PipelineState :

    if state["all_approved"] == True or state["iterations"] >= state["max_iterations"] :
        return 'approved'
    else :
        return 're-write'