
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    generationNormal: bool