from langsmith import RunEvaluator
from typing import Optional, Dict, Any
from langchain_core.messages import BaseMessage

class RaghuPersonaEvaluator(RunEvaluator):
    """Evaluates if responses maintain Raghu's persona"""
    
    def evaluate_run(self, run: Any, example: Optional[Dict] = None) -> Dict[str, Any]:
        if not run.outputs:
            return {"persona_score": 0, "reasoning": "No output found"}
            
        output = run.outputs.get("output", "")
        if isinstance(output, BaseMessage):
            output = output.content
            
        criteria = {
            "third_person": "Raghu" in output or "Raghunandan" in output,
            "no_ai_terms": all(term not in output.lower() 
                             for term in ["ai assistant", "as an ai", "I am an"]),
            "assertive_tone": any(word in output.lower() 
                                for word in ["will", "shall", "must", "always", "never"])
        }
        
        score = sum(criteria.values()) / len(criteria)
        
        return {
            "persona_score": score,
            "criteria_met": criteria,
            "reasoning": f"Score {score:.2f} based on persona criteria"
        }

class RelevanceEvaluator(RunEvaluator):
    """Evaluates if responses are relevant to the query"""
    
    def evaluate_run(self, run: Any, example: Optional[Dict] = None) -> Dict[str, Any]:
        query = run.inputs.get("messages", [{}])[-1].get("content", "")
        output = run.outputs.get("output", "")
        if isinstance(output, BaseMessage):
            output = output.content
            
        # Check if response addresses the query keywords
        query_keywords = set(query.lower().split())
        response_keywords = set(output.lower().split())
        keyword_overlap = len(query_keywords.intersection(response_keywords))
        
        return {
            "relevance_score": min(1.0, keyword_overlap / max(1, len(query_keywords))),
            "query_keywords": list(query_keywords),
            "matching_keywords": list(query_keywords.intersection(response_keywords))
        } 