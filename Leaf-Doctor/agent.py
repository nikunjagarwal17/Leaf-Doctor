"""
Agentic Orchestrator for Leaf Doctor

1. Autonomous Tool Selection - Agent decides which tools to use
2. Multi-step Reasoning - Plans, executes, and tracks progress
3. Self-Correction - Reflects on outputs and retries if needed
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from pathlib import Path


class ToolType(Enum):
    """Available tools the agent can use."""
    CNN_CLASSIFIER = "cnn_classifier"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    LLM_ADVISOR = "llm_advisor"
    CONFIDENCE_CHECKER = "confidence_checker"


class TaskStatus(Enum):
    """Status of a task in the reasoning chain."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


@dataclass
class ReasoningStep:
    """A single step in the multi-step reasoning chain."""
    step_id: int
    action: str
    tool: Optional[ToolType]
    input_data: dict
    output_data: Optional[dict] = None
    status: TaskStatus = TaskStatus.PENDING
    reflection: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class AgentPlan:
    """Complete execution plan created by the agent."""
    goal: str
    steps: list[ReasoningStep] = field(default_factory=list)
    current_step: int = 0
    final_output: Optional[dict] = None
    execution_trace: list[dict] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Result of the agent's self-reflection."""
    is_satisfactory: bool
    confidence: float
    issues: list[str]
    suggestions: list[str]
    should_retry: bool


class AgentOrchestrator:
    """
    Main agentic orchestrator that autonomously:
    1. Analyzes user intent
    2. Plans multi-step execution
    3. Selects and invokes tools
    4. Reflects on outputs and self-corrects
    """
    
    # Confidence thresholds for self-correction
    MIN_PREDICTION_CONFIDENCE = 0.6
    MIN_RETRIEVAL_SCORE = 0.3
    MAX_RETRIES = 2
    
    def __init__(self, tools: dict[str, Callable]):
        """
        Initialize the agent with available tools.
        
        Args:
            tools: Dictionary mapping tool names to callable functions
        """
        self.tools = tools
        self.execution_history: list[AgentPlan] = []
    
    def analyze_intent(self, user_input: dict) -> dict:
        """
        Analyze user intent to determine what the user wants.
        
        Returns:
            Intent classification with required tools
        """
        intent = {
            "has_images": False,
            "has_question": False,
            "has_disease_context": False,
            "requires_diagnosis": False,
            "requires_advice": False,
            "requires_retrieval": False,
            "complexity": "simple"
        }
        
        # Check for images
        if user_input.get("images") or user_input.get("image_paths"):
            intent["has_images"] = True
            intent["requires_diagnosis"] = True
        
        # Check for text query
        question = user_input.get("prompt", "").strip()
        if question:
            intent["has_question"] = True
            intent["requires_advice"] = True
            intent["requires_retrieval"] = True
        
        # Check for disease context
        if user_input.get("disease"):
            intent["has_disease_context"] = True
        
        # Determine complexity
        if intent["has_images"] and intent["has_question"]:
            intent["complexity"] = "complex"
        elif intent["has_images"] or intent["has_question"]:
            intent["complexity"] = "moderate"
        
        return intent
    
    def plan_execution(self, intent: dict, user_input: dict) -> AgentPlan:
        """
        Create a multi-step execution plan based on analyzed intent.
        
        This is where autonomous tool selection happens - the agent
        decides which tools to use and in what order.
        """
        goal = self._formulate_goal(intent)
        plan = AgentPlan(goal=goal)
        step_id = 0
        
        # Step 1: Image Classification (if images present)
        if intent["requires_diagnosis"]:
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Classify leaf images to identify disease",
                tool=ToolType.CNN_CLASSIFIER,
                input_data={
                    "image_paths": user_input.get("image_paths", []),
                    "note": user_input.get("note", "")
                }
            ))
            
            # Step 1.5: Confidence Check (for self-correction)
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Verify prediction confidence is acceptable",
                tool=ToolType.CONFIDENCE_CHECKER,
                input_data={"check_type": "prediction"}
            ))
        
        # Step 2: Knowledge Retrieval (if question or diagnosis)
        if intent["requires_retrieval"] or intent["requires_diagnosis"]:
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Retrieve relevant knowledge from disease database",
                tool=ToolType.KNOWLEDGE_RETRIEVER,
                input_data={
                    "query": user_input.get("prompt", ""),
                    "disease": user_input.get("disease")
                }
            ))
            
            # Step 2.5: Retrieval Quality Check
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Verify retrieved context is relevant",
                tool=ToolType.CONFIDENCE_CHECKER,
                input_data={"check_type": "retrieval"}
            ))
        
        # Step 3: LLM Advice Generation (if question present)
        if intent["requires_advice"]:
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Generate expert advice using retrieved context",
                tool=ToolType.LLM_ADVISOR,
                input_data={
                    "prompt": user_input.get("prompt", ""),
                    "disease": user_input.get("disease")
                }
            ))
            
            # Step 3.5: Response Quality Check
            step_id += 1
            plan.steps.append(ReasoningStep(
                step_id=step_id,
                action="Evaluate response quality and completeness",
                tool=ToolType.CONFIDENCE_CHECKER,
                input_data={"check_type": "response"}
            ))
        
        return plan
    
    def execute_plan(self, plan: AgentPlan) -> dict:
        """
        Execute the planned steps with reflection and self-correction.
        """
        accumulated_context = {}
        
        for i, step in enumerate(plan.steps):
            plan.current_step = i
            step.status = TaskStatus.IN_PROGRESS
            
            # Log the step
            trace_entry = {
                "step_id": step.step_id,
                "action": step.action,
                "tool": step.tool.value if step.tool else None,
                "status": "executing"
            }
            plan.execution_trace.append(trace_entry)
            
            try:
                # Execute the tool
                result = self._execute_tool(step, accumulated_context)
                step.output_data = result
                
                # Store results in accumulated context
                self._update_context(accumulated_context, step.tool, result)
                
                # Reflect on the output
                reflection = self._reflect_on_output(step, accumulated_context)
                step.reflection = reflection.issues[0] if reflection.issues else "Output looks good"
                
                # Self-correction logic
                if reflection.should_retry and step.retry_count < step.max_retries:
                    step.status = TaskStatus.NEEDS_RETRY
                    step.retry_count += 1
                    
                    # Apply correction strategy
                    corrected_result = self._apply_correction(step, reflection, accumulated_context)
                    if corrected_result:
                        step.output_data = corrected_result
                        self._update_context(accumulated_context, step.tool, corrected_result)
                        step.status = TaskStatus.COMPLETED
                        step.reflection = f"Self-corrected after {step.retry_count} attempt(s)"
                    else:
                        step.status = TaskStatus.COMPLETED  # Accept even if not perfect
                else:
                    step.status = TaskStatus.COMPLETED
                
                trace_entry["status"] = step.status.value
                trace_entry["reflection"] = step.reflection
                
            except Exception as e:
                step.status = TaskStatus.FAILED
                step.reflection = f"Error: {str(e)}"
                trace_entry["status"] = "failed"
                trace_entry["error"] = str(e)
        
        # Compile final output
        plan.final_output = self._compile_final_output(plan, accumulated_context)
        return plan.final_output
    
    def _execute_tool(self, step: ReasoningStep, context: dict) -> dict:
        """Execute a specific tool based on the step configuration."""
        if step.tool == ToolType.CNN_CLASSIFIER:
            if "cnn_predict" not in self.tools:
                return {"error": "CNN classifier not available"}
            return self.tools["cnn_predict"](step.input_data.get("image_paths", []))
        
        elif step.tool == ToolType.KNOWLEDGE_RETRIEVER:
            if "retrieve" not in self.tools:
                return {"context": [], "formatted": "No retriever available"}
            
            # Use disease from classification if available
            disease = step.input_data.get("disease") or context.get("disease")
            query = step.input_data.get("query", "")
            
            # If no query but we have a diagnosis, create a query
            if not query and disease:
                query = f"What are the treatment options for {disease}?"
            
            return self.tools["retrieve"](query, disease)
        
        elif step.tool == ToolType.LLM_ADVISOR:
            if "generate_advice" not in self.tools:
                return {"message": "LLM advisor not available", "source": "none"}
            
            disease = step.input_data.get("disease") or context.get("disease")
            prompt = step.input_data.get("prompt", "")
            retrieved_context = context.get("retrieved_context", [])
            formatted_context = context.get("formatted_context", "")
            
            return self.tools["generate_advice"](prompt, disease, retrieved_context, formatted_context)
        
        elif step.tool == ToolType.CONFIDENCE_CHECKER:
            return self._check_confidence(step.input_data.get("check_type"), context)
        
        return {"error": f"Unknown tool: {step.tool}"}
    
    def _check_confidence(self, check_type: str, context: dict) -> dict:
        """Check confidence levels for self-correction decisions."""
        result = {
            "check_type": check_type,
            "passed": True,
            "confidence": 1.0,
            "issues": []
        }
        
        if check_type == "prediction":
            confidence = context.get("prediction_confidence", 0)
            result["confidence"] = confidence
            if confidence < self.MIN_PREDICTION_CONFIDENCE:
                result["passed"] = False
                result["issues"].append(
                    f"Low prediction confidence ({confidence:.2f}). "
                    f"Consider multiple images or manual verification."
                )
        
        elif check_type == "retrieval":
            scores = context.get("retrieval_scores", [])
            if scores:
                avg_score = sum(scores) / len(scores)
                result["confidence"] = avg_score
                if avg_score < self.MIN_RETRIEVAL_SCORE:
                    result["passed"] = False
                    result["issues"].append(
                        f"Low retrieval relevance ({avg_score:.2f}). "
                        f"Query may not match knowledge base well."
                    )
        
        elif check_type == "response":
            response_text = context.get("llm_response", "")
            # Check for quality indicators
            if len(response_text) < 50:
                result["passed"] = False
                result["confidence"] = 0.3
                result["issues"].append("Response too short, may lack detail")
            elif any(err in response_text.lower() for err in ["error", "unavailable", "failed"]):
                result["passed"] = False
                result["confidence"] = 0.2
                result["issues"].append("Response contains error indicators")
        
        return result
    
    def _reflect_on_output(self, step: ReasoningStep, context: dict) -> ReflectionResult:
        """
        Reflect on the output of a step to determine if it's satisfactory.
        This is the core of the self-correction mechanism.
        """
        issues = []
        suggestions = []
        confidence = 1.0
        
        if step.tool == ToolType.CNN_CLASSIFIER:
            pred_conf = step.output_data.get("confidence", 0)
            confidence = pred_conf
            
            if pred_conf < 0.5:
                issues.append(f"Very low confidence ({pred_conf:.2%})")
                suggestions.append("Request additional images for better accuracy")
            elif pred_conf < self.MIN_PREDICTION_CONFIDENCE:
                issues.append(f"Moderate confidence ({pred_conf:.2%})")
                suggestions.append("Consider alternative diagnoses in ranked list")
            
            # Check for disagreement among multi-image predictions
            per_image = step.output_data.get("per_image", [])
            if len(per_image) > 1:
                diseases = [img["disease"] for img in per_image]
                if len(set(diseases)) > 1:
                    issues.append("Multiple images show different diseases")
                    suggestions.append("Review individual image predictions")
        
        elif step.tool == ToolType.KNOWLEDGE_RETRIEVER:
            retrieved = step.output_data.get("context", [])
            if not retrieved:
                issues.append("No relevant context retrieved")
                suggestions.append("Broaden the search query")
                confidence = 0.0
            else:
                scores = [item.get("score", 0) for item in retrieved]
                avg_score = sum(scores) / len(scores) if scores else 0
                confidence = avg_score
                if avg_score < self.MIN_RETRIEVAL_SCORE:
                    issues.append(f"Low average relevance score ({avg_score:.2f})")
                    suggestions.append("Query may need rephrasing")
        
        elif step.tool == ToolType.LLM_ADVISOR:
            message = step.output_data.get("message", "")
            source = step.output_data.get("source", "")
            
            if "error" in source.lower() or "error" in message.lower():
                issues.append("LLM response contains errors")
                suggestions.append("Fall back to knowledge base")
                confidence = 0.2
            elif len(message) < 50:
                issues.append("Response is too brief")
                suggestions.append("Request more detailed advice")
                confidence = 0.5
            elif "unavailable" in message.lower():
                issues.append("Advisor service unavailable")
                suggestions.append("Use cached knowledge")
                confidence = 0.3
        
        should_retry = len(issues) > 0 and confidence < 0.5
        
        return ReflectionResult(
            is_satisfactory=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            should_retry=should_retry
        )
    
    def _apply_correction(self, step: ReasoningStep, reflection: ReflectionResult, 
                         context: dict) -> Optional[dict]:
        """
        Apply self-correction strategies based on reflection results.
        """
        if step.tool == ToolType.CNN_CLASSIFIER:
            # For CNN, we can't really retry - but we can enhance the output
            # by including alternative predictions
            original = step.output_data or {}
            original["low_confidence_warning"] = True
            original["agent_note"] = (
                "Low prediction confidence detected. "
                "Please verify with additional images or expert consultation."
            )
            return original
        
        elif step.tool == ToolType.KNOWLEDGE_RETRIEVER:
            # Retry with broader query
            if "retrieve" in self.tools:
                disease = context.get("disease", "plant disease")
                broader_query = f"General treatment and prevention for {disease}"
                retry_result = self.tools["retrieve"](broader_query, disease)
                
                # Check if retry improved results
                new_scores = [item.get("score", 0) for item in retry_result.get("context", [])]
                if new_scores and sum(new_scores) / len(new_scores) > self.MIN_RETRIEVAL_SCORE:
                    retry_result["retry_improved"] = True
                    return retry_result
        
        elif step.tool == ToolType.LLM_ADVISOR:
            # Fall back to formatted knowledge base content
            retrieved = context.get("retrieved_context", [])
            if retrieved:
                fallback_msg = "Based on our knowledge base:\n\n"
                for item in retrieved[:2]:
                    fallback_msg += f"• {item.get('text', '')[:200]}...\n\n"
                return {
                    "message": fallback_msg,
                    "source": "knowledge_base_fallback",
                    "reasoning": "Agent fell back to knowledge base due to LLM issues",
                    "context": retrieved
                }
        
        return None
    
    def _update_context(self, context: dict, tool: ToolType, result: dict):
        """Update accumulated context with tool results."""
        if tool == ToolType.CNN_CLASSIFIER:
            context["disease"] = result.get("disease")
            context["prediction_confidence"] = result.get("confidence", 0)
            context["diagnosis_result"] = result
        
        elif tool == ToolType.KNOWLEDGE_RETRIEVER:
            retrieved = result.get("context", [])
            context["retrieved_context"] = retrieved
            context["retrieval_scores"] = [item.get("score", 0) for item in retrieved]
            context["formatted_context"] = result.get("formatted", "")
        
        elif tool == ToolType.LLM_ADVISOR:
            context["llm_response"] = result.get("message", "")
            context["advice_result"] = result
        
        elif tool == ToolType.CONFIDENCE_CHECKER:
            context[f"{result.get('check_type', 'unknown')}_check"] = result
    
    def _formulate_goal(self, intent: dict) -> str:
        """Create a human-readable goal statement."""
        goals = []
        if intent["requires_diagnosis"]:
            goals.append("diagnose plant disease from images")
        if intent["requires_advice"]:
            goals.append("provide expert treatment advice")
        if intent["requires_retrieval"]:
            goals.append("retrieve relevant knowledge")
        
        return "Agent goal: " + " and ".join(goals) if goals else "Process user request"
    
    def _compile_final_output(self, plan: AgentPlan, context: dict) -> dict:
        """Compile all results into the final output."""
        output = {
            "goal": plan.goal,
            "success": all(s.status == TaskStatus.COMPLETED for s in plan.steps),
            "reasoning_chain": []
        }
        
        # Build reasoning chain for transparency
        for step in plan.steps:
            output["reasoning_chain"].append({
                "step": step.step_id,
                "action": step.action,
                "tool": step.tool.value if step.tool else None,
                "status": step.status.value,
                "reflection": step.reflection,
                "retries": step.retry_count
            })
        
        # Add diagnosis results if available
        if "diagnosis_result" in context:
            output["diagnosis"] = context["diagnosis_result"]
        
        # Add advice results if available
        if "advice_result" in context:
            output["advice"] = context["advice_result"]
        
        # Add retrieved context
        if "retrieved_context" in context:
            output["context"] = context["retrieved_context"]
        
        # Add any quality checks
        for key in context:
            if key.endswith("_check"):
                output[key] = context[key]
        
        return output
    
    def get_execution_summary(self, plan: AgentPlan) -> str:
        """Generate a human-readable summary of execution."""
        lines = [f"🎯 {plan.goal}", ""]
        
        for step in plan.steps:
            status_icon = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.NEEDS_RETRY: "🔄",
                TaskStatus.IN_PROGRESS: "⏳",
                TaskStatus.PENDING: "⏸️"
            }.get(step.status, "❓")
            
            lines.append(f"{status_icon} Step {step.step_id}: {step.action}")
            if step.reflection:
                lines.append(f"   └─ {step.reflection}")
            if step.retry_count > 0:
                lines.append(f"   └─ Retried {step.retry_count} time(s)")
        
        return "\n".join(lines)


def create_agent_tools(
    cnn_predict_fn: Callable,
    retrieve_fn: Callable,
    generate_advice_fn: Callable
) -> dict[str, Callable]:
    """
    Factory function to create the tools dictionary for the agent.
    
    Args:
        cnn_predict_fn: Function that takes image paths and returns prediction
        retrieve_fn: Function that takes query and disease, returns context
        generate_advice_fn: Function that takes (prompt, disease, context, formatted_context) and returns advice
    """
    return {
        "cnn_predict": cnn_predict_fn,
        "retrieve": retrieve_fn,
        "generate_advice": generate_advice_fn
    }
