"""
LAYER 5: NARRATIVE & REPORTING
Generates human-readable stories, briefs, and explanations.
"""
import time
from datetime import datetime

class NarrativeGenerator:
    def __init__(self):
        print("📝 Initializing Narrative Layer...")
        
    def generate_realtime_log(self, entity, cognitive_state):
        """Generate a one-line log for the live feed"""
        intent = cognitive_state['intent']
        conf = cognitive_state['intent_confidence']
        return f"[{datetime.now().strftime('%H:%M:%S')}] Subject {entity['id']} classified as {intent.upper()} ({conf:.0%})."

    def generate_incident_brief(self, event_id, entity, cognitive_state, history):
        """
        Generate a professional Markdown brief.
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        intent = cognitive_state['intent']
        risk = cognitive_state['risk_score']
        
        brief = f"""
# INCIDENT BRIEF #{event_id}
**Time:** {now}
**Subject:** Track ID {entity['id']}
**Classification:** {intent.upper()}
**Risk Score:** {risk:.2f}/1.00

## 1. Executive Summary
At {now}, surveillance detected an individual engaged in **{intent}** behavior. 
Cognitive analysis suggests a confidence of **{cognitive_state['intent_confidence']:.2f}**. 
The behavior was flagged as anomalous with an impact score of {risk:.2f}.

## 2. Reasoning Trace
- **Observation:** Subject moving at {entity['speed']:.2f} px/s.
- **Intent Inference:** {intent} (Confidence: {cognitive_state['intent_confidence']:.2f})
- **Causal Logic:** {cognitive_state['cause_chain']}
- **Counterfactual:** {cognitive_state['counterfactual']}
- **Ethical Context:** {cognitive_state['ethical_context']}

## 3. Historical Context
Subject has been tracked for {len(history)} frames.
Previous interactions: {len(history)} data points logged in memory.

## 4. Recommendation
{self._get_recommendation(risk)}
"""
        return brief

    def _get_recommendation(self, risk):
        if risk > 0.8: return "**URGENT:** Dispatch security personnel immediately."
        if risk > 0.5: return "**ALERT:** Continue monitoring closely. Log incident."
        return "**INFO:** No immediate action required. Normalized behavior."
