"""
LAYER 4: REASONING & INTELLIGENCE
The Cognitive Core. Implements:
1. Intent Inference
2. Causal Reasoning
3. Counterfactual Reasoning
4. Ethical Awareness
5. Risk Scoring
6. Confidence Fusion
"""
import time

class CognitiveEngine:
    def __init__(self):
        print("🧠 Initializing Reasoning Layer...")
        
    def analyze_situation(self, entity, context, anomaly_score):
        """
        Run the full cognitive pipeline on a tracked entity.
        Returns: Comprehensive Risk Assessment
        """
        # 1. Intent Inference
        intent, intent_conf = self._infer_intent(entity, context)
        
        # 2. Causal Reasoning
        cause_effect = self._reason_causality(intent, context)
        
        # 3. Counterfactual Reasoning
        counterfactual = self._reason_counterfactual(intent, context)
        
        # 4. Impact Severity (Potential)
        impact = self._assess_impact(intent, context)
        
        # 5. Risk Scoring (Dual-Axis)
        raw_risk = (intent_conf * 0.7 + anomaly_score * 0.3) * impact
        
        # 6. Ethical Filter
        final_risk, ethical_note = self._apply_ethics(raw_risk, entity, context)
        
        # 7. Confidence Fusion
        trust_score = self._fuse_confidence(entity, intent_conf, anomaly_score)
        
        return {
            'intent': intent,
            'intent_confidence': intent_conf,
            'cause_chain': cause_effect,
            'counterfactual': counterfactual,
            'risk_score': final_risk,
            'trust_score': trust_score,
            'ethical_context': ethical_note,
            'justification': f"Detected {intent} with {intent_conf:.2f} confidence. Risk scaled to {final_risk:.2f} due to {ethical_note}."
        }
        
    def _infer_intent(self, entity, context):
        """Classify behavior intent"""
        speed = entity['speed']
        pose = entity['pose']
        
        if pose and pose['state'] == 'hands_up':
            return 'surrender_or_panic', 0.95
            
        if speed > 10.0: # Arbitrary high unit
            return 'evasive_running', 0.8
        elif speed < 0.5:
            return 'loitering', 0.6
        else:
            return 'transit', 0.9

    def _reason_causality(self, intent, context):
        """Build cause-effect chain"""
        if intent == 'evasive_running':
            return "Running likely caused by perceived threat or panic trigger."
        return "Behavior appears self-directed."

    def _reason_counterfactual(self, intent, context):
        """What if?"""
        if intent == 'loitering':
            return "If subject had moved within 30s, incident would be classified as transit."
        return "No alternative outcome generated."
        
    def _assess_impact(self, intent, context):
        """Estimate severity of impact (0.0-1.0)"""
        severity_map = {
            'surrender_or_panic': 0.9,
            'evasive_running': 0.7,
            'loitering': 0.3,
            'transit': 0.0
        }
        return severity_map.get(intent, 0.1)

    def _apply_ethics(self, risk, entity, context):
        """Adjust based on social context"""
        # Example: Reduce severity for loitering in waiting zones
        # (This would use scene semantics in a real full map)
        if False: # Placeholder for 'is_waiting_zone'
            return risk * 0.5, "Reduced risk: Authorized waiting area."
        return risk, "Standard risk profile applied."

    def _fuse_confidence(self, entity, intent_conf, anomaly_score):
        """Fuse detection conf, intent conf, etc."""
        det_conf = entity['confidence']
        return (det_conf + intent_conf + (1.0 - anomaly_score)) / 3.0
