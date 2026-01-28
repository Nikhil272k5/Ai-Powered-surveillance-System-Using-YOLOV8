"""
AbnoGuard Cognition - Explainability Engine
Every alert must justify itself with clear reasoning
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class DecisionFactor:
    """A single factor contributing to the decision"""
    factor_type: str        # detection, motion, pose, zone, history
    factor_name: str        # specific factor
    value: Any              # actual value
    threshold: Any          # comparison threshold
    contribution: float     # 0-1, how much this contributed
    explanation: str        # human-readable explanation


@dataclass
class ExplainableDecision:
    """A fully explainable alert decision"""
    decision_id: str
    timestamp: float
    
    # The decision
    alert_raised: bool
    alert_type: str
    severity: str
    
    # Confidence
    overall_confidence: float
    
    # Contributing factors
    factors: List[DecisionFactor]
    
    # Reasoning chain
    reasoning_steps: List[str]
    
    # Counter-arguments (why it might be false positive)
    counter_arguments: List[str]
    
    # Final justification
    summary: str
    
    # Comparable cases
    similar_past_cases: List[str]


class ExplainabilityEngine:
    """
    Generates explanations for every alert decision.
    Enables AI accountability and human oversight.
    
    Every alert includes:
    - What triggered it (detection factors)
    - Why it's concerning (reasoning)
    - How confident the system is (score breakdown)
    - What could make it wrong (counter-arguments)
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Decision history for similar case lookup
        self.decision_history: List[ExplainableDecision] = []
        self.max_history = self.config.get('max_history', 100)
        
        # Threshold configurations (for comparison in explanations)
        self.thresholds = {
            'speed_spike': 15.0,
            'loitering_duration': 120.0,
            'abandoned_duration': 60.0,
            'restricted_zone_dwell': 5.0,
            'stress_score': 0.6,
            'counterflow_angle': 120.0,
            'threat_score': 0.5
        }
        
        print("🔍 Explainability Engine initialized")
    
    def explain_alert(self, alert: Dict, perception_data: Dict, 
                     cognition_data: Dict) -> ExplainableDecision:
        """
        Generate complete explanation for an alert
        
        Args:
            alert: The alert being explained
            perception_data: Data from perception layer
            cognition_data: Data from cognition layer
        
        Returns:
            ExplainableDecision with full breakdown
        """
        decision_id = f"DEC-{int(time.time())}-{len(self.decision_history)}"
        
        # Extract factors from perception
        perception_factors = self._extract_perception_factors(alert, perception_data)
        
        # Extract factors from cognition
        cognition_factors = self._extract_cognition_factors(alert, cognition_data)
        
        all_factors = perception_factors + cognition_factors
        
        # Build reasoning chain
        reasoning = self._build_reasoning_chain(alert, all_factors)
        
        # Generate counter-arguments
        counter_args = self._generate_counter_arguments(alert, perception_data)
        
        # Find similar past cases
        similar = self._find_similar_cases(alert)
        
        # Compute overall confidence
        confidence = self._compute_confidence(all_factors, counter_args)
        
        # Generate summary
        summary = self._generate_summary(alert, all_factors, confidence)
        
        decision = ExplainableDecision(
            decision_id=decision_id,
            timestamp=time.time(),
            alert_raised=True,
            alert_type=alert.get('type', 'unknown'),
            severity=alert.get('severity', 'medium'),
            overall_confidence=confidence,
            factors=all_factors,
            reasoning_steps=reasoning,
            counter_arguments=counter_args,
            summary=summary,
            similar_past_cases=similar
        )
        
        # Store in history
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
        
        return decision
    
    def _extract_perception_factors(self, alert: Dict, 
                                    perception_data: Dict) -> List[DecisionFactor]:
        """Extract factors from perception layer"""
        factors = []
        
        # Detection confidence
        det_conf = perception_data.get('detection_confidence', 0)
        if det_conf:
            factors.append(DecisionFactor(
                factor_type="detection",
                factor_name="object_detection_confidence",
                value=det_conf,
                threshold=0.5,
                contribution=0.2 if det_conf > 0.5 else 0.1,
                explanation=f"Object detected with {det_conf:.0%} confidence"
            ))
        
        # Speed/motion
        speed = perception_data.get('speed', 0)
        if speed > 0:
            threshold = self.thresholds['speed_spike']
            is_high = speed > threshold
            factors.append(DecisionFactor(
                factor_type="motion",
                factor_name="movement_speed",
                value=speed,
                threshold=threshold,
                contribution=0.3 if is_high else 0.1,
                explanation=f"Speed: {speed:.1f} (threshold: {threshold})" + 
                           (" - EXCEEDED" if is_high else " - normal")
            ))
        
        # Stress/pose
        stress = perception_data.get('stress_score', 0)
        if stress > 0:
            threshold = self.thresholds['stress_score']
            is_high = stress > threshold
            factors.append(DecisionFactor(
                factor_type="pose",
                factor_name="stress_indicator",
                value=stress,
                threshold=threshold,
                contribution=0.25 if is_high else 0.05,
                explanation=f"Stress score: {stress:.2f} (threshold: {threshold})" +
                           (" - ELEVATED" if is_high else " - normal")
            ))
        
        # Zone violation
        zone_violation = perception_data.get('zone_violation', False)
        if zone_violation:
            factors.append(DecisionFactor(
                factor_type="zone",
                factor_name="zone_violation",
                value=True,
                threshold=False,
                contribution=0.4,
                explanation=f"Zone violation detected in {perception_data.get('current_zone', 'unknown zone')}"
            ))
        
        return factors
    
    def _extract_cognition_factors(self, alert: Dict,
                                   cognition_data: Dict) -> List[DecisionFactor]:
        """Extract factors from cognition layer"""
        factors = []
        
        # Risk assessment
        risk_score = cognition_data.get('risk_score', 0)
        if risk_score:
            factors.append(DecisionFactor(
                factor_type="cognition",
                factor_name="risk_assessment",
                value=risk_score,
                threshold=0.5,
                contribution=0.3,
                explanation=f"Cognitive risk assessment: {risk_score:.2f}"
            ))
        
        # History comparison
        if cognition_data.get('is_unusual'):
            factors.append(DecisionFactor(
                factor_type="history",
                factor_name="historical_anomaly",
                value=True,
                threshold=False,
                contribution=0.2,
                explanation=cognition_data.get('historical_context', 'Unusual compared to history')
            ))
        
        # Pattern match
        patterns = cognition_data.get('patterns_triggered', [])
        if patterns:
            factors.append(DecisionFactor(
                factor_type="pattern",
                factor_name="pattern_match",
                value=len(patterns),
                threshold=1,
                contribution=0.25,
                explanation=f"Matches known patterns: {', '.join(patterns[:3])}"
            ))
        
        return factors
    
    def _build_reasoning_chain(self, alert: Dict, 
                              factors: List[DecisionFactor]) -> List[str]:
        """Build step-by-step reasoning chain"""
        steps = []
        
        # Step 1: Detection
        det_factors = [f for f in factors if f.factor_type == "detection"]
        if det_factors:
            steps.append(f"1. DETECTION: {det_factors[0].explanation}")
        else:
            steps.append("1. DETECTION: Object/person identified in scene")
        
        # Step 2: Behavior analysis
        motion_factors = [f for f in factors if f.factor_type in ["motion", "pose"]]
        if motion_factors:
            for f in motion_factors:
                steps.append(f"2. BEHAVIOR: {f.explanation}")
        
        # Step 3: Zone context
        zone_factors = [f for f in factors if f.factor_type == "zone"]
        if zone_factors:
            steps.append(f"3. CONTEXT: {zone_factors[0].explanation}")
        else:
            steps.append("3. CONTEXT: No specific zone rules violated")
        
        # Step 4: Cognitive analysis
        cog_factors = [f for f in factors if f.factor_type in ["cognition", "history", "pattern"]]
        if cog_factors:
            for f in cog_factors:
                steps.append(f"4. ANALYSIS: {f.explanation}")
        
        # Step 5: Decision
        total_contribution = sum(f.contribution for f in factors)
        steps.append(f"5. DECISION: Alert raised with severity '{alert.get('severity', 'medium')}' " +
                    f"(total signal strength: {total_contribution:.2f})")
        
        return steps
    
    def _generate_counter_arguments(self, alert: Dict,
                                   perception_data: Dict) -> List[str]:
        """Generate reasons why this might be a false positive"""
        counter_args = []
        
        alert_type = alert.get('type', '')
        
        if 'loitering' in alert_type:
            zone = perception_data.get('zone_type')
            if zone == 'waiting':
                counter_args.append("Person is in a designated waiting area where stationary behavior is expected")
            counter_args.append("Person might be waiting for someone or using their phone")
        
        if 'speed' in alert_type or 'running' in alert_type:
            counter_args.append("Person might be late for an appointment")
            counter_args.append("Could be responding to a call or notification")
        
        if 'abandoned' in alert_type:
            counter_args.append("Owner might have briefly stepped away")
            counter_args.append("Object might belong to nearby seated person")
        
        if 'restricted' in alert_type:
            counter_args.append("Person might be lost or confused")
            counter_args.append("Could be authorized personnel without visible identification")
        
        # Generic
        if perception_data.get('detection_confidence', 1) < 0.7:
            counter_args.append(f"Detection confidence is relatively low ({perception_data.get('detection_confidence', 0):.0%})")
        
        return counter_args
    
    def _find_similar_cases(self, alert: Dict) -> List[str]:
        """Find similar past decisions"""
        similar = []
        alert_type = alert.get('type', '')
        
        for past in reversed(self.decision_history[-20:]):
            if past.alert_type == alert_type:
                time_ago = time.time() - past.timestamp
                if time_ago < 86400:  # Within 24 hours
                    similar.append(f"{past.decision_id} ({time_ago/3600:.1f}h ago)")
        
        return similar[:5]
    
    def _compute_confidence(self, factors: List[DecisionFactor],
                           counter_args: List[str]) -> float:
        """Compute overall decision confidence"""
        if not factors:
            return 0.5
        
        # Base confidence from factors
        weighted_sum = sum(f.contribution for f in factors)
        base_confidence = min(1.0, weighted_sum)
        
        # Reduce for counter-arguments
        counter_penalty = len(counter_args) * 0.05
        
        # Final confidence
        confidence = max(0.1, base_confidence - counter_penalty)
        
        return round(confidence, 2)
    
    def _generate_summary(self, alert: Dict, factors: List[DecisionFactor],
                         confidence: float) -> str:
        """Generate human-readable summary"""
        alert_type = alert.get('type', 'anomaly').replace('_', ' ')
        severity = alert.get('severity', 'medium')
        
        summary = f"Alert raised for {alert_type} with {severity} severity. "
        
        # Top contributing factors
        top_factors = sorted(factors, key=lambda f: f.contribution, reverse=True)[:3]
        if top_factors:
            reasons = [f.factor_name.replace('_', ' ') for f in top_factors]
            summary += f"Primary factors: {', '.join(reasons)}. "
        
        summary += f"Confidence: {confidence:.0%}."
        
        return summary
    
    def format_decision_report(self, decision: ExplainableDecision) -> str:
        """Format decision as human-readable report"""
        lines = []
        
        lines.append("=" * 60)
        lines.append(f"DECISION REPORT: {decision.decision_id}")
        lines.append(f"Time: {datetime.fromtimestamp(decision.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        
        lines.append(f"\n📋 ALERT: {decision.alert_type} ({decision.severity})")
        lines.append(f"📊 CONFIDENCE: {decision.overall_confidence:.0%}")
        
        lines.append("\n🔍 CONTRIBUTING FACTORS:")
        for f in sorted(decision.factors, key=lambda x: x.contribution, reverse=True):
            lines.append(f"  • [{f.contribution:.0%}] {f.explanation}")
        
        lines.append("\n📝 REASONING:")
        for step in decision.reasoning_steps:
            lines.append(f"  {step}")
        
        if decision.counter_arguments:
            lines.append("\n⚠️ POSSIBLE FALSE POSITIVE REASONS:")
            for arg in decision.counter_arguments:
                lines.append(f"  • {arg}")
        
        if decision.similar_past_cases:
            lines.append(f"\n📚 SIMILAR PAST CASES: {', '.join(decision.similar_past_cases)}")
        
        lines.append(f"\n✅ SUMMARY: {decision.summary}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_brief_explanation(self, decision: ExplainableDecision) -> str:
        """Get brief one-line explanation"""
        top_factor = max(decision.factors, key=lambda f: f.contribution) if decision.factors else None
        
        if top_factor:
            return f"{decision.alert_type}: {top_factor.explanation} ({decision.overall_confidence:.0%} confident)"
        else:
            return f"{decision.alert_type}: {decision.summary}"
