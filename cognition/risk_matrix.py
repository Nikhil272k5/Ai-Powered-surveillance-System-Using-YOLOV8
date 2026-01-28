"""
AbnoGuard Cognition - Risk Matrix
Dual-axis risk assessment: Intent Confidence × Impact Severity
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class IntentConfidence(Enum):
    """How certain is the AI about the intent"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class ImpactSeverity(Enum):
    """Worst-case consequence assessment"""
    MINIMAL = 0.1       # No real harm
    LOW = 0.3           # Minor disruption
    MEDIUM = 0.5        # Property damage or minor injury possible
    HIGH = 0.7          # Serious injury possible
    CRITICAL = 0.9      # Life-threatening


class RiskCategory(Enum):
    """Resulting risk category from matrix multiplication"""
    MONITOR = "monitor"           # Low risk, keep watching
    INVESTIGATE = "investigate"   # Medium risk, needs attention
    ALERT = "alert"               # High risk, notify security
    URGENT = "urgent"             # Very high risk, immediate action
    EMERGENCY = "emergency"       # Critical risk, emergency protocol


@dataclass
class RiskAssessment:
    """Complete risk assessment for an entity or incident"""
    entity_id: int
    
    # Dual axes
    intent_confidence: float
    intent_level: IntentConfidence
    impact_severity: float
    impact_level: ImpactSeverity
    
    # Fused risk
    risk_score: float          # 0-1
    risk_category: RiskCategory
    
    # Contributing factors
    intent_factors: List[str]
    impact_factors: List[str]
    
    # Recommendations
    recommended_actions: List[str]
    
    # Context
    uncertainty_note: Optional[str]


class RiskMatrix:
    """
    Dual-axis risk intelligence system.
    Mirrors human security risk assessment by evaluating:
    1. Intent Confidence - How sure is the system about malicious intent
    2. Impact Severity - What's the worst that could happen
    
    Risk = f(Intent, Impact)
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Risk matrix thresholds (Intent × Impact → Category)
        self.risk_matrix = {
            # (intent, impact) -> category
            (0.1, 0.1): RiskCategory.MONITOR,
            (0.1, 0.3): RiskCategory.MONITOR,
            (0.1, 0.5): RiskCategory.INVESTIGATE,
            (0.1, 0.7): RiskCategory.INVESTIGATE,
            (0.1, 0.9): RiskCategory.ALERT,
            
            (0.3, 0.1): RiskCategory.MONITOR,
            (0.3, 0.3): RiskCategory.MONITOR,
            (0.3, 0.5): RiskCategory.INVESTIGATE,
            (0.3, 0.7): RiskCategory.ALERT,
            (0.3, 0.9): RiskCategory.ALERT,
            
            (0.5, 0.1): RiskCategory.MONITOR,
            (0.5, 0.3): RiskCategory.INVESTIGATE,
            (0.5, 0.5): RiskCategory.ALERT,
            (0.5, 0.7): RiskCategory.ALERT,
            (0.5, 0.9): RiskCategory.URGENT,
            
            (0.7, 0.1): RiskCategory.INVESTIGATE,
            (0.7, 0.3): RiskCategory.ALERT,
            (0.7, 0.5): RiskCategory.ALERT,
            (0.7, 0.7): RiskCategory.URGENT,
            (0.7, 0.9): RiskCategory.EMERGENCY,
            
            (0.9, 0.1): RiskCategory.INVESTIGATE,
            (0.9, 0.3): RiskCategory.ALERT,
            (0.9, 0.5): RiskCategory.URGENT,
            (0.9, 0.7): RiskCategory.EMERGENCY,
            (0.9, 0.9): RiskCategory.EMERGENCY,
        }
        
        # Intent indicators (behavior -> intent contribution)
        self.intent_indicators = {
            'abandoned_object': 0.7,
            'restricted_zone_entry': 0.6,
            'loitering': 0.4,
            'counterflow': 0.3,
            'running': 0.3,
            'erratic_motion': 0.5,
            'stress_detected': 0.4,
            'crouching': 0.5,
            'fallen': 0.2,  # Could be accident
            'hands_up': 0.6,
            'aggressive_posture': 0.7,
        }
        
        # Impact indicators (context -> impact contribution)
        self.impact_indicators = {
            'crowded_area': 0.6,
            'restricted_zone': 0.7,
            'emergency_zone': 0.8,
            'near_exit': 0.4,
            'near_entry': 0.3,
            'unattended_object': 0.6,
            'multiple_subjects': 0.5,
            'prolonged_duration': 0.4,
            'escalating_behavior': 0.7,
        }
        
        # Action recommendations by category
        self.action_recommendations = {
            RiskCategory.MONITOR: ["Continue monitoring", "Log event"],
            RiskCategory.INVESTIGATE: ["Review footage", "Increase monitoring frequency"],
            RiskCategory.ALERT: ["Notify security personnel", "Prepare for intervention"],
            RiskCategory.URGENT: ["Immediate security response", "Alert on-site personnel"],
            RiskCategory.EMERGENCY: ["Emergency protocol", "Evacuate if needed", "Contact authorities"]
        }
        
        print("📊 Risk Matrix initialized")
    
    def assess_risk(self, entity_data: Dict) -> RiskAssessment:
        """
        Perform dual-axis risk assessment for an entity
        
        Args:
            entity_data: Dict with behavior indicators and context
        
        Returns:
            RiskAssessment with complete analysis
        """
        entity_id = entity_data.get('track_id', 0)
        behaviors = entity_data.get('behaviors', [])
        context = entity_data.get('context', [])
        threat_score = entity_data.get('threat_score', 0.0)
        
        # Calculate intent confidence
        intent_score, intent_factors = self._calculate_intent(behaviors, threat_score)
        intent_level = self._score_to_intent_level(intent_score)
        
        # Calculate impact severity
        impact_score, impact_factors = self._calculate_impact(context, entity_data)
        impact_level = self._score_to_impact_level(impact_score)
        
        # Get risk category from matrix
        risk_category = self._lookup_category(intent_score, impact_score)
        
        # Compute overall risk score
        risk_score = self._compute_risk_score(intent_score, impact_score)
        
        # Get recommendations
        actions = self.action_recommendations.get(risk_category, [])
        
        # Add uncertainty note if confidence is low
        uncertainty = None
        if intent_score < 0.4:
            uncertainty = "Intent confidence is low. Additional observation recommended."
        elif intent_score > 0.7 and impact_score < 0.3:
            uncertainty = "High intent confidence but low impact. May be false positive."
        
        return RiskAssessment(
            entity_id=entity_id,
            intent_confidence=intent_score,
            intent_level=intent_level,
            impact_severity=impact_score,
            impact_level=impact_level,
            risk_score=risk_score,
            risk_category=risk_category,
            intent_factors=intent_factors,
            impact_factors=impact_factors,
            recommended_actions=actions,
            uncertainty_note=uncertainty
        )
    
    def _calculate_intent(self, behaviors: List[str], base_threat: float) -> Tuple[float, List[str]]:
        """Calculate intent confidence from behaviors"""
        factors = []
        scores = [base_threat * 0.5]  # Base contribution
        
        for behavior in behaviors:
            if behavior in self.intent_indicators:
                scores.append(self.intent_indicators[behavior])
                factors.append(f"{behavior}: +{self.intent_indicators[behavior]:.1f}")
        
        # Combine scores (weighted average with diminishing returns)
        if len(scores) == 1:
            final = scores[0]
        else:
            final = np.mean(scores) + 0.1 * (len(scores) - 1)
        
        return min(1.0, final), factors
    
    def _calculate_impact(self, context: List[str], entity_data: Dict) -> Tuple[float, List[str]]:
        """Calculate impact severity from context"""
        factors = []
        scores = [0.2]  # Baseline
        
        for ctx in context:
            if ctx in self.impact_indicators:
                scores.append(self.impact_indicators[ctx])
                factors.append(f"{ctx}: +{self.impact_indicators[ctx]:.1f}")
        
        # Zone-based impact
        zone_type = entity_data.get('zone_type')
        if zone_type == 'restricted':
            scores.append(0.7)
            factors.append("restricted_zone: +0.7")
        elif zone_type == 'emergency':
            scores.append(0.8)
            factors.append("emergency_zone: +0.8")
        
        # Duration factor
        duration = entity_data.get('duration', 0)
        if duration > 120:  # > 2 minutes
            scores.append(0.4)
            factors.append(f"prolonged_duration({duration:.0f}s): +0.4")
        
        final = np.mean(scores) + 0.05 * max(0, len(scores) - 2)
        return min(1.0, final), factors
    
    def _score_to_intent_level(self, score: float) -> IntentConfidence:
        """Convert score to intent level"""
        if score >= 0.8:
            return IntentConfidence.VERY_HIGH
        elif score >= 0.6:
            return IntentConfidence.HIGH
        elif score >= 0.4:
            return IntentConfidence.MEDIUM
        elif score >= 0.2:
            return IntentConfidence.LOW
        else:
            return IntentConfidence.VERY_LOW
    
    def _score_to_impact_level(self, score: float) -> ImpactSeverity:
        """Convert score to impact level"""
        if score >= 0.8:
            return ImpactSeverity.CRITICAL
        elif score >= 0.6:
            return ImpactSeverity.HIGH
        elif score >= 0.4:
            return ImpactSeverity.MEDIUM
        elif score >= 0.2:
            return ImpactSeverity.LOW
        else:
            return ImpactSeverity.MINIMAL
    
    def _lookup_category(self, intent: float, impact: float) -> RiskCategory:
        """Look up risk category from matrix"""
        # Quantize to matrix levels
        intent_q = round(intent * 10) / 10
        impact_q = round(impact * 10) / 10
        
        # Clamp to valid levels
        intent_q = max(0.1, min(0.9, intent_q))
        impact_q = max(0.1, min(0.9, impact_q))
        
        # Round to matrix indices
        intent_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        impact_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        intent_idx = min(range(len(intent_levels)), key=lambda i: abs(intent_levels[i] - intent_q))
        impact_idx = min(range(len(impact_levels)), key=lambda i: abs(impact_levels[i] - impact_q))
        
        key = (intent_levels[intent_idx], impact_levels[impact_idx])
        return self.risk_matrix.get(key, RiskCategory.INVESTIGATE)
    
    def _compute_risk_score(self, intent: float, impact: float) -> float:
        """Compute overall risk score from dual axes"""
        # Geometric mean gives balanced weighting
        return np.sqrt(intent * impact)
    
    def get_risk_matrix_visualization(self) -> List[List[str]]:
        """Get risk matrix as 2D grid for visualization"""
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        matrix = []
        
        for intent in levels:
            row = []
            for impact in levels:
                category = self.risk_matrix.get((intent, impact), RiskCategory.INVESTIGATE)
                row.append(category.value)
            matrix.append(row)
        
        return matrix
    
    def assess_scene(self, entities: List[Dict]) -> Dict:
        """Assess overall scene risk from multiple entities"""
        if not entities:
            return {
                'scene_risk': 0.0,
                'scene_category': RiskCategory.MONITOR,
                'highest_risk_entity': None,
                'aggregate_factors': []
            }
        
        assessments = [self.assess_risk(e) for e in entities]
        
        # Find highest risk
        highest = max(assessments, key=lambda a: a.risk_score)
        
        # Aggregate factors
        all_intent_factors = []
        all_impact_factors = []
        for a in assessments:
            all_intent_factors.extend(a.intent_factors)
            all_impact_factors.extend(a.impact_factors)
        
        # Scene risk (weighted by number of high-risk entities)
        risk_scores = [a.risk_score for a in assessments]
        high_risk_count = sum(1 for s in risk_scores if s > 0.5)
        
        scene_risk = np.mean(risk_scores) + 0.1 * high_risk_count
        scene_risk = min(1.0, scene_risk)
        
        scene_category = self._lookup_category(scene_risk, scene_risk)
        
        return {
            'scene_risk': scene_risk,
            'scene_category': scene_category,
            'highest_risk_entity': highest.entity_id,
            'entity_count': len(entities),
            'high_risk_count': high_risk_count,
            'aggregate_factors': list(set(all_intent_factors + all_impact_factors))[:10]
        }
