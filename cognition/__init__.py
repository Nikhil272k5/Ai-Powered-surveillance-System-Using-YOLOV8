"""
AbnoGuard Cognition Package - Brain-2
High-level cognitive reasoning for understanding reality
"""

from cognition.narrative_engine import NarrativeEngine, NarrativeStyle, Incident
from cognition.temporal_memory import TemporalMemory, MemoryEntry, Pattern, PatternType
from cognition.risk_matrix import RiskMatrix, RiskAssessment, RiskCategory, IntentConfidence, ImpactSeverity
from cognition.ethics_layer import EthicsLayer, EthicalAssessment, AlertModification
from cognition.explainability import ExplainabilityEngine, ExplainableDecision, DecisionFactor
from cognition.brief_generator import BriefGenerator, IncidentBrief

__all__ = [
    'NarrativeEngine', 'NarrativeStyle', 'Incident',
    'TemporalMemory', 'MemoryEntry', 'Pattern', 'PatternType',
    'RiskMatrix', 'RiskAssessment', 'RiskCategory', 'IntentConfidence', 'ImpactSeverity',
    'EthicsLayer', 'EthicalAssessment', 'AlertModification',
    'ExplainabilityEngine', 'ExplainableDecision', 'DecisionFactor',
    'BriefGenerator', 'IncidentBrief'
]
