"""
Intelligence Package - Core intelligence modules for AbnoGuard
5-Layer Security Architecture with Threat Intelligence
"""

# Import new Threat Intelligence modules first (they have no problematic dependencies)
from .event_bus import EventBus, SecurityEvent, EventType as SecurityEventType, event_bus
from .threat_engine import ThreatIntelligenceEngine, ThreatConclusion, ThreatSeverity, ThreatType, threat_engine
from .threat_profiler import ThreatProfiler, ThreatProfile, RiskLevel, threat_profiler
from .database import ThreatDatabase, threat_db

# Try to import original modules (may have optional dependencies)
try:
    from .normality_engine import NormalityEngine
except ImportError:
    NormalityEngine = None

try:
    from .confidence_fusion import ConfidenceFusionEngine
except ImportError:
    ConfidenceFusionEngine = None

try:
    from .behavior_classifier import BehaviorClassifier, IntentClass
except ImportError:
    BehaviorClassifier = None
    IntentClass = None

try:
    from .causal_reasoning import CausalReasoningEngine, EventType
except ImportError:
    CausalReasoningEngine = None
    EventType = None

try:
    from .self_improvement import SelfImprovementEngine
except ImportError:
    SelfImprovementEngine = None

try:
    from .audio_analyzer import AudioAnalyzer
except Exception:
    AudioAnalyzer = None

# Initialize SOC defaults
try:
    from . import init_soc
except Exception:
    pass

__all__ = [
    # Threat Intelligence (always available)
    'EventBus', 'SecurityEvent', 'SecurityEventType', 'event_bus',
    'ThreatIntelligenceEngine', 'ThreatConclusion', 'ThreatSeverity', 'ThreatType', 'threat_engine',
    'ThreatProfiler', 'ThreatProfile', 'RiskLevel', 'threat_profiler',
    'ThreatDatabase', 'threat_db',
    
    # Original (may be None if dependencies missing)
    'NormalityEngine',
    'ConfidenceFusionEngine', 
    'BehaviorClassifier',
    'IntentClass',
    'CausalReasoningEngine',
    'EventType',
    'SelfImprovementEngine',
    'AudioAnalyzer'
]
