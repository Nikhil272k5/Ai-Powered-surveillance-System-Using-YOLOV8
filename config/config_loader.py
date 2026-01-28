"""
Configuration Loader for AbnoGuard Intelligence Platform
Handles loading, validation, and management of configuration settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class SystemConfig:
    """System-level configuration"""
    mode: str = "enhanced"
    log_level: str = "INFO"
    profiles_dir: str = "profiles"
    auto_save_state: bool = True


@dataclass
class NormalityEngineConfig:
    """Normality Engine configuration"""
    enabled: bool = True
    learning_window_minutes: int = 30
    adaptation_rate: float = 0.1
    gmm_components: int = 5
    anomaly_threshold: float = 2.5
    min_observations: int = 100
    features: list = field(default_factory=lambda: [
        "motion_speed", "dwell_time", "crowd_density",
        "object_presence_duration", "spatial_distribution"
    ])


@dataclass
class ConfidenceFusionConfig:
    """Confidence Fusion Engine configuration"""
    enabled: bool = True
    trust_threshold: int = 60
    weights: Dict[str, float] = field(default_factory=lambda: {
        "vision_confidence": 0.25,
        "temporal_persistence": 0.20,
        "motion_stability": 0.15,
        "crowd_context": 0.15,
        "zone_rules": 0.10,
        "historical_accuracy": 0.15
    })
    log_suppressed_alerts: bool = True
    explanation_verbosity: str = "standard"


@dataclass
class BehaviorClassifierConfig:
    """Behavior Classifier configuration"""
    enabled: bool = True
    model_type: str = "random_forest"
    classification_threshold: float = 0.6
    intent_classes: list = field(default_factory=lambda: [
        "normal_transit", "waiting", "loitering", "panic_movement",
        "evasive_behavior", "careless_abandonment", "suspicious_abandonment"
    ])
    history_length: int = 30
    update_interval: int = 5


@dataclass
class CausalReasoningConfig:
    """Causal Reasoning Engine configuration"""
    enabled: bool = True
    lookback_seconds: int = 30
    spatial_proximity: int = 200
    min_chain_length: int = 2
    tracked_events: list = field(default_factory=lambda: [
        "crowd_surge", "exit_blockage", "panic_movement",
        "object_obstruction", "sudden_gathering", "mass_direction_change"
    ])
    explanation_templates: bool = True


@dataclass
class SelfImprovementConfig:
    """Self-Improvement Engine configuration"""
    enabled: bool = True
    learning_rate: float = 0.05
    min_alerts_for_adjustment: int = 20
    performance_window_hours: int = 24
    auto_dismiss_timeout: int = 300
    max_adjustment_per_cycle: float = 0.1
    accept_human_feedback: bool = True
    learn_from_timeouts: bool = True


@dataclass
class AudioAnalyzerConfig:
    """Audio Analyzer configuration"""
    enabled: bool = False
    extraction_method: str = "moviepy"
    sample_rate: int = 22050
    scream_enabled: bool = True
    scream_freq_min: int = 1000
    scream_freq_max: int = 4000
    scream_amplitude_threshold: float = 0.7
    glass_break_enabled: bool = True
    glass_break_freq_min: int = 2000
    glass_break_freq_max: int = 8000
    sudden_noise_enabled: bool = True
    sudden_noise_spike_ratio: float = 3.0


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    websocket_enabled: bool = True
    websocket_update_interval_ms: int = 100
    video_stream_enabled: bool = True
    video_stream_quality: int = 80
    video_stream_max_fps: int = 15
    max_alerts_in_memory: int = 1000
    max_events_in_memory: int = 5000


class ConfigLoader:
    """
    Configuration loader and manager for the Intelligence Platform.
    
    Handles:
    - Loading default configuration
    - Loading user overrides
    - Runtime configuration updates
    - Configuration validation
    """
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure single configuration instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = {}
        self._load_default_config()
    
    def _get_config_path(self, filename: str) -> Path:
        """Get path to configuration file"""
        # Try multiple locations
        locations = [
            Path(__file__).parent / filename,  # config/ directory
            Path(__file__).parent.parent / filename,  # project root
            Path.cwd() / filename,  # current working directory
            Path.cwd() / "config" / filename,  # config/ from cwd
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
        
        # Return default location even if it doesn't exist
        return Path(__file__).parent / filename
    
    def _load_default_config(self) -> None:
        """Load default configuration from YAML file"""
        config_path = self._get_config_path("default_config.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                print(f"✅ Loaded configuration from: {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}")
                self._config = self._get_hardcoded_defaults()
        else:
            print(f"⚠️ Config file not found at {config_path}, using defaults")
            self._config = self._get_hardcoded_defaults()
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Return hardcoded default configuration"""
        return {
            "system": {
                "mode": "enhanced",
                "log_level": "INFO",
                "profiles_dir": "profiles",
                "auto_save_state": True
            },
            "normality_engine": {"enabled": True},
            "confidence_fusion": {"enabled": True, "trust_threshold": 60},
            "behavior_classifier": {"enabled": True},
            "causal_reasoning": {"enabled": True},
            "self_improvement": {"enabled": True},
            "audio_analyzer": {"enabled": False},
            "dashboard": {"enabled": True, "port": 8080}
        }
    
    def load_user_config(self, config_path: Optional[str] = None) -> None:
        """
        Load user-specific configuration overrides.
        
        Args:
            config_path: Path to user config file. If None, looks for
                        'intelligence_config.yaml' in project root.
        """
        if config_path is None:
            config_path = self._get_config_path("intelligence_config.yaml")
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                
                # Deep merge user config into default config
                self._config = self._deep_merge(self._config, user_config)
                print(f"✅ Loaded user configuration from: {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading user config: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Example:
            config.get("normality_engine.enabled")
            config.get("confidence_fusion.weights.vision_confidence")
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value at runtime using dot notation.
        
        Example:
            config.set("confidence_fusion.trust_threshold", 70)
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self._config.get(section, {})
    
    @property
    def system(self) -> SystemConfig:
        """Get system configuration as dataclass"""
        section = self.get_section("system")
        return SystemConfig(**{k: section.get(k, getattr(SystemConfig, k, None)) 
                              for k in SystemConfig.__dataclass_fields__})
    
    @property
    def normality_engine(self) -> NormalityEngineConfig:
        """Get normality engine configuration as dataclass"""
        section = self.get_section("normality_engine")
        return NormalityEngineConfig(
            enabled=section.get("enabled", True),
            learning_window_minutes=section.get("learning_window_minutes", 30),
            adaptation_rate=section.get("adaptation_rate", 0.1),
            gmm_components=section.get("gmm_components", 5),
            anomaly_threshold=section.get("anomaly_threshold", 2.5),
            min_observations=section.get("min_observations", 100),
            features=section.get("features", [])
        )
    
    @property
    def confidence_fusion(self) -> ConfidenceFusionConfig:
        """Get confidence fusion configuration as dataclass"""
        section = self.get_section("confidence_fusion")
        return ConfidenceFusionConfig(
            enabled=section.get("enabled", True),
            trust_threshold=section.get("trust_threshold", 60),
            weights=section.get("weights", {}),
            log_suppressed_alerts=section.get("log_suppressed_alerts", True),
            explanation_verbosity=section.get("explanation_verbosity", "standard")
        )
    
    @property
    def behavior_classifier(self) -> BehaviorClassifierConfig:
        """Get behavior classifier configuration as dataclass"""
        section = self.get_section("behavior_classifier")
        fe = section.get("feature_extraction", {})
        return BehaviorClassifierConfig(
            enabled=section.get("enabled", True),
            model_type=section.get("model_type", "random_forest"),
            classification_threshold=section.get("classification_threshold", 0.6),
            intent_classes=section.get("intent_classes", []),
            history_length=fe.get("history_length", 30),
            update_interval=fe.get("update_interval", 5)
        )
    
    @property
    def causal_reasoning(self) -> CausalReasoningConfig:
        """Get causal reasoning configuration as dataclass"""
        section = self.get_section("causal_reasoning")
        return CausalReasoningConfig(
            enabled=section.get("enabled", True),
            lookback_seconds=section.get("lookback_seconds", 30),
            spatial_proximity=section.get("spatial_proximity", 200),
            min_chain_length=section.get("min_chain_length", 2),
            tracked_events=section.get("tracked_events", []),
            explanation_templates=section.get("explanation_templates", True)
        )
    
    @property
    def self_improvement(self) -> SelfImprovementConfig:
        """Get self-improvement configuration as dataclass"""
        section = self.get_section("self_improvement")
        feedback = section.get("feedback", {})
        return SelfImprovementConfig(
            enabled=section.get("enabled", True),
            learning_rate=section.get("learning_rate", 0.05),
            min_alerts_for_adjustment=section.get("min_alerts_for_adjustment", 20),
            performance_window_hours=section.get("performance_window_hours", 24),
            auto_dismiss_timeout=section.get("auto_dismiss_timeout", 300),
            max_adjustment_per_cycle=section.get("max_adjustment_per_cycle", 0.1),
            accept_human_feedback=feedback.get("accept_human_feedback", True),
            learn_from_timeouts=feedback.get("learn_from_timeouts", True)
        )
    
    @property
    def audio_analyzer(self) -> AudioAnalyzerConfig:
        """Get audio analyzer configuration as dataclass"""
        section = self.get_section("audio_analyzer")
        detection = section.get("detection", {})
        scream = detection.get("scream", {})
        glass = detection.get("glass_break", {})
        noise = detection.get("sudden_noise", {})
        return AudioAnalyzerConfig(
            enabled=section.get("enabled", False),
            extraction_method=section.get("extraction_method", "moviepy"),
            sample_rate=section.get("sample_rate", 22050),
            scream_enabled=scream.get("enabled", True),
            scream_freq_min=scream.get("frequency_min", 1000),
            scream_freq_max=scream.get("frequency_max", 4000),
            scream_amplitude_threshold=scream.get("amplitude_threshold", 0.7),
            glass_break_enabled=glass.get("enabled", True),
            glass_break_freq_min=glass.get("frequency_min", 2000),
            glass_break_freq_max=glass.get("frequency_max", 8000),
            sudden_noise_enabled=noise.get("enabled", True),
            sudden_noise_spike_ratio=noise.get("amplitude_spike_ratio", 3.0)
        )
    
    @property
    def dashboard(self) -> DashboardConfig:
        """Get dashboard configuration as dataclass"""
        section = self.get_section("dashboard")
        video = section.get("video_stream", {})
        return DashboardConfig(
            enabled=section.get("enabled", True),
            host=section.get("host", "127.0.0.1"),
            port=section.get("port", 8080),
            websocket_enabled=section.get("websocket_enabled", True),
            websocket_update_interval_ms=section.get("websocket_update_interval_ms", 100),
            video_stream_enabled=video.get("enabled", True),
            video_stream_quality=video.get("quality", 80),
            video_stream_max_fps=video.get("max_fps", 15),
            max_alerts_in_memory=section.get("max_alerts_in_memory", 1000),
            max_events_in_memory=section.get("max_events_in_memory", 5000)
        )
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a specific intelligence module is enabled"""
        return self.get(f"{module_name}.enabled", False)
    
    def is_enhanced_mode(self) -> bool:
        """Check if system is in enhanced (full intelligence) mode"""
        return self.get("system.mode", "basic") == "enhanced"
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        if path is None:
            path = Path(__file__).parent.parent / "intelligence_config.yaml"
        else:
            path = Path(path)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to: {path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def __repr__(self) -> str:
        return f"ConfigLoader(mode={self.get('system.mode')}, modules_enabled={sum(1 for m in ['normality_engine', 'confidence_fusion', 'behavior_classifier', 'causal_reasoning', 'self_improvement'] if self.is_module_enabled(m))})"


# Global configuration instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get the global configuration instance"""
    return config


def reload_config() -> ConfigLoader:
    """Reload configuration from files"""
    global config
    ConfigLoader._instance = None
    config = ConfigLoader()
    return config
