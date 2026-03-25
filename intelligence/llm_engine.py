"""
LLM Engine for Threat Intelligence

Provides LLM-powered semantic reasoning for:
- Threat explanations
- Entity profiling
- Adversarial behavior analysis
- Incident briefings
- Response suggestions

Supports:
- OpenAI API (GPT-4o-mini, GPT-3.5-turbo)
- Ollama local (Mistral, LLaMA)
- Graceful fallback if no LLM available
"""

import os
import json
import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Try to import LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM Engine"""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    cache_enabled: bool = True
    max_cache_size: int = 100
    timeout_seconds: int = 30


@dataclass
class LLMResponse:
    """Response from LLM"""
    text: str
    model: str
    cached: bool = False
    latency_ms: int = 0
    tokens_used: int = 0


class LLMEngine:
    """
    LLM Engine for semantic threat intelligence.
    
    Primary: OpenAI API (if key available)
    Fallback: Ollama local (if running)
    Final fallback: Rule-based templates
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._cache: Dict[str, LLMResponse] = {}
        self._openai_client = None
        self._backend = "none"
        
        # Try to get API key from environment
        if not self.config.openai_api_key:
            self.config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Initialize backends
        self._init_backends()
    
    def _init_backends(self):
        """Initialize available LLM backends"""
        # Try OpenAI first
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            try:
                self._openai_client = openai.OpenAI(
                    api_key=self.config.openai_api_key
                )
                self._backend = "openai"
                print(f"[LLM] OpenAI backend initialized ({self.config.openai_model})")
                return
            except Exception as e:
                print(f"[LLM] OpenAI init failed: {e}")
        
        # Try Ollama
        if HTTPX_AVAILABLE:
            try:
                import httpx
                response = httpx.get(f"{self.config.ollama_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    self._backend = "ollama"
                    print(f"[LLM] Ollama backend initialized ({self.config.ollama_model})")
                    return
            except:
                pass
        
        # No LLM available - use templates
        self._backend = "template"
        print("[LLM] No LLM backend available - using template fallback")
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM backend is available"""
        return self._backend in ["openai", "ollama"]
    
    @property
    def backend_name(self) -> str:
        """Get current backend name"""
        return self._backend
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    async def _call_openai(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Call OpenAI API"""
        start = datetime.now()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            latency = int((datetime.now() - start).total_seconds() * 1000)
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.config.openai_model,
                latency_ms=latency,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            print(f"[LLM] OpenAI error: {e}")
            return None
    
    async def _call_ollama(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Call Ollama API"""
        start = datetime.now()
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.ollama_model,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=self.config.timeout_seconds
                )
                
                if response.status_code == 200:
                    data = response.json()
                    latency = int((datetime.now() - start).total_seconds() * 1000)
                    
                    return LLMResponse(
                        text=data.get("response", ""),
                        model=self.config.ollama_model,
                        latency_ms=latency
                    )
        except Exception as e:
            print(f"[LLM] Ollama error: {e}")
        
        return None
    
    async def _generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Generate response using available backend"""
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._get_cache_key(f"{system_prompt}|{prompt}")
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.cached = True
                return cached
        
        response = None
        
        # Try OpenAI
        if self._backend == "openai" and self._openai_client:
            response = await self._call_openai(prompt, system_prompt)
        
        # Try Ollama
        if response is None and self._backend == "ollama":
            response = await self._call_ollama(prompt, system_prompt)
        
        # Cache response
        if response and self.config.cache_enabled:
            if len(self._cache) >= self.config.max_cache_size:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = response
        
        return response
    
    # ==================== THREAT INTELLIGENCE ====================
    
    async def generate_threat_explanation(
        self, 
        threat_type: str,
        signals: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> str:
        """Generate LLM-powered threat explanation"""
        
        system_prompt = """You are a security analyst AI. Generate a concise, professional threat assessment explanation.
Focus on: WHY this is a threat, WHAT evidence supports it, and WHAT the risk level implies.
Be specific but brief (2-3 sentences max). Use security terminology appropriately."""
        
        # Build context from signals
        signal_summary = []
        for s in signals[:5]:  # Limit to 5 signals
            signal_summary.append(f"- {s.get('signal_type', 'unknown')}: confidence {s.get('confidence', 0):.0%}")
        
        zone_info = context.get('zone_ids', ['unknown']) if context else ['unknown']
        
        prompt = f"""Analyze this security threat:

Threat Type: {threat_type}
Detected Signals:
{chr(10).join(signal_summary)}
Location: {', '.join(zone_info)}
Time: {datetime.now().strftime('%H:%M')}

Generate a professional threat explanation."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            return response.text.strip()
        
        # Template fallback
        return self._template_threat_explanation(threat_type, signals)
    
    def _template_threat_explanation(self, threat_type: str, signals: List[Dict]) -> str:
        """Fallback template-based explanation"""
        signal_types = [s.get('signal_type', 'unknown') for s in signals[:3]]
        return f"This {threat_type} threat was identified based on {len(signals)} correlated signals including {', '.join(signal_types)}. The pattern suggests potential security concern requiring attention."
    
    async def generate_entity_profile(
        self,
        entity_id: str,
        behavior_history: List[Dict[str, Any]],
        risk_level: str
    ) -> str:
        """Generate LLM-powered entity profile summary"""
        
        system_prompt = """You are a behavioral analyst AI. Generate a brief profile summary for a tracked entity.
Focus on: behavioral patterns, risk indicators, and trajectory over time.
Be concise (2-3 sentences). Use professional security language."""
        
        # Summarize history
        history_points = []
        for h in behavior_history[-5:]:  # Last 5 behaviors
            history_points.append(f"- {h.get('behavior', 'unknown')} at {h.get('time', 'unknown')}")
        
        prompt = f"""Generate a security profile summary:

Entity ID: {entity_id}
Current Risk Level: {risk_level}
Recent Behavior History:
{chr(10).join(history_points) if history_points else '- No significant history'}

Provide a behavioral profile assessment."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            return response.text.strip()
        
        return f"Entity {entity_id[:8]} has {risk_level} risk level with {len(behavior_history)} recorded behaviors."
    
    async def analyze_adversarial_behavior(
        self,
        signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze if behavior is adversarial/evasive"""
        
        system_prompt = """You are a security behavior analyst. Determine if observed behavior indicates adversarial intent.
Consider: camera avoidance, evasive movement, blind-spot usage, suspicious timing.
Respond in JSON format: {"is_adversarial": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
        
        signal_desc = [f"- {s.get('signal_type')}: {s.get('details', '')}" for s in signals[:5]]
        
        prompt = f"""Analyze these behavioral signals for adversarial intent:

{chr(10).join(signal_desc)}

Determine if this is adversarial behavior."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            try:
                # Try to parse JSON
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1].replace("json", "").strip()
                return json.loads(text)
            except:
                pass
        
        # Fallback
        return {
            "is_adversarial": False,
            "confidence": 0.5,
            "reasoning": "Unable to determine intent from available signals."
        }
    
    async def analyze_group_coordination(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze if group movement is coordinated"""
        
        system_prompt = """You are a crowd behavior analyst. Determine if observed group movement indicates coordination.
Consider: synchronized timing, convergence patterns, communication indicators, unnatural grouping.
Respond in JSON: {"is_coordinated": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
        
        traj_desc = [f"- Entity {t.get('entity_id', '?')}: {t.get('pattern', 'unknown')}" for t in trajectories[:5]]
        
        prompt = f"""Analyze these movement patterns for coordination:

{chr(10).join(traj_desc)}

Determine if this suggests coordinated action."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            try:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1].replace("json", "").strip()
                return json.loads(text)
            except:
                pass
        
        return {
            "is_coordinated": False,
            "confidence": 0.5,
            "reasoning": "Insufficient data to determine coordination."
        }
    
    async def generate_incident_briefing(
        self,
        incident_data: Dict[str, Any]
    ) -> str:
        """Generate LLM-powered incident briefing"""
        
        system_prompt = """You are a security operations analyst. Generate a professional incident briefing.
Include: incident summary, key events, current status, and recommended actions.
Format as a structured brief (3-5 bullet points). Be concise but comprehensive."""
        
        prompt = f"""Generate an incident briefing:

Incident ID: {incident_data.get('incident_id', 'Unknown')}
Title: {incident_data.get('title', 'Security Incident')}
Severity: {incident_data.get('severity', 'medium')}
Status: {incident_data.get('status', 'active')}
Description: {incident_data.get('description', 'No description')}
Created: {incident_data.get('created_at', 'Unknown')}

Provide a professional incident briefing."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            return response.text.strip()
        
        return f"Incident {incident_data.get('incident_id', 'Unknown')}: {incident_data.get('title', 'Security event')} - {incident_data.get('severity', 'medium')} severity, currently {incident_data.get('status', 'active')}."
    
    async def suggest_response_actions(
        self,
        threat_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest LLM-powered response actions"""
        
        system_prompt = """You are a security response advisor. Suggest appropriate response actions for a threat.
Consider: threat severity, available options, proportionality, human-in-the-loop requirements.
Respond in JSON array: [{"action": "name", "priority": 1-3, "reasoning": "why"}]"""
        
        prompt = f"""Suggest response actions for this threat:

Threat Type: {threat_data.get('threat_type', 'unknown')}
Severity: {threat_data.get('severity', 'medium')}
Confidence: {threat_data.get('confidence', 0.5):.0%}
Location: {threat_data.get('zone_ids', ['unknown'])}

Provide ranked response recommendations."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            try:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1].replace("json", "").strip()
                return json.loads(text)
            except:
                pass
        
        # Default actions
        return [
            {"action": "Increase monitoring", "priority": 1, "reasoning": "Standard first response"},
            {"action": "Alert security personnel", "priority": 2, "reasoning": "Human verification needed"},
            {"action": "Review camera footage", "priority": 3, "reasoning": "Gather more evidence"}
        ]
    
    async def analyze_false_alarm(
        self,
        alert_data: Dict[str, Any],
        user_feedback: str = ""
    ) -> Dict[str, Any]:
        """Analyze why an alert was a false alarm"""
        
        system_prompt = """You are a security system analyst. Analyze why an alert was a false alarm.
Identify: root cause, which signals were misleading, and how to prevent similar false alarms.
Respond in JSON: {"root_cause": "explanation", "misleading_signals": ["list"], "recommendations": ["list"]}"""
        
        prompt = f"""Analyze this false alarm:

Alert Type: {alert_data.get('alert_type', 'unknown')}
Original Signals: {alert_data.get('signals', [])}
User Feedback: {user_feedback or 'No feedback provided'}

Explain why this was a false alarm."""
        
        response = await self._generate(prompt, system_prompt)
        
        if response and response.text:
            try:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1].replace("json", "").strip()
                return json.loads(text)
            except:
                pass
        
        return {
            "root_cause": "Unable to determine root cause",
            "misleading_signals": [],
            "recommendations": ["Review detection thresholds", "Add contextual rules"]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get LLM engine status"""
        return {
            "available": self.is_available,
            "backend": self._backend,
            "model": self.config.openai_model if self._backend == "openai" else self.config.ollama_model,
            "cache_size": len(self._cache),
            "cache_enabled": self.config.cache_enabled
        }


# Global LLM engine instance
llm_engine = LLMEngine()
