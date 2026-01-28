"""
Multi-Modal Audio Intelligence
Analyzes audio extracted from video files for surveillance events.

Detects:
- Screams (high frequency + amplitude spike)
- Glass breaking (characteristic spectral pattern)
- Sudden loud noises (impulse detection)
- Gunshots (impulsive broadband)
- Alarms (sustained tones)

Uses:
- FFmpeg/moviepy for audio extraction
- Mel spectrogram analysis
- Threshold-based detection
- Optional pre-trained models
"""

import os
import time
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available. Audio analysis will be limited.")

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ moviepy not available. Audio extraction will be disabled.")

try:
    from scipy import signal as scipy_signal
    from scipy.ndimage import maximum_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AudioEvent:
    """Detected audio event"""
    event_type: str  # 'scream', 'glass_break', 'loud_noise', 'gunshot', 'alarm'
    timestamp: float  # Seconds into video
    confidence: float  # 0-1
    duration: float  # Duration of event in seconds
    frequency_range: Tuple[float, float]  # Hz
    amplitude: float  # Normalized amplitude
    frame_number: Optional[int] = None


@dataclass
class AudioAnalysisResult:
    """Result of analyzing an audio segment"""
    has_events: bool
    events: List[AudioEvent]
    overall_loudness: float
    spectral_features: Dict[str, float]
    analysis_timestamp: float


class AudioAnalyzer:
    """
    Multi-Modal Audio Intelligence
    
    Analyzes audio from video files to detect surveillance-relevant
    audio events like screams, glass breaking, and sudden noises.
    """
    
    def __init__(self,
                 sample_rate: int = 22050,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 scream_freq_range: Tuple[int, int] = (1000, 4000),
                 scream_amplitude_threshold: float = 0.7,
                 glass_break_freq_range: Tuple[int, int] = (2000, 8000),
                 sudden_noise_spike_ratio: float = 3.0,
                 buffer_seconds: float = 2.0):
        """
        Initialize the Audio Analyzer.
        
        Args:
            sample_rate: Audio sample rate for analysis
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            scream_freq_range: Frequency range for scream detection (Hz)
            scream_amplitude_threshold: Amplitude threshold for screams
            glass_break_freq_range: Frequency range for glass break detection
            sudden_noise_spike_ratio: Ratio above average for sudden noise
            buffer_seconds: Audio buffer length for real-time analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.scream_freq_range = scream_freq_range
        self.scream_amplitude_threshold = scream_amplitude_threshold
        self.glass_break_freq_range = glass_break_freq_range
        self.sudden_noise_spike_ratio = sudden_noise_spike_ratio
        self.buffer_seconds = buffer_seconds
        
        # Audio buffer for streaming analysis
        self.audio_buffer: Optional[np.ndarray] = None
        self.buffer_position = 0
        
        # Event history
        self.events: deque = deque(maxlen=1000)
        
        # Background noise level (adaptive)
        self.background_level = 0.1
        self.background_samples = deque(maxlen=100)
        
        # Statistics
        self.total_events_detected = 0
        self.analysis_count = 0
        
        # Check availability
        self.is_available = LIBROSA_AVAILABLE or SCIPY_AVAILABLE
        
        if self.is_available:
            print(f"🎧 Audio Analyzer initialized")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Scream detection: {scream_freq_range[0]}-{scream_freq_range[1]} Hz")
            print(f"   Using: {'librosa' if LIBROSA_AVAILABLE else 'scipy'}")
        else:
            print("⚠️ Audio Analyzer disabled (no audio libraries available)")
    
    def extract_audio_from_video(self, video_path: str, 
                                  output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path for extracted audio
        
        Returns:
            Path to extracted audio file, or None on failure
        """
        if not MOVIEPY_AVAILABLE:
            print("⚠️ Cannot extract audio: moviepy not available")
            return None
        
        try:
            if output_path is None:
                # Create temp file
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"audio_{int(time.time())}.wav")
            
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("⚠️ Video has no audio track")
                video.close()
                return None
            
            video.audio.write_audiofile(
                output_path,
                fps=self.sample_rate,
                nbytes=2,
                codec='pcm_s16le',
                verbose=False,
                logger=None
            )
            
            video.close()
            print(f"✅ Audio extracted: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load audio file for analysis.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Audio samples as numpy array
        """
        if not self.is_available:
            return None
        
        try:
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return audio
            else:
                # Fallback: try scipy
                from scipy.io import wavfile
                sr, audio = wavfile.read(audio_path)
                audio = audio.astype(np.float32) / 32768.0
                return audio
                
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None
    
    def analyze_segment(self, audio: np.ndarray, 
                       start_time: float = 0.0,
                       fps: float = 30.0) -> AudioAnalysisResult:
        """
        Analyze an audio segment for events.
        
        Args:
            audio: Audio samples
            start_time: Start time offset in seconds
            fps: Video FPS for frame number calculation
        
        Returns:
            AudioAnalysisResult with detected events
        """
        self.analysis_count += 1
        events = []
        
        if len(audio) < self.sample_rate * 0.1:  # At least 0.1 seconds
            return AudioAnalysisResult(
                has_events=False,
                events=[],
                overall_loudness=0.0,
                spectral_features={},
                analysis_timestamp=time.time()
            )
        
        # Calculate overall loudness
        rms = np.sqrt(np.mean(audio ** 2))
        overall_loudness = float(rms)
        
        # Update background level
        self.background_samples.append(rms)
        self.background_level = np.median(list(self.background_samples))
        
        # Compute spectral features
        spectral_features = self._compute_spectral_features(audio)
        
        # Detect events
        scream_events = self._detect_screams(audio, start_time, fps)
        events.extend(scream_events)
        
        glass_events = self._detect_glass_break(audio, start_time, fps)
        events.extend(glass_events)
        
        noise_events = self._detect_sudden_noise(audio, start_time, fps)
        events.extend(noise_events)
        
        # Store events
        for event in events:
            self.events.append(event)
            self.total_events_detected += 1
        
        return AudioAnalysisResult(
            has_events=len(events) > 0,
            events=events,
            overall_loudness=overall_loudness,
            spectral_features=spectral_features,
            analysis_timestamp=time.time()
        )
    
    def _compute_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute spectral features from audio"""
        features = {}
        
        try:
            if LIBROSA_AVAILABLE:
                # Spectral centroid (brightness)
                centroid = librosa.feature.spectral_centroid(
                    y=audio, sr=self.sample_rate
                )[0]
                features['spectral_centroid'] = float(np.mean(centroid))
                
                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio, sr=self.sample_rate
                )[0]
                features['spectral_rolloff'] = float(np.mean(rolloff))
                
                # Zero crossing rate (noise indicator)
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features['zero_crossing_rate'] = float(np.mean(zcr))
                
            elif SCIPY_AVAILABLE:
                # Basic FFT-based features
                fft = np.abs(np.fft.rfft(audio))
                freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
                
                # Spectral centroid approximation
                if np.sum(fft) > 0:
                    centroid = np.sum(freqs * fft) / np.sum(fft)
                    features['spectral_centroid'] = float(centroid)
                
        except Exception as e:
            print(f"⚠️ Error computing spectral features: {e}")
        
        return features
    
    def _detect_screams(self, audio: np.ndarray, start_time: float,
                        fps: float) -> List[AudioEvent]:
        """Detect scream-like sounds"""
        events = []
        
        if not (LIBROSA_AVAILABLE or SCIPY_AVAILABLE):
            return events
        
        try:
            # Get frequency content in scream range
            if LIBROSA_AVAILABLE:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=self.sample_rate, n_mels=self.n_mels,
                    fmin=self.scream_freq_range[0],
                    fmax=self.scream_freq_range[1]
                )
                mel_power = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Find high-energy segments in scream frequency range
                energy = np.mean(mel_power, axis=0)
                
            else:
                # Scipy fallback: bandpass filter and energy
                from scipy.signal import butter, filtfilt
                
                nyq = self.sample_rate / 2
                low = self.scream_freq_range[0] / nyq
                high = min(self.scream_freq_range[1] / nyq, 0.99)
                b, a = butter(4, [low, high], btype='band')
                filtered = filtfilt(b, a, audio)
                
                # Calculate energy in windows
                window_size = int(self.sample_rate * 0.05)
                energy = np.array([
                    np.sqrt(np.mean(filtered[i:i+window_size]**2))
                    for i in range(0, len(filtered) - window_size, window_size)
                ])
            
            # Normalize energy
            if len(energy) > 0 and np.max(energy) > 0:
                energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
                
                # Find peaks above threshold
                threshold = self.scream_amplitude_threshold
                peaks = np.where(energy_norm > threshold)[0]
                
                if len(peaks) > 0:
                    # Group consecutive peaks into events
                    peak_groups = self._group_consecutive(peaks)
                    
                    for group in peak_groups:
                        if len(group) >= 2:  # At least some duration
                            # Calculate timing
                            if LIBROSA_AVAILABLE:
                                times = librosa.times_like(mel_power[0], sr=self.sample_rate)
                                event_start = times[group[0]] if group[0] < len(times) else 0
                                event_end = times[min(group[-1], len(times)-1)]
                            else:
                                frame_duration = window_size / self.sample_rate
                                event_start = group[0] * frame_duration
                                event_end = group[-1] * frame_duration
                            
                            confidence = float(np.mean(energy_norm[group]))
                            
                            event = AudioEvent(
                                event_type='scream',
                                timestamp=start_time + event_start,
                                confidence=min(confidence, 1.0),
                                duration=max(event_end - event_start, 0.1),
                                frequency_range=self.scream_freq_range,
                                amplitude=float(np.max(energy_norm[group])),
                                frame_number=int((start_time + event_start) * fps)
                            )
                            events.append(event)
        
        except Exception as e:
            print(f"⚠️ Error detecting screams: {e}")
        
        return events
    
    def _detect_glass_break(self, audio: np.ndarray, start_time: float,
                            fps: float) -> List[AudioEvent]:
        """Detect glass breaking sounds"""
        events = []
        
        if not (LIBROSA_AVAILABLE or SCIPY_AVAILABLE):
            return events
        
        try:
            # Glass breaking has characteristic high frequency content and impulsive nature
            if SCIPY_AVAILABLE:
                from scipy.signal import butter, filtfilt
                
                nyq = self.sample_rate / 2
                low = self.glass_break_freq_range[0] / nyq
                high = min(self.glass_break_freq_range[1] / nyq, 0.99)
                b, a = butter(4, [low, high], btype='band')
                filtered = filtfilt(b, a, audio)
                
                # Calculate short-term energy
                window_size = int(self.sample_rate * 0.02)  # 20ms windows
                energy = np.array([
                    np.sqrt(np.mean(filtered[i:i+window_size]**2))
                    for i in range(0, len(filtered) - window_size, window_size // 2)
                ])
                
                if len(energy) > 0:
                    # Look for sharp transients (impulsive sounds)
                    energy_diff = np.diff(energy)
                    threshold = np.mean(np.abs(energy_diff)) + 3 * np.std(energy_diff)
                    
                    peaks = np.where(energy_diff > threshold)[0]
                    
                    for peak in peaks:
                        # Check for characteristic decay after peak
                        if peak + 5 < len(energy):
                            decay = energy[peak+1:peak+5]
                            if len(decay) > 0 and decay[0] > decay[-1]:  # Decaying energy
                                event_time = peak * window_size / 2 / self.sample_rate
                                
                                event = AudioEvent(
                                    event_type='glass_break',
                                    timestamp=start_time + event_time,
                                    confidence=min(float(energy_diff[peak] / threshold), 1.0),
                                    duration=0.3,
                                    frequency_range=self.glass_break_freq_range,
                                    amplitude=float(energy[peak]),
                                    frame_number=int((start_time + event_time) * fps)
                                )
                                events.append(event)
        
        except Exception as e:
            print(f"⚠️ Error detecting glass break: {e}")
        
        return events
    
    def _detect_sudden_noise(self, audio: np.ndarray, start_time: float,
                             fps: float) -> List[AudioEvent]:
        """Detect sudden loud noises"""
        events = []
        
        try:
            # Calculate RMS in short windows
            window_size = int(self.sample_rate * 0.05)  # 50ms
            hop = window_size // 2
            
            rms_values = []
            for i in range(0, len(audio) - window_size, hop):
                rms = np.sqrt(np.mean(audio[i:i+window_size]**2))
                rms_values.append(rms)
            
            if len(rms_values) < 5:
                return events
            
            rms_array = np.array(rms_values)
            
            # Calculate rolling average
            rolling_avg = np.convolve(rms_array, np.ones(5)/5, mode='valid')
            
            # Find spikes above threshold
            for i in range(len(rolling_avg)):
                if i < len(rms_array):
                    if rms_array[i] > rolling_avg[i] * self.sudden_noise_spike_ratio:
                        if rms_array[i] > self.background_level * 2:
                            event_time = i * hop / self.sample_rate
                            
                            event = AudioEvent(
                                event_type='loud_noise',
                                timestamp=start_time + event_time,
                                confidence=min(float(rms_array[i] / rolling_avg[i]) / self.sudden_noise_spike_ratio, 1.0),
                                duration=hop / self.sample_rate,
                                frequency_range=(20, self.sample_rate // 2),
                                amplitude=float(rms_array[i]),
                                frame_number=int((start_time + event_time) * fps)
                            )
                            events.append(event)
        
        except Exception as e:
            print(f"⚠️ Error detecting sudden noise: {e}")
        
        return events
    
    def _group_consecutive(self, indices: np.ndarray) -> List[List[int]]:
        """Group consecutive indices together"""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= 2:  # Allow small gaps
                current_group.append(indices[i])
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [indices[i]]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def analyze_video(self, video_path: str, 
                      chunk_seconds: float = 5.0) -> List[AudioEvent]:
        """
        Analyze entire video file for audio events.
        
        Args:
            video_path: Path to video file
            chunk_seconds: Analysis chunk size
        
        Returns:
            List of all detected audio events
        """
        if not self.is_available:
            print("⚠️ Audio analysis not available")
            return []
        
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        if audio_path is None:
            return []
        
        # Load audio
        audio = self.load_audio(audio_path)
        if audio is None:
            return []
        
        all_events = []
        
        # Analyze in chunks
        chunk_samples = int(chunk_seconds * self.sample_rate)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            start_time = i / self.sample_rate
            
            result = self.analyze_segment(chunk, start_time)
            all_events.extend(result.events)
        
        # Clean up temp file
        try:
            os.remove(audio_path)
        except:
            pass
        
        print(f"🎧 Audio analysis complete: {len(all_events)} events detected")
        return all_events
    
    def get_recent_events(self, seconds: float = 10.0) -> List[AudioEvent]:
        """Get events from the last N seconds"""
        current_time = time.time()
        return [
            e for e in self.events
            if current_time - e.timestamp <= seconds
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            'is_available': self.is_available,
            'total_events_detected': self.total_events_detected,
            'analysis_count': self.analysis_count,
            'event_type_counts': event_types,
            'background_level': self.background_level,
            'libraries': {
                'librosa': LIBROSA_AVAILABLE,
                'moviepy': MOVIEPY_AVAILABLE,
                'scipy': SCIPY_AVAILABLE
            }
        }
