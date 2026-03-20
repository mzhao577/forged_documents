"""
AI Detection Module
Multiple algorithms for detecting AI-generated text.

Supported detectors:
- GPTZero (API-based)
- Originality.ai (API-based)
- Copyleaks (API-based)
- ZeroGPT (API-based)
- Hugging Face RoBERTa / OpenAI Detector (local model)
- Binoculars (local model - perplexity comparison)
- Fast-DetectGPT (local model - curvature-based)
- LLMDet (local model - proxy perplexity with LLM source identification)
- ROUGE Similarity Checker (local - text comparison)
- Ensemble (combines multiple detectors)
"""

import os
import re
import csv
import math
import argparse
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class DetectorType(Enum):
    """Available AI detector types."""
    GPTZERO = "gptzero"
    ORIGINALITY = "originality"
    COPYLEAKS = "copyleaks"
    ZEROGPT = "zerogpt"
    HUGGINGFACE_ROBERTA = "huggingface_roberta"
    OPENAI_DETECTOR = "openai_detector"
    BINOCULARS = "binoculars"
    FAST_DETECTGPT = "fast_detectgpt"
    LLMDET = "llmdet"
    ROUGE_CHECKER = "rouge_checker"
    DESKLIB = "desklib"
    ENSEMBLE = "ensemble"


@dataclass
class AITextAnalysis:
    """Detailed analysis of AI-classified text."""
    has_duplicates: bool = False
    duplicate_phrases: List[str] = field(default_factory=list)
    duplicate_ratio: float = 0.0
    high_ai_segments: List[Dict] = field(default_factory=list)  # segments with high AI probability
    primary_reason: str = ""  # main reason for classification
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class AIDetectionResult:
    """Results from AI detection analysis."""
    detector_name: str
    is_ai_generated: bool
    ai_probability: float
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None
    analysis: Optional[AITextAnalysis] = None  # detailed analysis for AI-classified texts


@dataclass
class EnsembleResult:
    """Results from ensemble of multiple detectors."""
    individual_results: List[AIDetectionResult] = field(default_factory=list)
    consensus_is_ai: bool = False
    average_probability: float = 0.0
    weighted_probability: float = 0.0
    agreement_score: float = 0.0
    detectors_used: int = 0
    detectors_succeeded: int = 0


class BaseAIDetector(ABC):
    """Abstract base class for AI detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return detector name."""
        pass

    @abstractmethod
    def detect(self, text: str) -> AIDetectionResult:
        """Analyze text for AI-generated content."""
        pass

    def _validate_text(self, text: str, min_length: int = 50) -> Optional[str]:
        """Validate text meets minimum requirements."""
        if not text or len(text.strip()) < min_length:
            return f"Text too short (minimum {min_length} characters)"
        return None


# =============================================================================
# API-Based Detectors
# =============================================================================

class GPTZeroDetector(BaseAIDetector):
    """GPTZero API-based detector."""

    API_URL = "https://api.gptzero.me/v2/predict/text"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }

    @property
    def name(self) -> str:
        return "GPTZero"

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json={"document": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            documents = data.get("documents", [{}])
            if documents:
                doc = documents[0]
                ai_prob = doc.get("completely_generated_prob", 0.0)

                return AIDetectionResult(
                    detector_name=self.name,
                    is_ai_generated=ai_prob > 0.5,
                    ai_probability=ai_prob,
                    confidence=abs(ai_prob - 0.5) * 2,
                    details={
                        "average_generated_prob": doc.get("average_generated_prob"),
                        "burstiness": doc.get("burstiness"),
                        "sentences": doc.get("sentences", [])
                    }
                )

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error="No analysis data returned"
            )

        except requests.exceptions.RequestException as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"API request failed: {str(e)}"
            )


class OriginalityDetector(BaseAIDetector):
    """Originality.ai API-based detector."""

    API_URL = "https://api.originality.ai/api/v1/scan/ai"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "Originality.ai"

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "X-OAI-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={"content": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            ai_score = data.get("score", {}).get("ai", 0.0)

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_score > 0.5,
                ai_probability=ai_score,
                confidence=abs(ai_score - 0.5) * 2,
                details={
                    "original_score": data.get("score", {}).get("original", 0.0),
                    "credits_used": data.get("credits_used", 0)
                }
            )

        except requests.exceptions.RequestException as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"API request failed: {str(e)}"
            )


class CopyleaksDetector(BaseAIDetector):
    """Copyleaks API-based detector."""

    LOGIN_URL = "https://id.copyleaks.com/v3/account/login/api"
    SCAN_URL = "https://api.copyleaks.com/v1/ai-detector/scan"

    def __init__(self, email: str, api_key: str):
        self.email = email
        self.api_key = api_key
        self._token = None

    @property
    def name(self) -> str:
        return "Copyleaks"

    def _get_token(self) -> Optional[str]:
        if self._token:
            return self._token
        try:
            response = requests.post(
                self.LOGIN_URL,
                json={"email": self.email, "key": self.api_key},
                timeout=30
            )
            response.raise_for_status()
            self._token = response.json().get("access_token")
            return self._token
        except Exception:
            return None

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        token = self._get_token()
        if not token:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error="Failed to authenticate with Copyleaks"
            )

        try:
            response = requests.post(
                self.SCAN_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            ai_prob = data.get("summary", {}).get("ai", 0.0) / 100.0

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "human_score": data.get("summary", {}).get("human", 0),
                    "scan_id": data.get("scannedDocument", {}).get("scanId")
                }
            )

        except requests.exceptions.RequestException as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"API request failed: {str(e)}"
            )


class ZeroGPTDetector(BaseAIDetector):
    """ZeroGPT API-based detector."""

    API_URL = "https://api.zerogpt.com/api/detect/detectText"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "ZeroGPT"

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "ApiKey": self.api_key,
                    "Content-Type": "application/json"
                },
                json={"input_text": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            ai_percentage = data.get("data", {}).get("isHuman", 100)
            ai_prob = (100 - ai_percentage) / 100.0

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "feedback": data.get("data", {}).get("feedback"),
                    "text_words": data.get("data", {}).get("textWords", 0)
                }
            )

        except requests.exceptions.RequestException as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"API request failed: {str(e)}"
            )


# =============================================================================
# Local Model-Based Detectors
# =============================================================================

class HuggingFaceDetector(BaseAIDetector):
    """
    Local Hugging Face model-based detector.
    Uses RoBERTa-based models for offline AI detection.
    """

    # Default HuggingFace cache directory
    HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/")

    # Default model to use
    DEFAULT_MODEL = "roberta-base-openai-detector"

    # Model name aliases for convenience
    MODEL_ALIASES = {
        "fakespot": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "openai": "openai-community/roberta-base-openai-detector",
        "roberta": "roberta-base-openai-detector",
        "chatgpt": "Hello-SimpleAI/chatgpt-detector-roberta",
        "desklib": "desklib/ai-text-detector-v1.01",
    }

    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize with a Hugging Face model.

        Args:
            model_name: Model name or alias. If None, uses DEFAULT_MODEL.
                Options:
                - None (uses default roberta-base-openai-detector)
                - Alias: "fakespot", "openai", "roberta", "chatgpt"
                - Full model name: "fakespot-ai/roberta-base-ai-text-detection-v1"
                - Full local path to model directory
            cache_dir: HuggingFace cache directory (default: ~/.cache/huggingface/hub/)
        """
        self.cache_dir = cache_dir or self.HF_CACHE_DIR

        if model_name is None:
            self.model_name = self.DEFAULT_MODEL
        else:
            # Check if it's an alias
            self.model_name = self.MODEL_ALIASES.get(model_name.lower(), model_name)

        self._pipeline = None
        self._load_error = None
        self._resolved_path = None

    def _resolve_model_path(self) -> str:
        """
        Resolve model name to local cache path if available.

        Returns:
            Resolved model path or original model name
        """
        # If it's already a full path, use it directly
        if os.path.isdir(os.path.expanduser(self.model_name)):
            return os.path.expanduser(self.model_name)

        # Try to find in HuggingFace cache
        # Cache folder format: models--{org}--{model} or models--{model}
        cache_folder_name = "models--" + self.model_name.replace("/", "--")
        cache_path = os.path.join(self.cache_dir, cache_folder_name)

        if os.path.isdir(cache_path):
            # Find the latest snapshot
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    # Use the first snapshot (usually there's only one)
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    if os.path.isdir(snapshot_path):
                        return snapshot_path

        # Return original name - let transformers handle it
        return self.model_name

    @property
    def name(self) -> str:
        model_short = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
        return f"HuggingFace ({model_short})"

    def _load_model(self):
        if self._pipeline is not None or self._load_error is not None:
            return

        try:
            from transformers import pipeline

            # Resolve model path from cache
            self._resolved_path = self._resolve_model_path()

            self._pipeline = pipeline(
                "text-classification",
                model=self._resolved_path,
                truncation=True,
                max_length=512,
                local_files_only=True  # Always use locally cached models, never download
            )
        except ImportError:
            self._load_error = "transformers not installed. Run: pip install transformers torch"
        except Exception as e:
            error_msg = str(e)
            if "local_files_only" in error_msg or "not found" in error_msg.lower():
                available = self._list_available_models()
                self._load_error = (
                    f"Model '{self.model_name}' not found in local cache.\n"
                    f"Available models: {', '.join(available) if available else 'none'}\n"
                    f"Download with: python -c \"from transformers import pipeline; "
                    f"pipeline('text-classification', model='{self.model_name}')\""
                )
            else:
                self._load_error = f"Failed to load model: {error_msg}"

    def _list_available_models(self) -> List[str]:
        """List available models in the cache directory."""
        models = []
        if os.path.isdir(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                if item.startswith("models--"):
                    # Convert cache folder name back to model name
                    model_name = item[8:].replace("--", "/")  # Remove "models--" prefix
                    models.append(model_name)
        return models

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_model()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            truncated_text = text[:2000] if len(text) > 2000 else text
            result = self._pipeline(truncated_text)[0]
            label = result["label"].lower()
            score = result["score"]

            if "fake" in label or "ai" in label or "generated" in label:
                ai_prob = score
            elif "real" in label or "human" in label:
                ai_prob = 1 - score
            else:
                ai_prob = score

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "model": self.model_name,
                    "resolved_path": self._resolved_path,
                    "raw_label": result["label"],
                    "raw_score": result["score"]
                }
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"Detection failed: {str(e)}"
            )


class OpenAIDetector(HuggingFaceDetector):
    """
    OpenAI's RoBERTa-based GPT detector.
    Wrapper around HuggingFace detector with the specific OpenAI model.
    """

    def __init__(self):
        super().__init__(model_name="openai-community/roberta-base-openai-detector")

    @property
    def name(self) -> str:
        return "OpenAI-Detector"


class BinocularsDetector(BaseAIDetector):
    """
    Binoculars detector - uses perplexity comparison between two models.

    Based on the paper: "Spotting LLMs With Binoculars: Zero-Shot Detection of
    Machine-Generated Text"

    Compares perplexity from an observer model vs a performer model.
    AI-generated text tends to have lower perplexity ratio.
    """

    def __init__(
        self,
        observer_model: str = "tiiuae/falcon-7b",
        performer_model: str = "tiiuae/falcon-7b-instruct",
        threshold: float = 0.9
    ):
        """
        Initialize Binoculars detector.

        Args:
            observer_model: Base model for observation
            performer_model: Instruction-tuned model
            threshold: Detection threshold (lower ratio = more likely AI)
        """
        self.observer_model_name = observer_model
        self.performer_model_name = performer_model
        self.threshold = threshold
        self._observer = None
        self._performer = None
        self._tokenizer = None
        self._load_error = None
        self._device = None

    @property
    def name(self) -> str:
        return "Binoculars"

    def _load_models(self):
        if self._load_error is not None:
            return
        if self._observer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.observer_model_name,
                trust_remote_code=True
            )

            # Load observer model
            self._observer = AutoModelForCausalLM.from_pretrained(
                self.observer_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None
            )

            # Load performer model
            self._performer = AutoModelForCausalLM.from_pretrained(
                self.performer_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None
            )

            if self._device == "cpu":
                self._observer = self._observer.to(self._device)
                self._performer = self._performer.to(self._device)

        except ImportError:
            self._load_error = "transformers/torch not installed. Run: pip install transformers torch"
        except Exception as e:
            self._load_error = f"Failed to load Binoculars models: {str(e)}"

    def _compute_perplexity(self, model, text: str) -> float:
        """Compute perplexity of text using the given model."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self._device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        return math.exp(loss)

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text, min_length=100)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_models()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            # Compute perplexity with both models
            observer_ppl = self._compute_perplexity(self._observer, text)
            performer_ppl = self._compute_perplexity(self._performer, text)

            # Calculate ratio (AI text has lower ratio)
            ratio = performer_ppl / observer_ppl if observer_ppl > 0 else 1.0

            # Convert to probability (lower ratio = higher AI probability)
            # Using sigmoid-like transformation
            ai_prob = 1 / (1 + math.exp(5 * (ratio - self.threshold)))

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "observer_perplexity": observer_ppl,
                    "performer_perplexity": performer_ppl,
                    "perplexity_ratio": ratio,
                    "threshold": self.threshold
                }
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"Binoculars detection failed: {str(e)}"
            )


class FastDetectGPTDetector(BaseAIDetector):
    """
    Fast-DetectGPT detector - curvature-based detection without sampling.

    Based on the paper: "Fast-DetectGPT: Efficient Zero-Shot Detection of
    Machine-Generated Text via Conditional Probability Curvature"

    Uses the curvature of log probability as a detection signal.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        threshold: float = 0.0
    ):
        """
        Initialize Fast-DetectGPT detector.

        Args:
            model_name: Model to use for detection (gpt2, gpt2-medium, etc.)
            threshold: Detection threshold (positive = AI, negative = human)
        """
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._load_error = None
        self._device = None

    @property
    def name(self) -> str:
        return "Fast-DetectGPT"

    def _load_model(self):
        if self._load_error is not None:
            return
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()

        except ImportError:
            self._load_error = "transformers/torch not installed. Run: pip install transformers torch"
        except Exception as e:
            self._load_error = f"Failed to load model: {str(e)}"

    def _compute_curvature(self, text: str) -> Tuple[float, Dict]:
        """Compute the conditional probability curvature."""
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get the log probability of each actual token
        input_ids = inputs["input_ids"]
        token_log_probs = torch.gather(
            log_probs[:, :-1, :],
            2,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Compute curvature (second derivative approximation)
        if token_log_probs.shape[1] < 3:
            return 0.0, {}

        # Simple curvature: variance of log prob differences
        log_prob_diffs = token_log_probs[:, 1:] - token_log_probs[:, :-1]
        curvature = log_prob_diffs.var().item()

        # Mean log probability
        mean_log_prob = token_log_probs.mean().item()

        return curvature, {
            "mean_log_prob": mean_log_prob,
            "curvature": curvature,
            "num_tokens": token_log_probs.shape[1]
        }

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_model()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            curvature, details = self._compute_curvature(text)

            # AI text tends to have HIGHER curvature (more predictable patterns)
            # Observed values: AI ~17-18, Human ~14-15
            # Convert to probability using sigmoid transformation
            # Center around 16.0 (midpoint of observed range)
            # Scale factor of 0.5 gives good separation
            center = 16.0
            scale = 0.5
            normalized = (curvature - center) * scale
            ai_prob = 1 / (1 + math.exp(-normalized))  # Note: negative for higher=AI

            # Also factor in mean log probability (AI text has higher/less negative)
            mean_log_prob = details.get("mean_log_prob", -5.0)
            # AI text typically has mean_log_prob around -4, human around -5 to -6
            log_prob_factor = 1 / (1 + math.exp(-(mean_log_prob + 4.5) * 2))

            # Combine both signals (weighted average)
            combined_prob = 0.7 * ai_prob + 0.3 * log_prob_factor
            combined_prob = max(0.0, min(1.0, combined_prob))  # Clamp to [0, 1]

            details["threshold"] = self.threshold
            details["curvature_prob"] = ai_prob
            details["log_prob_factor"] = log_prob_factor

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=combined_prob > 0.5,
                ai_probability=combined_prob,
                confidence=abs(combined_prob - 0.5) * 2,
                details=details
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"Fast-DetectGPT detection failed: {str(e)}"
            )


class LLMDetDetector(BaseAIDetector):
    """
    LLMDet detector - uses proxy perplexity for detection.

    Based on the paper: "LLMDet: A Third Party Large Language Models Generated
    Text Detection Tool"

    Can identify specific LLM sources (GPT-2, LLaMA, OPT, etc.) and human text.
    Uses n-gram analysis and proxy perplexity.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize LLMDet detector.

        Args:
            threshold: Probability threshold for AI detection (default 0.5)
        """
        self.threshold = threshold
        self._loaded = False
        self._load_error = None

    @property
    def name(self) -> str:
        return "LLMDet"

    def _load_model(self):
        if self._load_error is not None:
            return
        if self._loaded:
            return

        try:
            import llmdet
            llmdet.load_probability()
            self._loaded = True
        except ImportError as e:
            if "unilm" in str(e):
                self._load_error = "llmdet requires 'unilm' package which is not on PyPI. See: https://github.com/TrustedLLM/LLMDet for manual installation."
            else:
                self._load_error = f"llmdet not fully installed. Run: pip install llmdet datasets. Error: {str(e)}"
        except Exception as e:
            self._load_error = f"Failed to load LLMDet: {str(e)}"

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_model()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            import llmdet
            result = llmdet.detect(text)

            # result is a list of dicts with model probabilities
            # e.g., [{"OPT": 0.545, "GPT-2": 0.439, "Human_write": 0.00001, ...}]
            if not result or not isinstance(result, list):
                return AIDetectionResult(
                    detector_name=self.name,
                    is_ai_generated=False,
                    ai_probability=0.0,
                    error="No detection result returned"
                )

            probs = result[0] if result else {}

            # Get human probability
            human_prob = probs.get("Human_write", 0.0)

            # AI probability is 1 - human probability
            ai_prob = 1.0 - human_prob

            # Find the most likely LLM source (excluding Human_write)
            llm_sources = {k: v for k, v in probs.items() if k != "Human_write"}
            most_likely_source = max(llm_sources.items(), key=lambda x: x[1]) if llm_sources else ("Unknown", 0.0)

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > self.threshold,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "human_probability": human_prob,
                    "most_likely_source": most_likely_source[0],
                    "most_likely_source_prob": most_likely_source[1],
                    "all_probabilities": probs,
                    "threshold": self.threshold
                }
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"LLMDet detection failed: {str(e)}"
            )


class DesklibDetector(BaseAIDetector):
    """
    Desklib AI Text Detector - uses DeBERTa-v3-large fine-tuned model.

    Based on microsoft/deberta-v3-large, this model leads the RAID Benchmark
    for AI Detection. Uses mean pooling and a linear classifier.

    Model: desklib/ai-text-detector-v1.01
    Paper: https://huggingface.co/desklib/ai-text-detector-v1.01
    """

    MODEL_NAME = "desklib/ai-text-detector-v1.01"

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 768,
        threshold: float = 0.5,
        local_files_only: bool = False
    ):
        """
        Initialize Desklib detector.

        Args:
            model_path: Path to model (defaults to HuggingFace hub model)
            max_length: Maximum token length (default 768)
            threshold: Classification threshold (default 0.5)
            local_files_only: If True, only use locally cached model
        """
        self.model_path = model_path or self.MODEL_NAME
        self.max_length = max_length
        self.threshold = threshold
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load_error = None

    @property
    def name(self) -> str:
        return "Desklib"

    def _load_model(self):
        if self._load_error is not None:
            return
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer, AutoConfig, DebertaV2Model, PreTrainedModel

            # Define the custom model architecture matching Desklib's saved format
            class DesklibAIDetectionModel(PreTrainedModel):
                config_class = AutoConfig

                def __init__(self, config):
                    super().__init__(config)
                    self.model = DebertaV2Model(config)
                    self.classifier = nn.Linear(config.hidden_size, 1)
                    self.post_init()

                def forward(self, input_ids, attention_mask=None, labels=None):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs[0]

                    # Mean pooling
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                    pooled_output = sum_embeddings / sum_mask

                    logits = self.classifier(pooled_output)
                    loss = None
                    if labels is not None:
                        loss_fct = nn.BCEWithLogitsLoss()
                        loss = loss_fct(logits.view(-1), labels.float())

                    return {"logits": logits, "loss": loss} if loss else {"logits": logits}

            # Detect device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load tokenizer and model
            print(f"Loading Desklib model from: {self.model_path}")
            print(f"Using device: {self._device}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only
            )

            # Load config and initialize model
            config = AutoConfig.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only
            )
            self._model = DesklibAIDetectionModel.from_pretrained(
                self.model_path,
                config=config,
                local_files_only=self.local_files_only,
                ignore_mismatched_sizes=True
            )
            self._model.to(self._device)
            self._model.eval()

            print("Desklib model loaded successfully!")

        except ImportError:
            self._load_error = "transformers/torch not installed. Run: pip install transformers torch"
        except Exception as e:
            error_msg = str(e)
            if "local_files_only" in error_msg or "not found" in error_msg.lower():
                self._load_error = (
                    f"Model '{self.model_path}' not found. "
                    f"Download it first with: "
                    f"python -c \"from transformers import AutoTokenizer, AutoModel; "
                    f"AutoTokenizer.from_pretrained('{self.model_path}'); "
                    f"AutoModel.from_pretrained('{self.model_path}')\""
                )
            else:
                self._load_error = f"Failed to load Desklib model: {error_msg}"

    def _predict(self, text: str) -> Tuple[float, int]:
        """
        Predict AI probability for a single text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (probability, label)
        """
        import torch

        encoded = self._tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probability = torch.sigmoid(logits).item()

        label = 1 if probability >= self.threshold else 0
        return probability, label

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_model()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            probability, label = self._predict(text)

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=label == 1,
                ai_probability=probability,
                confidence=abs(probability - 0.5) * 2,
                details={
                    "model": self.model_path,
                    "threshold": self.threshold,
                    "max_length": self.max_length,
                    "text_length": len(text)
                }
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"Desklib detection failed: {str(e)}"
            )


class ROUGESimilarityChecker(BaseAIDetector):
    """
    ROUGE-based similarity checker.

    Compares text against known AI-generated patterns or reference corpus.
    Uses pyrouge/rouge-score for evaluation.

    Note: This is more useful for plagiarism/template detection than
    pure AI detection, but can identify AI text that follows common patterns.
    """

    def __init__(
        self,
        reference_patterns: Optional[List[str]] = None,
        threshold: float = 0.3
    ):
        """
        Initialize ROUGE checker.

        Args:
            reference_patterns: List of known AI-generated text patterns
            threshold: ROUGE-L score threshold for flagging
        """
        self.threshold = threshold
        self.reference_patterns = reference_patterns or self._default_patterns()
        self._scorer = None
        self._load_error = None

    @property
    def name(self) -> str:
        return "ROUGE-Similarity"

    def _default_patterns(self) -> List[str]:
        """Default AI-associated text patterns."""
        return [
            "In conclusion, it is important to note that",
            "Furthermore, it is worth mentioning that",
            "Additionally, one must consider the fact that",
            "It is essential to understand that",
            "As we have discussed, the importance of",
            "To summarize the key points discussed above",
            "In summary, we can conclude that",
            "Based on the analysis presented above",
            "Taking all factors into consideration",
            "It is crucial to acknowledge that"
        ]

    def _load_scorer(self):
        if self._load_error is not None:
            return
        if self._scorer is not None:
            return

        try:
            from rouge_score import rouge_scorer
            self._scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        except ImportError:
            self._load_error = "rouge-score not installed. Run: pip install rouge-score"

    def detect(self, text: str) -> AIDetectionResult:
        error = self._validate_text(text)
        if error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=error
            )

        self._load_scorer()

        if self._load_error:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=self._load_error
            )

        try:
            max_rouge_l = 0.0
            max_rouge1 = 0.0
            matched_pattern = None

            for pattern in self.reference_patterns:
                scores = self._scorer.score(pattern, text)
                rouge_l = scores['rougeL'].fmeasure
                rouge1 = scores['rouge1'].fmeasure

                if rouge_l > max_rouge_l:
                    max_rouge_l = rouge_l
                    max_rouge1 = rouge1
                    matched_pattern = pattern

            # Convert to AI probability
            ai_prob = min(max_rouge_l / self.threshold, 1.0) if self.threshold > 0 else 0.0

            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=ai_prob > 0.5,
                ai_probability=ai_prob,
                confidence=abs(ai_prob - 0.5) * 2,
                details={
                    "max_rouge_l": max_rouge_l,
                    "max_rouge1": max_rouge1,
                    "matched_pattern": matched_pattern[:50] + "..." if matched_pattern and len(matched_pattern) > 50 else matched_pattern,
                    "threshold": self.threshold,
                    "patterns_checked": len(self.reference_patterns)
                }
            )

        except Exception as e:
            return AIDetectionResult(
                detector_name=self.name,
                is_ai_generated=False,
                ai_probability=0.0,
                error=f"ROUGE analysis failed: {str(e)}"
            )


# =============================================================================
# Ensemble Detector
# =============================================================================

class EnsembleDetector(BaseAIDetector):
    """
    Ensemble detector that combines multiple AI detectors.
    Uses voting and weighted averaging for final decision.
    """

    def __init__(
        self,
        detectors: List[BaseAIDetector],
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        self.detectors = detectors
        self.threshold = threshold
        self.weights = weights or {d.name: 1.0 for d in detectors}

    @property
    def name(self) -> str:
        return "Ensemble"

    def detect(self, text: str) -> AIDetectionResult:
        ensemble_result = self.detect_ensemble(text)

        return AIDetectionResult(
            detector_name=self.name,
            is_ai_generated=ensemble_result.consensus_is_ai,
            ai_probability=ensemble_result.weighted_probability,
            confidence=ensemble_result.agreement_score,
            details={
                "individual_results": [
                    {
                        "detector": r.detector_name,
                        "ai_probability": r.ai_probability,
                        "is_ai": r.is_ai_generated,
                        "error": r.error
                    }
                    for r in ensemble_result.individual_results
                ],
                "average_probability": ensemble_result.average_probability,
                "detectors_used": ensemble_result.detectors_used,
                "detectors_succeeded": ensemble_result.detectors_succeeded
            }
        )

    def detect_ensemble(self, text: str) -> EnsembleResult:
        result = EnsembleResult()
        result.detectors_used = len(self.detectors)

        successful_results = []

        for detector in self.detectors:
            try:
                detection = detector.detect(text)
                result.individual_results.append(detection)
                if not detection.error:
                    successful_results.append(detection)
            except Exception as e:
                result.individual_results.append(AIDetectionResult(
                    detector_name=detector.name,
                    is_ai_generated=False,
                    ai_probability=0.0,
                    error=f"Detector failed: {str(e)}"
                ))

        result.detectors_succeeded = len(successful_results)

        if not successful_results:
            return result

        result.average_probability = sum(
            r.ai_probability for r in successful_results
        ) / len(successful_results)

        total_weight = 0.0
        weighted_sum = 0.0

        for r in successful_results:
            weight = self.weights.get(r.detector_name, 1.0)
            weighted_sum += r.ai_probability * weight
            total_weight += weight

        if total_weight > 0:
            result.weighted_probability = weighted_sum / total_weight

        result.consensus_is_ai = result.weighted_probability > self.threshold

        if len(successful_results) > 1:
            ai_votes = sum(1 for r in successful_results if r.is_ai_generated)
            human_votes = len(successful_results) - ai_votes
            majority = max(ai_votes, human_votes)
            result.agreement_score = majority / len(successful_results)
        else:
            result.agreement_score = 1.0

        return result


# =============================================================================
# Text Analysis Utilities
# =============================================================================

def detect_duplicate_content(text: str, min_phrase_length: int = 20, min_occurrences: int = 2) -> Tuple[bool, List[str], float]:
    """
    Detect duplicate/repeated phrases in text.

    Args:
        text: Input text to analyze
        min_phrase_length: Minimum characters for a phrase to be considered
        min_occurrences: Minimum times a phrase must appear to be flagged

    Returns:
        Tuple of (has_duplicates, list of duplicate phrases, duplicate ratio)
    """
    # Normalize text
    normalized = ' '.join(text.lower().split())

    # Split into sentences
    sentences = re.split(r'[.!?]+', normalized)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_phrase_length]

    # Find duplicate sentences
    sentence_counts = {}
    for sentence in sentences:
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

    duplicates = [s for s, count in sentence_counts.items() if count >= min_occurrences]

    # Also check for repeated phrases within sentences (n-grams)
    words = normalized.split()
    phrase_counts = {}

    # Check 5-8 word phrases
    for n in range(5, 9):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            if len(phrase) >= min_phrase_length:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    repeated_phrases = [p for p, count in phrase_counts.items() if count >= min_occurrences]

    # Combine and dedupe (prefer longer phrases)
    all_duplicates = list(set(duplicates + repeated_phrases))
    all_duplicates.sort(key=len, reverse=True)

    # Remove phrases that are substrings of longer duplicates
    filtered_duplicates = []
    for phrase in all_duplicates:
        is_substring = any(phrase in longer and phrase != longer for longer in filtered_duplicates)
        if not is_substring:
            filtered_duplicates.append(phrase)

    # Calculate duplicate ratio (how much of text is duplicated)
    duplicate_chars = sum(len(p) * (sentence_counts.get(p, 1) + phrase_counts.get(p, 1) - 1)
                         for p in filtered_duplicates)
    duplicate_ratio = min(duplicate_chars / len(normalized), 1.0) if normalized else 0.0

    return len(filtered_duplicates) > 0, filtered_duplicates[:5], duplicate_ratio


def analyze_text_segments(
    text: str,
    detector: 'BaseAIDetector',
    segment_size: int = 200,
    overlap: int = 50
) -> List[Dict]:
    """
    Analyze text in segments to identify which portions have high AI probability.

    Args:
        text: Input text to analyze
        detector: AI detector to use
        segment_size: Approximate words per segment
        overlap: Word overlap between segments

    Returns:
        List of dictionaries with segment info and AI probability
    """
    words = text.split()

    if len(words) < segment_size:
        # Text too short to segment meaningfully
        return []

    segments = []
    i = 0
    segment_num = 1

    while i < len(words):
        end = min(i + segment_size, len(words))
        segment_text = ' '.join(words[i:end])

        # Get start/end character positions for reference
        char_start = len(' '.join(words[:i])) + (1 if i > 0 else 0)
        char_end = char_start + len(segment_text)

        try:
            result = detector.detect(segment_text)
            segments.append({
                "segment_num": segment_num,
                "char_start": char_start,
                "char_end": char_end,
                "word_start": i,
                "word_end": end,
                "ai_probability": result.ai_probability,
                "is_ai": result.is_ai_generated,
                "preview": segment_text[:100] + "..." if len(segment_text) > 100 else segment_text
            })
        except Exception:
            pass

        i += segment_size - overlap
        segment_num += 1

        # Limit to reasonable number of segments
        if segment_num > 20:
            break

    return segments


def determine_ai_reason(
    result: 'AIDetectionResult',
    has_duplicates: bool,
    duplicate_ratio: float,
    high_ai_segments: List[Dict]
) -> Tuple[str, List[str]]:
    """
    Determine the primary reason for AI classification.

    Args:
        result: Detection result
        has_duplicates: Whether duplicates were found
        duplicate_ratio: Ratio of duplicate content
        high_ai_segments: Segments with high AI probability

    Returns:
        Tuple of (primary_reason, list of contributing factors)
    """
    reasons = []
    factors = []

    # Check for duplicates
    if has_duplicates:
        if duplicate_ratio > 0.3:
            reasons.append(("High duplicate content", 0.9))
            factors.append(f"Duplicate content ratio: {duplicate_ratio:.1%}")
        elif duplicate_ratio > 0.1:
            reasons.append(("Moderate duplicate content", 0.6))
            factors.append(f"Duplicate content ratio: {duplicate_ratio:.1%}")

    # Check segment consistency
    if high_ai_segments:
        ai_segments = [s for s in high_ai_segments if s.get("ai_probability", 0) > 0.7]
        if len(ai_segments) == len(high_ai_segments) and len(high_ai_segments) > 1:
            reasons.append(("Uniformly high AI probability across all segments", 0.85))
            factors.append("All text segments show AI patterns")
        elif ai_segments:
            segment_nums = [s["segment_num"] for s in ai_segments]
            reasons.append(("Specific segments show high AI probability", 0.7))
            factors.append(f"High AI segments: {segment_nums}")

    # Check detector-specific details
    details = result.details or {}

    if "curvature" in details:
        curvature = details["curvature"]
        if curvature > 17:
            reasons.append(("High text predictability (curvature analysis)", 0.75))
            factors.append(f"Curvature score: {curvature:.2f}")

    if "mean_log_prob" in details:
        mlp = details["mean_log_prob"]
        if mlp > -4:
            reasons.append(("High token probability (low perplexity)", 0.7))
            factors.append(f"Mean log probability: {mlp:.2f}")

    if "raw_label" in details:
        factors.append(f"Model classification: {details['raw_label']}")

    if "most_likely_source" in details:
        source = details["most_likely_source"]
        if source != "Human_write":
            reasons.append((f"Text patterns match {source} model", 0.8))
            factors.append(f"Identified source: {source}")

    # Check overall probability
    if result.ai_probability > 0.9:
        reasons.append(("Very high overall AI probability", 0.95))
    elif result.ai_probability > 0.7:
        reasons.append(("High overall AI probability", 0.8))

    # Sort by confidence and pick primary
    reasons.sort(key=lambda x: x[1], reverse=True)
    primary_reason = reasons[0][0] if reasons else "AI patterns detected in text structure"

    # Add probability as a factor
    factors.insert(0, f"AI probability: {result.ai_probability:.1%}")

    return primary_reason, factors


def perform_detailed_analysis(
    text: str,
    result: 'AIDetectionResult',
    detector: 'BaseAIDetector'
) -> AITextAnalysis:
    """
    Perform detailed analysis on AI-classified text.

    Args:
        text: Original text
        result: Detection result
        detector: Detector used

    Returns:
        AITextAnalysis with detailed findings
    """
    analysis = AITextAnalysis()

    # Detect duplicates
    has_duplicates, duplicate_phrases, duplicate_ratio = detect_duplicate_content(text)
    analysis.has_duplicates = has_duplicates
    analysis.duplicate_phrases = duplicate_phrases
    analysis.duplicate_ratio = duplicate_ratio

    # Analyze segments (only for longer texts to avoid redundant processing)
    if len(text) > 500:
        segments = analyze_text_segments(text, detector)
        # Keep only high AI segments
        analysis.high_ai_segments = [s for s in segments if s.get("ai_probability", 0) > 0.6]

    # Determine reason
    primary_reason, factors = determine_ai_reason(
        result, has_duplicates, duplicate_ratio, analysis.high_ai_segments
    )
    analysis.primary_reason = primary_reason
    analysis.contributing_factors = factors

    return analysis


# =============================================================================
# Factory
# =============================================================================

class AIDetectorFactory:
    """Factory for creating AI detectors."""

    @staticmethod
    def create(detector_type: DetectorType, **kwargs) -> BaseAIDetector:
        """Create a detector instance."""
        if detector_type == DetectorType.GPTZERO:
            api_key = kwargs.get("api_key") or os.environ.get("GPTZERO_API_KEY")
            if not api_key:
                raise ValueError("GPTZero requires api_key or GPTZERO_API_KEY env var")
            return GPTZeroDetector(api_key)

        elif detector_type == DetectorType.ORIGINALITY:
            api_key = kwargs.get("api_key") or os.environ.get("ORIGINALITY_API_KEY")
            if not api_key:
                raise ValueError("Originality.ai requires api_key or ORIGINALITY_API_KEY env var")
            return OriginalityDetector(api_key)

        elif detector_type == DetectorType.COPYLEAKS:
            email = kwargs.get("email") or os.environ.get("COPYLEAKS_EMAIL")
            api_key = kwargs.get("api_key") or os.environ.get("COPYLEAKS_API_KEY")
            if not email or not api_key:
                raise ValueError("Copyleaks requires email and api_key")
            return CopyleaksDetector(email, api_key)

        elif detector_type == DetectorType.ZEROGPT:
            api_key = kwargs.get("api_key") or os.environ.get("ZEROGPT_API_KEY")
            if not api_key:
                raise ValueError("ZeroGPT requires api_key or ZEROGPT_API_KEY env var")
            return ZeroGPTDetector(api_key)

        elif detector_type == DetectorType.HUGGINGFACE_ROBERTA:
            model_name = kwargs.get("model_name", "roberta-base-openai-detector")
            return HuggingFaceDetector(model_name)

        elif detector_type == DetectorType.OPENAI_DETECTOR:
            return OpenAIDetector()

        elif detector_type == DetectorType.BINOCULARS:
            return BinocularsDetector(
                observer_model=kwargs.get("observer_model", "tiiuae/falcon-7b"),
                performer_model=kwargs.get("performer_model", "tiiuae/falcon-7b-instruct"),
                threshold=kwargs.get("threshold", 0.9)
            )

        elif detector_type == DetectorType.FAST_DETECTGPT:
            return FastDetectGPTDetector(
                model_name=kwargs.get("model_name", "gpt2-medium"),
                threshold=kwargs.get("threshold", 0.0)
            )

        elif detector_type == DetectorType.LLMDET:
            return LLMDetDetector(
                threshold=kwargs.get("threshold", 0.5)
            )

        elif detector_type == DetectorType.ROUGE_CHECKER:
            return ROUGESimilarityChecker(
                reference_patterns=kwargs.get("reference_patterns"),
                threshold=kwargs.get("threshold", 0.3)
            )

        elif detector_type == DetectorType.DESKLIB:
            return DesklibDetector(
                model_path=kwargs.get("model_path"),
                max_length=kwargs.get("max_length", 768),
                threshold=kwargs.get("threshold", 0.5),
                local_files_only=kwargs.get("local_files_only", False)
            )

        elif detector_type == DetectorType.ENSEMBLE:
            detectors = kwargs.get("detectors", [])
            if not detectors:
                raise ValueError("Ensemble requires list of detectors")
            return EnsembleDetector(
                detectors,
                weights=kwargs.get("weights"),
                threshold=kwargs.get("threshold", 0.5)
            )

        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    @staticmethod
    def create_from_env() -> List[BaseAIDetector]:
        """Create all available detectors based on environment variables."""
        detectors = []

        if os.environ.get("GPTZERO_API_KEY"):
            detectors.append(AIDetectorFactory.create(DetectorType.GPTZERO))

        if os.environ.get("ORIGINALITY_API_KEY"):
            detectors.append(AIDetectorFactory.create(DetectorType.ORIGINALITY))

        if os.environ.get("ZEROGPT_API_KEY"):
            detectors.append(AIDetectorFactory.create(DetectorType.ZEROGPT))

        if os.environ.get("COPYLEAKS_EMAIL") and os.environ.get("COPYLEAKS_API_KEY"):
            detectors.append(AIDetectorFactory.create(DetectorType.COPYLEAKS))

        # Try to add local detectors
        try:
            detectors.append(AIDetectorFactory.create(DetectorType.HUGGINGFACE_ROBERTA))
        except Exception:
            pass

        try:
            detectors.append(AIDetectorFactory.create(DetectorType.ROUGE_CHECKER))
        except Exception:
            pass

        return detectors

    @staticmethod
    def list_available() -> Dict[str, str]:
        """List all available detector types with descriptions."""
        return {
            "gptzero": "GPTZero API - Commercial AI detection service",
            "originality": "Originality.ai API - AI and plagiarism detection",
            "copyleaks": "Copyleaks API - AI content detection",
            "zerogpt": "ZeroGPT API - AI text detection",
            "huggingface_roberta": "HuggingFace RoBERTa - Local OpenAI detector model",
            "openai_detector": "OpenAI Detector - Official OpenAI RoBERTa model",
            "binoculars": "Binoculars - Perplexity comparison between two models",
            "fast_detectgpt": "Fast-DetectGPT - Curvature-based detection",
            "llmdet": "LLMDet - Proxy perplexity based detection with LLM source identification",
            "rouge_checker": "ROUGE Similarity - Pattern matching with ROUGE metrics",
            "desklib": "Desklib - DeBERTa-v3-large based detector (RAID benchmark leader)",
            "ensemble": "Ensemble - Combine multiple detectors with voting"
        }


# =============================================================================
# File Processing and CSV Output
# =============================================================================

def load_notes_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all text files from a folder.

    Args:
        folder_path: Path to the folder containing note files

    Returns:
        List of dictionaries with 'filename' and 'text' keys
    """
    notes = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return notes

    # Get all text files
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        print(f"Warning: No .txt files found in '{folder_path}'")
        return notes

    print(f"Found {len(txt_files)} text files in '{folder_path}'")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            notes.append({
                "filename": txt_file.name,
                "text": text
            })
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")

    return notes


def generate_explanation(result: AIDetectionResult, include_analysis: bool = True) -> str:
    """
    Generate a human-readable explanation for the classification.

    Args:
        result: AIDetectionResult from a detector
        include_analysis: Whether to include detailed analysis

    Returns:
        Explanation string
    """
    explanations = []

    if result.error:
        return f"Error: {result.error}"

    # Add detector name
    explanations.append(f"Detector: {result.detector_name}")

    # Add probability interpretation
    ai_prob = result.ai_probability
    if ai_prob > 0.8:
        explanations.append("Strong AI-generated text patterns detected (>80%)")
    elif ai_prob > 0.6:
        explanations.append("Moderate AI-generated text indicators (60-80%)")
    elif ai_prob > 0.4:
        explanations.append("Borderline - mixed human/AI characteristics (40-60%)")
    elif ai_prob > 0.2:
        explanations.append("Mostly human-written with minor AI-like patterns (20-40%)")
    else:
        explanations.append("Strong human-written text characteristics (<20%)")

    # Add confidence level
    explanations.append(f"Confidence: {result.confidence:.1%}")

    # Add detailed analysis for AI-classified texts
    if include_analysis and result.analysis and result.is_ai_generated:
        analysis = result.analysis

        # Add primary reason
        if analysis.primary_reason:
            explanations.append(f"Primary reason: {analysis.primary_reason}")

        # Add duplicate info
        if analysis.has_duplicates:
            explanations.append(f"Duplicate content detected ({analysis.duplicate_ratio:.1%} of text)")
            if analysis.duplicate_phrases:
                sample = analysis.duplicate_phrases[0][:50]
                explanations.append(f"Sample duplicate: '{sample}...'")

        # Add segment info
        if analysis.high_ai_segments:
            num_segments = len(analysis.high_ai_segments)
            avg_prob = sum(s["ai_probability"] for s in analysis.high_ai_segments) / num_segments
            explanations.append(f"High AI segments: {num_segments} (avg prob: {avg_prob:.1%})")

        # Add contributing factors
        if analysis.contributing_factors:
            factors_str = "; ".join(analysis.contributing_factors[:3])
            explanations.append(f"Factors: {factors_str}")

    # Add details if available
    if result.details:
        if "raw_label" in result.details:
            explanations.append(f"Raw output: {result.details['raw_label']}")
        if "model" in result.details:
            explanations.append(f"Model: {result.details['model']}")
        if "most_likely_source" in result.details:
            explanations.append(f"Likely source: {result.details['most_likely_source']}")

    return " | ".join(explanations)


def generate_analysis_reason(result: AIDetectionResult) -> str:
    """
    Generate a concise analysis reason for CSV output.

    Args:
        result: AIDetectionResult with analysis

    Returns:
        Concise reason string
    """
    if not result.is_ai_generated:
        return "N/A - Human written"

    if result.error:
        return f"Error: {result.error}"

    if not result.analysis:
        return "AI patterns detected"

    analysis = result.analysis
    reasons = []

    # Primary reason
    if analysis.primary_reason:
        reasons.append(analysis.primary_reason)

    # Duplicate indicator
    if analysis.has_duplicates:
        reasons.append(f"Duplicate: {analysis.duplicate_ratio:.0%}")

    # High segments indicator
    if analysis.high_ai_segments:
        reasons.append(f"High-AI segments: {len(analysis.high_ai_segments)}")

    return " | ".join(reasons) if reasons else "AI patterns detected"


def save_results_csv(
    results: List[Dict],
    output_path: str
):
    """
    Save detection results to CSV file.

    Args:
        results: List of detection result dictionaries
        output_path: Path to save CSV file

    Output columns:
        1. file_name: Name of the document
        2. classification: "AI_text" or "human_created"
        3. ai_probability: Probability score (0.0 to 1.0)
        4. analysis_reason: Primary reason for AI classification (for AI_text only)
        5. has_duplicates: Whether duplicate content was detected
        6. duplicate_ratio: Percentage of text that is duplicated
        7. high_ai_segments: Number of text segments with high AI probability
        8. explanation: Detailed explanation of the classification
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            "file_name",
            "classification",
            "ai_probability",
            "analysis_reason",
            "has_duplicates",
            "duplicate_ratio",
            "high_ai_segments",
            "explanation"
        ])

        # Write data rows
        for r in results:
            writer.writerow([
                r["filename"],
                r["classification"],
                f"{r['ai_probability']:.4f}",
                r.get("analysis_reason", ""),
                r.get("has_duplicates", ""),
                r.get("duplicate_ratio", ""),
                r.get("high_ai_segments", ""),
                r["explanation"]
            ])

    print(f"CSV results saved to: {output_path}")


def run_detection(
    data_folder: str = "note_data/cms_notes",
    detector_type: str = "huggingface_roberta",
    csv_file: str = "ai_detection_results.csv",
    model_name: Optional[str] = None
):
    """
    Run AI detection on all notes in the specified folder.

    Args:
        data_folder: Path to folder containing note files
        detector_type: Type of detector to use
        csv_file: Path to save CSV results
        model_name: Optional model name for HuggingFace detector
    """
    # Load notes
    notes = load_notes_from_folder(data_folder)

    if not notes:
        print("No notes to process. Exiting.")
        return

    print(f"\nProcessing {len(notes)} notes...")
    print("=" * 60)

    # Create detector
    try:
        if detector_type == "huggingface_roberta" and model_name:
            detector = HuggingFaceDetector(model_name=model_name)
        else:
            detector = AIDetectorFactory.create(DetectorType(detector_type))
        print(f"Using detector: {detector.name}")
    except Exception as e:
        print(f"Failed to create detector: {e}")
        return

    print("=" * 60)

    all_results = []

    for i, note in enumerate(notes, 1):
        filename = note["filename"]
        text = note["text"]

        print(f"\n[{i}/{len(notes)}] Processing: {filename}")

        try:
            result = detector.detect(text)

            # Determine classification
            classification = "AI_text" if result.is_ai_generated else "human_created"

            # Perform detailed analysis for AI-classified texts
            if result.is_ai_generated and not result.error:
                result.analysis = perform_detailed_analysis(text, result, detector)

            # Generate explanation
            explanation = generate_explanation(result)

            # Generate analysis reason
            analysis_reason = generate_analysis_reason(result)

            # Extract analysis fields
            has_duplicates = ""
            duplicate_ratio = ""
            high_ai_segments = ""

            if result.analysis:
                has_duplicates = "Yes" if result.analysis.has_duplicates else "No"
                duplicate_ratio = f"{result.analysis.duplicate_ratio:.1%}"
                high_ai_segments = str(len(result.analysis.high_ai_segments))

            # Store result
            result_dict = {
                "filename": filename,
                "classification": classification,
                "ai_probability": result.ai_probability,
                "analysis_reason": analysis_reason,
                "has_duplicates": has_duplicates,
                "duplicate_ratio": duplicate_ratio,
                "high_ai_segments": high_ai_segments,
                "explanation": explanation,
                "is_ai_generated": result.is_ai_generated,
                "confidence": result.confidence,
                "error": result.error
            }
            all_results.append(result_dict)

            # Print summary
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print(f"  AI Probability: {result.ai_probability:.2%}")
                print(f"  Classification: {classification}")
                if result.is_ai_generated and result.analysis:
                    print(f"  Analysis: {result.analysis.primary_reason}")
                    if result.analysis.has_duplicates:
                        print(f"  Duplicates: {result.analysis.duplicate_ratio:.1%} of text")
                    if result.analysis.high_ai_segments:
                        print(f"  High-AI segments: {len(result.analysis.high_ai_segments)}")

        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({
                "filename": filename,
                "classification": "error",
                "ai_probability": 0.0,
                "analysis_reason": f"Error: {str(e)}",
                "has_duplicates": "",
                "duplicate_ratio": "",
                "high_ai_segments": "",
                "explanation": f"Error: {str(e)}",
                "is_ai_generated": False,
                "confidence": 0.0,
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = [r for r in all_results if not r.get("error")]
    ai_count = sum(1 for r in valid_results if r["is_ai_generated"])
    total = len(valid_results)

    if total > 0:
        avg_prob = sum(r["ai_probability"] for r in valid_results) / total
        print(f"Total notes processed: {len(notes)}")
        print(f"Successful detections: {total}")
        print(f"Detected as AI-generated: {ai_count} ({ai_count/total*100:.1f}%)")
        print(f"Detected as Human-written: {total - ai_count} ({(total-ai_count)/total*100:.1f}%)")
        print(f"Average AI probability: {avg_prob:.2%}")
    else:
        print(f"Total notes processed: {len(notes)}")
        print("No successful detections")

    # Save CSV results
    save_results_csv(all_results, csv_file)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Detect AI-generated text using various detection algorithms"
    )

    parser.add_argument(
        "--data-folder",
        default="note_data/cms_notes",
        help="Path to folder containing note text files (default: note_data/cms_notes)"
    )

    parser.add_argument(
        "--detector",
        choices=[
            "huggingface_roberta", "openai_detector", "fast_detectgpt",
            "binoculars", "llmdet", "rouge_checker", "desklib"
        ],
        default="huggingface_roberta",
        help="Detector type to use (default: huggingface_roberta)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model name for HuggingFace detector (e.g., 'openai-community/roberta-base-openai-detector')"
    )

    parser.add_argument(
        "--csv",
        default="ai_detection_results.csv",
        help="Path to save CSV results (default: ai_detection_results.csv)"
    )

    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors and exit"
    )

    args = parser.parse_args()

    # List detectors if requested
    if args.list_detectors:
        print("Available AI detectors:")
        print("-" * 60)
        for key, desc in AIDetectorFactory.list_available().items():
            print(f"  {key:20} - {desc}")
        return

    # Run detection
    run_detection(
        data_folder=args.data_folder,
        detector_type=args.detector,
        csv_file=args.csv,
        model_name=args.model
    )


if __name__ == "__main__":
    main()
