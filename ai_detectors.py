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
import math
import requests
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


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
    ENSEMBLE = "ensemble"


@dataclass
class AIDetectionResult:
    """Results from AI detection analysis."""
    detector_name: str
    is_ai_generated: bool
    ai_probability: float
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None


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

    def __init__(self, model_name: str = "roberta-base-openai-detector"):
        """
        Initialize with a Hugging Face model.

        Args:
            model_name: Model to use. Options:
                - "roberta-base-openai-detector" (OpenAI's detector)
                - "Hello-SimpleAI/chatgpt-detector-roberta" (ChatGPT specific)
                - "openai-community/roberta-base-openai-detector"
        """
        self.model_name = model_name
        self._pipeline = None
        self._load_error = None

    @property
    def name(self) -> str:
        return f"HuggingFace ({self.model_name.split('/')[-1]})"

    def _load_model(self):
        if self._pipeline is not None or self._load_error is not None:
            return

        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=512
            )
        except ImportError:
            self._load_error = "transformers not installed. Run: pip install transformers torch"
        except Exception as e:
            self._load_error = f"Failed to load model: {str(e)}"

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
            "ensemble": "Ensemble - Combine multiple detectors with voting"
        }
