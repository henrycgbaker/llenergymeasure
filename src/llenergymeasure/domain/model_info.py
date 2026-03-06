"""Model information domain models for LLM Bench."""

from pydantic import BaseModel, Field


class QuantizationSpec(BaseModel):
    """Quantization configuration for a model."""

    enabled: bool = Field(default=False, description="Whether quantization is enabled")
    bits: int | None = Field(default=None, description="Quantization bits (4 or 8)")
    method: str = Field(default="none", description="Quantization method (bitsandbytes, gptq, etc)")
    compute_dtype: str = Field(
        default="float16", description="Compute dtype (float16 for BNB, varies for others)"
    )

    @property
    def is_bnb(self) -> bool:
        """Check if using BitsAndBytes quantization."""
        return self.method == "bitsandbytes"


class ModelInfo(BaseModel):
    """Information about the model being benchmarked."""

    name: str = Field(..., description="Model name/path (e.g., meta-llama/Llama-2-7b-hf)")
    revision: str | None = Field(default=None, description="Model revision/commit hash")
    num_parameters: int = Field(..., description="Total number of parameters")
    num_layers: int = Field(0, description="Number of transformer layers")
    hidden_size: int = Field(0, description="Hidden dimension size")
    num_attention_heads: int = Field(0, description="Number of attention heads")
    vocab_size: int = Field(0, description="Vocabulary size")
    model_type: str = Field("unknown", description="Model architecture type")
    torch_dtype: str = Field("float16", description="PyTorch dtype used")
    quantization: QuantizationSpec = Field(
        default_factory=QuantizationSpec, description="Quantization config"
    )

    @property
    def parameters_billions(self) -> float:
        """Number of parameters in billions."""
        return self.num_parameters / 1e9

    @property
    def is_quantized(self) -> bool:
        """Check if model is quantized."""
        return self.quantization.enabled

    @classmethod
    def from_hf_config(cls, model_name: str, config: object, **kwargs: object) -> "ModelInfo":
        """Create ModelInfo from a HuggingFace model config.

        Args:
            model_name: Model name/path.
            config: HuggingFace PretrainedConfig object.
            **kwargs: Additional fields (num_parameters, quantization, etc).

        Returns:
            ModelInfo instance.
        """
        return cls(
            name=model_name,
            num_parameters=kwargs.get("num_parameters", 0),  # type: ignore[arg-type]
            num_layers=getattr(config, "num_hidden_layers", 0),
            hidden_size=getattr(config, "hidden_size", 0),
            num_attention_heads=getattr(config, "num_attention_heads", 0),
            vocab_size=getattr(config, "vocab_size", 0),
            model_type=getattr(config, "model_type", "unknown"),
            torch_dtype=kwargs.get("torch_dtype", "float16"),  # type: ignore[arg-type]
            quantization=kwargs.get("quantization", QuantizationSpec()),  # type: ignore[arg-type]
        )
