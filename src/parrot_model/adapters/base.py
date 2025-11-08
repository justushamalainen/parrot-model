"""
Base adapter interface and utilities for the Parrot Model.

This module defines the base adapter interface and common functionality
that all framework-specific adapters can use, ensuring consistent behavior
across different LLM frameworks and SDKs.
"""

from parrot_model.core.base import ParrotModel
from parrot_model.core.config import ParrotConfig
from parrot_model.utils.deterministic import (
    generate_id_for_config,
    get_timestamp_for_config,
)


class BaseSDKAdapter:
    """
    Base mixin class for SDK adapters with common functionality.

    This class provides shared initialization and utility methods that are
    common across all SDK adapters (OpenAI, Anthropic, etc.). It eliminates
    code duplication by centralizing configuration handling, model initialization,
    and deterministic ID/timestamp generation.

    Attributes:
        _parrot_config: Configuration object controlling model behavior.
        _parrot_model: Internal ParrotModel instance for generating responses.

    Example:
        >>> class MySyncAdapter(BaseSDKAdapter):
        ...     def create(self, prompt: str):
        ...         # Use inherited helpers
        ...         response_id = self._generate_id(prompt, "my-model", prefix="req_")
        ...         timestamp = self._get_timestamp()
        ...         return self._parrot_model.generate(prompt)
    """

    def __init__(self, parrot_config: ParrotConfig | None = None):
        """
        Initialize the SDK adapter with configuration.

        Args:
            parrot_config: Optional configuration object. If None, uses default ParrotConfig.

        Example:
            >>> adapter = BaseSDKAdapter()
            >>> adapter = BaseSDKAdapter(ParrotConfig(max_tokens=100))
        """
        self._parrot_config = parrot_config or ParrotConfig()
        self._parrot_model = ParrotModel(config=self._parrot_config)

    def _get_timestamp(self) -> int:
        """
        Get timestamp based on deterministic configuration.

        Returns a fixed timestamp when deterministic mode is enabled,
        or the current timestamp otherwise.

        Returns:
            Unix timestamp as an integer.

        Example:
            >>> adapter = BaseSDKAdapter(ParrotConfig(deterministic=True))
            >>> adapter._get_timestamp()
            1700000000
        """
        return get_timestamp_for_config(self._parrot_config)

    def _generate_id(
        self,
        *inputs,
        prefix: str = "",
        length: int = 8,
    ) -> str:
        """
        Generate an ID based on deterministic configuration.

        Creates content-based deterministic IDs when deterministic mode is enabled,
        or random UUID-based IDs otherwise.

        Args:
            *inputs: Variable number of inputs for ID generation (e.g., prompt, model name).
            prefix: Optional prefix to add before the ID.
            length: Length of the hash/UUID portion of the ID.

        Returns:
            Generated ID string with prefix.

        Example:
            >>> adapter = BaseSDKAdapter(ParrotConfig(deterministic=True))
            >>> adapter._generate_id("test prompt", "gpt-4", prefix="chatcmpl-", length=8)
            'chatcmpl-4837e7c0'
        """
        return generate_id_for_config(
            self._parrot_config,
            *inputs,
            prefix=prefix,
            length=length,
        )
