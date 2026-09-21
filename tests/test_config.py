import pytest
from unittest.mock import MagicMock, patch
from src.config import AppConfig, ModelConfig, ModelType
from src.utils import get_logger

@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger."""
    return MagicMock(spec=get_logger())

@pytest.fixture
def mock_config(mock_logger):
    """Fixture to create a mock configuration object."""
    return AppConfig(
        model_config=ModelConfig(
            model_type=ModelType.OPENROUTER,
            "***REDACTED***",
            model_name="test_model",
            max_tokens=100,
            temperature=0.7
        ),
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello!",
        logger=mock_logger
    )

def test_init(mock_config, mock_logger):
    """Test the initialization of the AppConfig."""
    assert mock_config.model_config.model_type == ModelType.OPENROUTER
    assert mock_config.model_config."***REDACTED***"
    assert mock_config.model_config.model_name == "test_model"
    assert mock_config.model_config.max_tokens == 100
    assert mock_config.model_config.temperature == 0.7
    assert mock_config.system_prompt == "You are a helpful assistant."
    assert mock_config.user_prompt == "Hello!"
    assert mock_config.logger == mock_logger

def test_validate_config(mock_config):
    """Test the validation of the configuration."""
    # Test valid configuration
    mock_config.model_config."***REDACTED***"
    assert mock_config.validate_config() is None
    
    # Test invalid configuration
    mock_config.model_config."***REDACTED***"
    with pytest.raises(ValueError):
        mock_config.validate_config()
