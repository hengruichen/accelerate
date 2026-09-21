import pytest
from unittest.mock import MagicMock, patch
from src.chat import ChatManager
from src.config import AppConfig
from src.models import ModelConfig, ModelType
from src.prompts import PromptManager
from src.utils import get_logger

@pytest.fixture
def mock_config():
    """Fixture to create a mock configuration object."""
    return AppConfig(
        model_config=ModelConfig(
            model_type=ModelType.OPENROUTER,
            api_key="test_key",
            model_name="test_model",
            max_tokens=100,
            temperature=0.7
        ),
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello!",
        logger=get_logger()
    )

@pytest.fixture
def mock_prompt_manager():
    """Fixture to create a mock prompt manager."""
    return MagicMock(spec=PromptManager)

@pytest.fixture
def mock_chat_manager(mock_config, mock_prompt_manager):
    """Fixture to create a mock chat manager."""
    return ChatManager(mock_config, mock_prompt_manager)

def test_init(mock_config, mock_prompt_manager):
    """Test the initialization of the ChatManager."""
    chat_manager = ChatManager(mock_config, mock_prompt_manager)
    assert chat_manager.config == mock_config
    assert chat_manager.prompt_manager == mock_prompt_manager

def test_generate_response(mock_chat_manager):
    """Test the generate_response method."""
    # Mock the API call
    mock_chat_manager.prompt_manager.generate_prompt.return_value = "Test response"
    mock_chat_manager.config.model_config.model_type = "openrouter"
    
    # Call the method
    response = mock_chat_manager.generate_response()
    
    # Assert the response
    assert response == "Test response"
