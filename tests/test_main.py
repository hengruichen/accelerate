import pytest
from unittest.mock import MagicMock, patch
from src.main import main
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
            "***REDACTED***",
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
    return MagicMock(spec=ChatManager)

@pytest.fixture
def mock_app():
    """Fixture to create a mock app."""
    return MagicMock(spec=App)

@pytest.fixture
def mock_logger():
    """Fixture to create a mock logger."""
    return MagicMock(spec=get_logger())

@pytest.fixture
def mock_main(mock_config, mock_prompt_manager, mock_chat_manager, mock_app, mock_logger):
    """Fixture to create a mock main function."""
    with patch('src.main.ChatManager') as mock_chat_manager_class, \
         patch('src.main.App') as mock_app_class, \
         patch('src.main.get_logger') as mock_get_logger:
        
        # Set up the mock objects
        mock_chat_manager_class.return_value = mock_chat_manager
        mock_app_class.return_value = mock_app
        mock_get_logger.return_value = mock_logger
        
        # Call the main function
        main()
        
        # Assert that the mock objects were called correctly
        mock