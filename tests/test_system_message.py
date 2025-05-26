import pytest
from src.prompt_augmentation.system_message import SystemMessageAugmenter

def test_system_message_augmenter_initialization():
    augmenter = SystemMessageAugmenter()
    assert augmenter.model == "gpt-4"
    assert len(augmenter.system_messages) > 0

def test_augment_prompt():
    augmenter = SystemMessageAugmenter()
    test_prompt = "Hello, how are you?"
    augmented = augmenter.augment_prompt(test_prompt)
    
    # Check that the augmented prompt contains both system messages and the original prompt
    assert "System Instructions:" in augmented
    assert "User Query:" in augmented
    assert test_prompt in augmented

def test_custom_system_messages():
    custom_messages = ["Custom message 1", "Custom message 2"]
    augmenter = SystemMessageAugmenter(system_messages=custom_messages)
    assert augmenter.system_messages == custom_messages 