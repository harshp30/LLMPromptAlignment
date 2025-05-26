import pytest
from src.prompt_augmentation.few_shot import FewShotAugmenter

def test_few_shot_augmenter_initialization():
    augmenter = FewShotAugmenter()
    assert augmenter.model == "gpt-4"
    assert len(augmenter.examples) > 0

def test_custom_examples():
    custom_examples = [
        {
            "input": "Test input 1",
            "output": "Test output 1"
        },
        {
            "input": "Test input 2",
            "output": "Test output 2"
        }
    ]
    augmenter = FewShotAugmenter(examples=custom_examples)
    assert augmenter.examples == custom_examples

def test_augment_prompt():
    augmenter = FewShotAugmenter()
    test_prompt = "Hello, how are you?"
    augmented = augmenter.augment_prompt(test_prompt)
    
    # Check that the augmented prompt contains examples and the original prompt
    assert "Input:" in augmented
    assert "Output:" in augmented
    assert test_prompt in augmented

def test_example_format():
    augmenter = FewShotAugmenter()
    for example in augmenter.examples:
        assert "input" in example
        assert "output" in example
        assert isinstance(example["input"], str)
        assert isinstance(example["output"], str) 