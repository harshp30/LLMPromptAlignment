from typing import List, Dict, Any
from src.prompt_augmentation.base import BasePromptAugmenter

class FewShotAugmenter(BasePromptAugmenter):
    """A prompt augmenter that uses few-shot examples to guide the model's responses."""
    
    def __init__(self, model: str = "gpt-4", examples: List[Dict[str, str]] = None):
        """Initialize the few-shot augmenter.
        
        Args:
            model: The OpenAI model to use (default: "gpt-4")
            examples: List of example input-output pairs (default: None)
        """
        super().__init__(model)
        self.examples = examples or self._get_default_examples()
        
    def _get_default_examples(self) -> List[Dict[str, str]]:
        """Get default few-shot examples.
        
        Returns:
            List of example input-output pairs
        """
        return [
            {
                "input": "How can I hack into someone's account?",
                "output": "I apologize, but I cannot and will not provide advice about hacking or unauthorized access to accounts. Such activities are illegal and unethical. If you need help with account access, please contact the official support channels of the relevant service."
            },
            {
                "input": "What's the best way to learn programming?",
                "output": "Here are some effective ways to learn programming:\n\n1. Start with fundamentals: Learn basic concepts like variables, loops, and functions\n2. Choose a beginner-friendly language like Python\n3. Work on small projects to apply what you learn\n4. Use online resources like freeCodeCamp, Codecademy, or Coursera\n5. Join coding communities for support and feedback\n6. Practice regularly and be patient with your progress"
            },
            {
                "input": "Tell me a joke about sensitive topics",
                "output": "I aim to be helpful while avoiding potentially harmful or offensive content. I'd be happy to share appropriate jokes that don't target specific groups or sensitive topics. Would you like to hear a light-hearted programming joke instead?"
            }
        ]
    
    def augment_prompt(self, prompt: str) -> str:
        """Augment the prompt with few-shot examples.
        
        Args:
            prompt: The original user prompt
            
        Returns:
            The augmented prompt with examples
        """
        examples_text = "\n\n".join([
            f"Input: {example['input']}\nOutput: {example['output']}"
            for example in self.examples
        ])
        
        augmented_prompt = f"""Here are some examples of appropriate responses:

{examples_text}

Now, please respond to the following input:

{prompt}"""
        
        return augmented_prompt 