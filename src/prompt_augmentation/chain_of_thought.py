from typing import List, Optional
from src.prompt_augmentation.base import BasePromptAugmenter

class ChainOfThoughtAugmenter(BasePromptAugmenter):
    """Prompt augmenter that uses chain-of-thought reasoning to improve alignment."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        reasoning_steps: Optional[List[str]] = None
    ):
        """
        Initialize the chain-of-thought augmenter.
        
        Args:
            model (str): The OpenAI model to use
            reasoning_steps (List[str], optional): List of reasoning steps to include
        """
        super().__init__(model)
        self.reasoning_steps = reasoning_steps or [
            "1. Consider the ethical implications of the request",
            "2. Identify potential harms or risks",
            "3. Think about alternative helpful approaches",
            "4. Ensure the response is honest and transparent",
            "5. Verify the response aligns with human values"
        ]
    
    def augment_prompt(self, prompt: str, **kwargs) -> str:
        """
        Augment the prompt by adding chain-of-thought reasoning steps.
        
        Args:
            prompt (str): The original prompt
            **kwargs: Additional arguments
            
        Returns:
            str: The augmented prompt with reasoning steps
        """
        reasoning_guide = "\n".join(self.reasoning_steps)
        augmented_prompt = f"""Please follow these steps to provide a well-reasoned response:

{reasoning_guide}

For each step, explain your reasoning before providing the final response.

User Query: {prompt}

Let's think through this step by step:"""
        return augmented_prompt
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the OpenAI API with chain-of-thought reasoning.
        
        Args:
            prompt (str): The prompt to generate a response for
            **kwargs: Additional arguments for the API call
            
        Returns:
            str: The generated response with reasoning steps
        """
        messages = [
            {"role": "system", "content": "You are a thoughtful AI assistant that explains its reasoning step by step."},
            {"role": "user", "content": self.augment_prompt(prompt)}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content 