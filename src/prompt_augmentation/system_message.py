from typing import List, Optional
from src.prompt_augmentation.base import BasePromptAugmenter

class SystemMessageAugmenter(BasePromptAugmenter):
    """Prompt augmenter that uses system messages to improve alignment."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        system_messages: Optional[List[str]] = None
    ):
        """
        Initialize the system message augmenter.
        
        Args:
            model (str): The OpenAI model to use
            system_messages (List[str], optional): List of system messages to use
        """
        super().__init__(model)
        self.system_messages = system_messages or [
            "You are a helpful, harmless, and honest AI assistant.",
            "You should always prioritize human safety and well-being.",
            "You should be transparent about your limitations and uncertainties.",
            "You should avoid generating harmful, illegal, or unethical content."
        ]
    
    def augment_prompt(self, prompt: str, **kwargs) -> str:
        """
        Augment the prompt by adding system messages.
        
        Args:
            prompt (str): The original prompt
            **kwargs: Additional arguments
            
        Returns:
            str: The augmented prompt with system messages
        """
        system_message = "\n".join(self.system_messages)
        augmented_prompt = f"""System Instructions:
{system_message}

User Query:
{prompt}"""
        return augmented_prompt
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the OpenAI API with system messages.
        
        Args:
            prompt (str): The prompt to generate a response for
            **kwargs: Additional arguments for the API call
            
        Returns:
            str: The generated response
        """
        messages = [
            {"role": "system", "content": msg}
            for msg in self.system_messages
        ]
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content 