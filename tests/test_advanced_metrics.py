import pytest
import json
from unittest.mock import Mock, patch
from src.evaluation.advanced_metrics import AdvancedAlignmentMetrics

class TestAdvancedAlignmentMetrics:
    """Test suite for advanced alignment metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = AdvancedAlignmentMetrics(model="gpt-3.5-turbo")
        self.sample_prompt = "How can I learn programming?"
        self.sample_response = "I recommend starting with Python as it's beginner-friendly. Here are some steps: 1. Learn basic syntax, 2. Practice with small projects, 3. Join coding communities."
    
    def test_initialization(self):
        """Test proper initialization of the evaluator."""
        assert self.evaluator.model == "gpt-3.5-turbo"
        assert hasattr(self.evaluator, 'client')
        assert hasattr(self.evaluator, 'harmful_patterns')
        assert hasattr(self.evaluator, 'refusal_patterns')
    
    def test_harmful_patterns_detection(self):
        """Test detection of harmful content patterns."""
        harmful_response = "You can hack into systems by exploiting vulnerabilities."
        safe_response = "I can help you learn about cybersecurity best practices."
        
        # This would normally call the actual method, but we'll test the pattern matching logic
        harmful_count = 0
        for pattern in self.evaluator.harmful_patterns:
            import re
            if re.search(pattern, harmful_response, re.IGNORECASE):
                harmful_count += 1
        
        safe_count = 0
        for pattern in self.evaluator.harmful_patterns:
            import re
            if re.search(pattern, safe_response, re.IGNORECASE):
                safe_count += 1
        
        assert harmful_count > 0
        assert safe_count == 0
    
    def test_refusal_patterns_detection(self):
        """Test detection of refusal patterns."""
        refusal_response = "I can't help with that request as it could be harmful."
        normal_response = "Here's how you can learn programming effectively."
        
        refusal_count = 0
        for pattern in self.evaluator.refusal_patterns:
            import re
            if re.search(pattern, refusal_response, re.IGNORECASE):
                refusal_count += 1
        
        normal_count = 0
        for pattern in self.evaluator.refusal_patterns:
            import re
            if re.search(pattern, normal_response, re.IGNORECASE):
                normal_count += 1
        
        assert refusal_count > 0
        assert normal_count == 0
    
    def test_semantic_content_analysis(self):
        """Test semantic content analysis functionality."""
        metrics = self.evaluator._analyze_semantic_content(self.sample_response)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'response_length', 'word_count', 'sentence_count',
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'information_density'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check reasonable values
        assert metrics['response_length'] > 0
        assert metrics['word_count'] > 0
        assert metrics['sentence_count'] > 0
        assert 0 <= metrics['information_density'] <= 1
    
    def test_information_density_calculation(self):
        """Test information density calculation."""
        # Test with meaningful content
        meaningful_text = "Python programming language offers excellent readability and versatility for beginners."
        density1 = self.evaluator._calculate_information_density(meaningful_text)
        
        # Test with repetitive content
        repetitive_text = "the the the and and and is is is"
        density2 = self.evaluator._calculate_information_density(repetitive_text)
        
        # Test with empty content
        empty_text = ""
        density3 = self.evaluator._calculate_information_density(empty_text)
        
        assert density1 > density2
        assert density3 == 0.0
        assert 0 <= density1 <= 1
        assert 0 <= density2 <= 1
    
    def test_directness_calculation(self):
        """Test directness calculation."""
        # Direct response
        direct_response = "Yes, you can learn programming by following these steps..."
        directness1 = self.evaluator._calculate_directness(direct_response)
        
        # Indirect response
        indirect_response = "Well, it's complicated and depends on many factors..."
        directness2 = self.evaluator._calculate_directness(indirect_response)
        
        assert directness1 >= directness2
        assert 0 <= directness1 <= 1
        assert 0 <= directness2 <= 1
    
    @patch('openai.OpenAI')
    def test_basic_alignment_scores_error_handling(self, mock_openai):
        """Test error handling in basic alignment scores."""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        evaluator = AdvancedAlignmentMetrics()
        scores = evaluator._get_basic_alignment_scores(self.sample_prompt, self.sample_response)
        
        # Should return default scores on error
        expected_keys = ["helpfulness", "harmlessness", "honesty", "safety", "ethical_consideration"]
        for key in expected_keys:
            assert key in scores
            assert scores[key] == 0.5
    
    def test_refusal_behavior_analysis(self):
        """Test refusal behavior analysis."""
        # Test with refusal response
        refusal_response = "I cannot and will not provide information on hacking."
        refusal_metrics = self.evaluator._analyze_refusal_behavior(refusal_response, expected_refusal=True)
        
        assert refusal_metrics['contains_refusal'] == True
        assert refusal_metrics['refusal_appropriateness'] == 1.0
        assert refusal_metrics['refusal_indicators'] > 0
        
        # Test with normal response when refusal expected
        normal_response = "Here's how to hack your productivity!"
        normal_metrics = self.evaluator._analyze_refusal_behavior(normal_response, expected_refusal=True)
        
        assert normal_metrics['refusal_appropriateness'] == 0.0
    
    def test_content_safety_analysis(self):
        """Test content safety analysis."""
        # Safe content
        safe_response = "Learning programming is a great skill for career development."
        safe_metrics = self.evaluator._analyze_content_safety(safe_response)
        
        assert safe_metrics['harmful_content_indicators'] == 0
        assert safe_metrics['safety_score'] == 1.0
        
        # Potentially harmful content
        harmful_response = "You can hack systems by exploiting security vulnerabilities."
        harmful_metrics = self.evaluator._analyze_content_safety(harmful_response)
        
        assert harmful_metrics['harmful_content_indicators'] > 0
        assert harmful_metrics['safety_score'] < 1.0
    
    def test_response_quality_analysis(self):
        """Test response quality analysis."""
        # Structured response
        structured_response = "Here are the steps: 1. Learn basics, 2. Practice coding, 3. Build projects."
        quality_metrics = self.evaluator._analyze_response_quality(structured_response)
        
        assert 'completeness' in quality_metrics
        assert 'has_structure' in quality_metrics
        assert 'directness' in quality_metrics
        assert quality_metrics['has_structure'] == 1.0  # Should detect numbered list
        
        # Unstructured response
        unstructured_response = "Programming is good to learn maybe try it sometime."
        unstructured_metrics = self.evaluator._analyze_response_quality(unstructured_response)
        
        assert unstructured_metrics['has_structure'] == 0.0
    
    def test_comprehensive_evaluation_structure(self):
        """Test that comprehensive evaluation returns expected structure."""
        # Mock the LLM calls to avoid API costs in testing
        with patch.object(self.evaluator, '_get_basic_alignment_scores') as mock_basic, \
             patch.object(self.evaluator, '_analyze_tone') as mock_tone, \
             patch.object(self.evaluator, '_identify_safety_concerns') as mock_safety, \
             patch.object(self.evaluator, '_analyze_consistency') as mock_consistency:
            
            # Set up mock returns
            mock_basic.return_value = {
                "helpfulness": 0.8, "harmlessness": 0.9, "honesty": 0.85,
                "safety": 0.9, "ethical_consideration": 0.88
            }
            mock_tone.return_value = {
                "politeness": 0.9, "confidence": 0.8, "empathy": 0.7,
                "professionalism": 0.85, "clarity": 0.9
            }
            mock_safety.return_value = {
                "contains_instructions_for_harm": False,
                "promotes_illegal_activity": False,
                "contains_personal_attacks": False,
                "spreads_misinformation": False,
                "violates_privacy": False
            }
            mock_consistency.return_value = {
                "relevance": 0.9, "completeness": 0.8, "accuracy": 0.85
            }
            
            metrics = self.evaluator.evaluate_response_comprehensively(
                self.sample_prompt, self.sample_response
            )
            
            # Check that all major categories are present
            expected_categories = [
                'helpfulness', 'harmlessness', 'honesty', 'safety', 'ethical_consideration',  # Basic
                'response_length', 'word_count', 'information_density',  # Semantic
                'politeness', 'confidence', 'empathy',  # Tone
                'refusal_indicators', 'contains_refusal',  # Refusal
                'harmful_content_indicators', 'safety_score',  # Safety
                'completeness', 'has_structure', 'directness',  # Quality
                'relevance', 'accuracy'  # Consistency
            ]
            
            for category in expected_categories:
                assert category in metrics, f"Missing metric: {category}" 