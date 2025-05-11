import os
import argparse
from datetime import datetime
from pathlib import Path
import torch
import logging
from detection_repoter.model import RadarDetectionModel
from detection_repoter.feature_extraction import extract_features
from detection_repoter.utils import load_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalDocumentGenerator:
    def __init__(self, model_path=None, use_auth_token=None):
        """
        Initialize the technical document generator.
        
        Args:
            model_path (str, optional): Path to local model or HuggingFace model name
            use_auth_token (str, optional): HuggingFace token for accessing gated models
        """
        self.model = RadarDetectionModel(model_name=model_path, use_auth_token=use_auth_token)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_documentation(self, image_path, output_path=None):
        """Generate technical documentation for the model and results."""
        if output_path is None:
            output_path = f"technical_doc_{self.timestamp}.md"
            
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Load and process image
            image = load_image(image_path)
            detection_results = self.model.detect(image)
            features = extract_features(detection_results)
            
            # Generate documentation sections
            doc_sections = {
                "introduction": self._generate_introduction(),
                "project_structure": self._generate_project_structure(),
                "technical_impl": self._generate_technical_implementation(detection_results),
                "implementation_details": self._generate_implementation_details(features),
                "usage_guide": self._generate_usage_guide(),
                "best_practices": self._generate_best_practices(),
                "troubleshooting": self._generate_troubleshooting_guide()
            }
            
            # Combine all sections
            documentation = self._combine_documentation(doc_sections)
            
            # Save documentation
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
                
            logger.info(f"Documentation generated successfully at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            raise
    
    def _generate_introduction(self):
        return """# PaliGemma Multi-Task Fine-tuning Model Technical Documentation

## I. Introduction

PaliGemma is a vision-language model that combines SigLIP-So400m (image encoder) and Gemma-2B (text decoder). This document focuses on implementing joint fine-tuning for both object detection and multimodal tasks."""

    def _generate_project_structure(self):
        return """## II. Project Structure

### 1. Key Components

- Object Detection Module
- Multimodal Processing Module
- Joint Training Pipeline
- Evaluation Tools"""

    def _generate_technical_implementation(self, detection_results):
        # Format detection results into technical documentation
        implementation = """## III. Technical Implementation

### 1. Detection Results

```python
# Detection Results Summary
"""
        # Add detection results details
        implementation += f"detection_confidence = {detection_results['scores'].tolist()}\n"
        implementation += f"detection_boxes = {detection_results['boxes'].tolist()}\n"
        implementation += f"detection_labels = {detection_results['labels'].tolist()}\n"
        implementation += "```"
        return implementation

    def _generate_implementation_details(self, features):
        return f"""## IV. Implementation Details

### 1. Feature Analysis

```python
# Extracted Features
{features}
```"""

    def _generate_usage_guide(self):
        return """## V. Usage Guide

### 1. Basic Usage

```python
# Initialize components
model = RadarDetectionModel()
processor = JointDataProcessor(config)

# Process image
results = model.detect(image)
```"""

    def _generate_best_practices(self):
        return """## VI. Best Practices and Optimization

### 1. Memory Management

- Use gradient checkpointing
- Implement mixed precision training
- Optimize batch size based on available GPU memory"""

    def _generate_troubleshooting_guide(self):
        return """## VII. Troubleshooting Guide

### Common Issues and Solutions

1. Memory Issues
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training"""

    def _combine_documentation(self, sections):
        """Combine all documentation sections into a single string."""
        return "\n\n".join([
            sections["introduction"],
            sections["project_structure"],
            sections["technical_impl"],
            sections["implementation_details"],
            sections["usage_guide"],
            sections["best_practices"],
            sections["troubleshooting"]
        ])


def main():
    parser = argparse.ArgumentParser(description="Generate technical documentation from model analysis")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", help="Path for the output documentation file")
    parser.add_argument("--model", help="Path to local model or HuggingFace model name")
    parser.add_argument("--token", help="HuggingFace authentication token")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator with authentication if provided
        generator = TechnicalDocumentGenerator(
            model_path=args.model,
            use_auth_token=args.token
        )
        
        # Generate documentation
        doc_path = generator.generate_documentation(
            image_path=args.image,
            output_path=args.output
        )
        
        logger.info(f"Documentation generated at: {doc_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main() 