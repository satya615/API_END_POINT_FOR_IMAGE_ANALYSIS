from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from transformers import pipeline
from PIL import Image
import torch

# Load the model pipeline once (better performance)
sentiment_pipeline = pipeline("image-classification", model="Chorzy/Sentiment_Analysis")

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_image(request):
    """Receives an image, processes it, and returns sentiment classification."""
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)

    # Load image
    image = Image.open(request.FILES['image']).convert("RGB")

    # Run inference
    results = sentiment_pipeline(image)
    top_result = results[0]  # Get the highest probability prediction

    return JsonResponse({
        'sentiment': top_result['label'],
        'confidence': top_result['score']
    })
