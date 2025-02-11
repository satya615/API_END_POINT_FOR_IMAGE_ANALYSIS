from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

# Load the model once at startup
processor = AutoImageProcessor.from_pretrained("Chorzy/Sentiment_Analysis")
model = AutoModelForImageClassification.from_pretrained("Chorzy/Sentiment_Analysis")

def predict_sentiment(image):
    """Predicts sentiment from an uploaded image."""
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = model.config.id2label
    
    top_index = predictions.argmax().item()
    sentiment = labels[top_index]
    confidence = predictions[0, top_index].item()

    return sentiment, confidence

@api_view(['POST'])
@parser_classes([MultiPartParser])  # Allows file uploads
def classify_image(request):
    """API endpoint to classify an uploaded image."""
    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image file provided"}, status=400)

    image_file = request.FILES['image']

    try:
        # Open image
        image = Image.open(image_file).convert("RGB")
        
        # Predict sentiment
        sentiment, confidence = predict_sentiment(image)

        return JsonResponse({"sentiment": sentiment, "confidence": confidence})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
