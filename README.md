# Sentiment Analysis on IMDB ratings using LSTMs + TFLite
Simple prototype of using LSTMs to create a sentiment classifier based on IMDB data. 
Was implemented using tensorflow 1.15 as 2.0 had many operators not-implemented when it came to converting the model to tflite
Uses a custom lstm layer because the keras provided layer uses many control ops which are not supported while creating a tflite model
