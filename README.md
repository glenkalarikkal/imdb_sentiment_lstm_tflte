# Sentiment Analysis on IMDB ratings using LSTMs + TFLite
Simple prototype of using LSTMs to create a sentiment classifier based on IMDB data. 
Was implemented using tensorflow 1.15 as 2.0 had many operators not-implemented when it came to converting the model to  tflite. Has a bug with the `Embedding` Layer provided by keras https://github.com/tensorflow/tensorflow/issues/32849
Uses a custom lstm layer because the keras provided layer uses many control ops which are not supported while creating a tflite model. Custom lstm layer taken from https://gist.github.com/yaghmr/620f2465c913f338b909d5b403174db7
