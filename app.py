import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Carregar o modelo treinado
model = tf.keras.models.load_model('caminho/para/seu/modelo.h5')

# Dicionário de tradução: ajuste conforme suas classes
translation_dict = {
    0: "Hello",
    1: "Thank you",
    2: "Goodbye",
    3: "Please",
    4: "Yes",
    5: "No"
    # Adicione mais classes conforme necessário
}

# Função para prever o sinal e traduzir para inglês
def predict_and_translate(image_path):
    # Carregar e pré-processar a imagem
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalização

    # Fazer a previsão
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Obter a classe com maior probabilidade

    # Traduzir para inglês
    translation = translation_dict.get(predicted_class, "Unknown sign")
    return translation

# Exemplo de uso
image_path = 'caminho/para/sinal/libras.jpg'
translation = predict_and_translate(image_path)
print(f"A tradução do sinal é: {translation}")
