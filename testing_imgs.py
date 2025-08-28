import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carregamento do modelo
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.draw2ing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dicionário de labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# transformando no formato padrao jpg
imagem = input("digite o nome de sua imagem: ")
nome , *_ = imagem.rsplit('.', 1)
imagem_final = f"{nome}.jpg"





IMAGE_PATH = f'./images/{imagem}.jpg'
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Erro: Não foi possível carregar a imagem em {IMAGE_PATH}")
else:
    H, W, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Pega a primeira mão detectada para a predição
        for hand_landmarks in results.multi_hand_landmarks:
            
            # --- Correção 1: Extrai e normaliza os dados em um único loop ---
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalização e preenchimento de data_aux
            min_x = min(x_)
            min_y = min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)
            
            # --- Correção 2: Calcula a bounding box corretamente ---
            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            # --- Correção 3: Adiciona a predição ao array para o modelo ---
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Desenha os landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Desenha a bounding box e o texto
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Exibe a imagem
    cv2.imshow('image', image)
    cv2.waitKey(0)

# Limpeza
hands.close()
cv2.destroyAllWindows()