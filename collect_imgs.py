import os
import cv2

# Define o diretório para salvar os dados
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Configurações do dataset
number_of_classes = 26
dataset_size = 100

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    # Cria a pasta para a classe
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Coletando dados para a classe {}'.format(j))

    # Loop para aguardar o usuário se preparar
    while True:
        ret, frame = cap.read()
        
        # Inverte o quadro horizontalmente
        frame = cv2.flip(frame, 1)

        # Exibe a mensagem de 'Pronto?'
        cv2.putText(frame, 'Pronto? Pressione "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Coleta as imagens para o dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # Inverte o quadro horizontalmente antes de exibir e salvar
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Salva a imagem no diretório da classe
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        
        counter += 1

# Libera os recursos
cap.release()
cv2.destroyAllWindows()