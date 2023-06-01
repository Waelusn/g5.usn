from PIL import Image
import  tensorflow.lite  as tflite
import numpy as np
from os import listdir

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
run='f'
while run=='f':
    sample_rate = 44100
    duration = 4

    # Record audio from the microphone
    print("Recording started...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording stopped.")

    # Convert audio data to spectrogram
    spectrogram, frequencies, times, _ = plt.specgram(audio_data[:, 0], Fs=sample_rate)

    # Plot the spectrogram
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Save the spectrogram as an image
    plt.savefig('C:/Users/Bruker/Documents/wael/Ny mappe (3)/0_13.png')

    # Show the spectrogram
    plt.show()


    # Preprocess images
    def preprocess(img_dir, img_file):
        img = Image.open(img_dir + '/' + img_file)
        img = img.convert('RGB')  # Convert to RGB if necessary
        img = img.resize((640, 480))  # Resize the image to match the model's input shape
        img = np.array(img)
        img = img.astype(np.float32) / 255.0  # Normalize the pixel values
        img = np.expand_dims(img, axis=0)
        return img



    # Create interpreter
    ip = tflite.Interpreter(model_path='waeltrain8')
    ip.allocate_tensors()

    # Get input/output indices
    input_index = ip.get_input_details()[0]['index']
    output_index = ip.get_output_details()[0]['index']

    # Load test images
    img_dir = 'Ny mappe (3)'
    num_correct = 0
    for img_file in listdir(img_dir):
        
        # Send image to interpreter
        img = preprocess(img_dir, img_file)
        ip.set_tensor(input_index, img)

        # Launch interpreter and get prediction
        ip.invoke()
        preds = ip.get_tensor(output_index)
        
        # Test classifications
        label = int(img_file.split('_')[0])
        if np.argmax(preds) == label:
            num_correct += 1

    # Display accuracy
    num_imgs = len(listdir(img_dir))
    print('{} matching with wake up word {}'.format(num_correct, num_imgs))
    
    if num_correct==1:
        print('Wake word recognised')

        run='wael'

    else:
        print('Wake word not recognised')

