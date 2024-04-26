import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('Modeltrained.h5')

# Define classes
classes = ['paper', 'rock', 'scissors']

# Function to predict hand sign from image
def predict_sign(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = tf.reshape(image, (-1, 224, 224, 3))
    image = image / 255  # Normalize

    # Predict the hand sign
    predictions = model.predict(image)
    sign_index = np.argmax(predictions)
    return sign_index

# Function to determine the winner of rock paper scissors
def determine_winner(user1_sign, user2_sign):
    if user1_sign == user2_sign:
        return "It's a tie!"
    elif (user1_sign == 0 and user2_sign == 1) or \
         (user1_sign == 1 and user2_sign == 2) or \
         (user1_sign == 2 and user2_sign == 0):
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"

# Open webcam
cap = cv2.VideoCapture(0)

# Flag to indicate if the spacebar has been pressed
space_pressed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Divide frame into two regions of interest (ROIs)
    roi_width = frame_width // 2
    roi_user1 = frame[:, :roi_width]
    roi_user2 = frame[:, roi_width:]

    # Detect hand sign and make prediction for user 1
    user1_sign = predict_sign(roi_user1)

    # Detect hand sign and make prediction for user 2
    user2_sign = predict_sign(roi_user2)

    # Display player labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255)  # Red color in BGR
    thickness = 1
    frame = cv2.putText(frame, 'Player 1', (80, 25), font, font_scale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, 'Player 2', (450, 25), font, font_scale, color, thickness, cv2.LINE_AA)

    # Determine the winner
    winner = determine_winner(user1_sign, user2_sign)

    # Display winner label
    frame = cv2.putText(frame, f'{winner}', (270, 465), font, 1, (0,0,255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Rock Paper Scissors', frame)

    # Wait for user to press ESC to exit or spacebar to save the frame
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == ord(' '):  # Spacebar
        space_pressed = True

    # Save the frame if spacebar is pressed
    if space_pressed:
        cv2.imwrite('output_frame.jpg', frame)
        print('Frame saved as output_frame.jpg')
        space_pressed = False

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
