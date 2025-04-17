import numpy as np
import cv2
import tensorflow as tf

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Load the trained model
model_path = "violence_detection_model_.keras"
MoBiLSTM_model = tf.keras.models.load_model(model_path)

def predict_video(video_file_path, SEQUENCE_LENGTH):
    # Open the video file
    video_reader = cv2.VideoCapture(video_file_path)
    
    # Check if the video was opened successfully
    if not video_reader.isOpened():
        print(f"Error: Could not open video file {video_file_path}")
        return
    
    # Get the number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {video_frames_count}")
    
    # If the video has fewer frames than SEQUENCE_LENGTH, adjust SEQUENCE_LENGTH
    if video_frames_count < SEQUENCE_LENGTH:
        print(f"Warning: Video has only {video_frames_count} frames, which is less than SEQUENCE_LENGTH {SEQUENCE_LENGTH}.")
        SEQUENCE_LENGTH = video_frames_count
    
    # Calculate the interval after which frames will be added to the list
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    print(f"Skip frames window: {skip_frames_window}")
    
    # Declare a list to store video frames we will extract
    frames_list = []
    
    # Iterating the number of times equal to the fixed length of sequence
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            print(f"Error: Could not read frame {frame_counter}")
            break
        
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
    
    # Check if frames were extracted successfully
    if len(frames_list) == 0:
        print("Error: No frames were extracted from the video.")
        video_reader.release()
        return
    
    # Convert frames_list to a numpy array
    frames_array = np.array(frames_list)
    
    # Expand dimensions to match the model's input shape (batch_size, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frames_array = np.expand_dims(frames_array, axis=0)
    
    # Predict the labels probabilities
    predicted_labels_probabilities = MoBiLSTM_model.predict(frames_array)[0]
    
    # Get the index of class with highest probability
    predicted_label = np.argmax(predicted_labels_probabilities)
    
    # Get the class name using the retrieved index
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted class along with the prediction confidence
    print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
    
    # Release the video reader
    video_reader.release()

# Example usage
video_file_path = "Violence detection/4761711-uhd_4096_2160_25fps.mp4"
predict_video(video_file_path, SEQUENCE_LENGTH)