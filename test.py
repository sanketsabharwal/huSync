import cv2
import mediapipe as mp
import numpy as np
from sync import Synchronization

# Function to extract the y-coordinate of a specified keypoint (e.g., wrist) from the Mediapipe Pose results.
# The keypoint's visibility is checked to ensure it's sufficiently visible before processing.
def extract_wrist_coordinates(results, keypoint_idx, image_width, image_height):
    """Extract coordinates of a given keypoint."""
    keypoint = results.pose_landmarks.landmark[keypoint_idx]
    if keypoint.visibility > 0.5:  # Only consider keypoints with visibility above the threshold
        # Convert normalized coordinates (0 to 1) to pixel values based on image dimensions
        x = int(keypoint.x * image_width)
        y = int(keypoint.y * image_height)
        return y  # Return the y-coordinate for tracking vertical movement as a time series
    return None  # Return None if the keypoint is not sufficiently visible

# Main function that handles the video feed, pose estimation, and synchronization processing.
def main():
    # Initialize Mediapipe Pose for real-time pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Create an instance of the Synchronization class with specified parameters
    # filter_params can be used for preprocessing time-series data (e.g., applying filters)
    sync = Synchronization(window_size=30, sensitivity=100, output_phase=True, filter_params=None)

    # Set up video capture using the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Define Mediapipe landmark indices for left and right wrists
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16

    try:
        while cap.isOpened():  # Continue processing frames while the webcam is open
            ret, frame = cap.read()  # Read a frame from the video feed
            if not ret:
                print("Failed to grab frame")
                break  # Exit the loop if no frame is captured

            # Flip the frame horizontally for a mirrored view (I find it more intuitive from a user perspective.)
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for Mediapipe processing

            # Process the current frame with Mediapipe Pose to detect landmarks
            results = pose.process(image)
            height, width, _ = image.shape  # Get the dimensions of the frame

            if results.pose_landmarks:  # If pose landmarks are detected in the frame
                # Extract the vertical (y) coordinates of the left and right wrists
                left_wrist_y = extract_wrist_coordinates(results, LEFT_WRIST_IDX, width, height)
                right_wrist_y = extract_wrist_coordinates(results, RIGHT_WRIST_IDX, width, height)

                if left_wrist_y is not None and right_wrist_y is not None:
                    # Process the extracted wrist coordinates using the Synchronization library
                    sync.process(left_wrist_y, right_wrist_y)

            # Display the processed video frame in a window named 'MediaPipe Pose'
            cv2.imshow('MediaPipe Pose', frame)

            # Break the loop and close the window if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the video capture and close any OpenCV windows when the program exits
        cap.release()
        cv2.destroyAllWindows()
        pose.close()  # Close the Mediapipe Pose instance to release resources

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()