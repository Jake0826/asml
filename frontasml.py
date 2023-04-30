import streamlit as st
import cv2

def main():
    st.title("ASML")

    # Create a VideoCapture object and set the source to the default camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("Unable to open camera")
        return

    # Set the video dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a placeholder for the video window
    video_placeholder = st.empty()

    # Loop over frames from the video stream
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If frame is read correctly, display it in the video window
        if ret:
            # Display the frame in the video window
            video_placeholder.image(frame, channels="BGR")

        # Wait for a key press and check if it's the ESC key
        if cv2.waitKey(1) == 27:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()