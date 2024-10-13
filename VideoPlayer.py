import cv2


class VideoPlayer:

    def __init__(self, video_path):
        self.capture = cv2.VideoCapture(video_path)

    def play(self):
        is_paused = False

        # Check if camera opened successfully
        if not self.capture.isOpened():
            print("Error opening video file")

        # Read until video is completed
        while True:

            # Read pressed key
            pressed_key = cv2.waitKey(25) & 0xFF

            # Press Q on keyboard to exit
            if pressed_key == ord('q'):
                break

            # Press SPACE on keyboard to stop/resume
            if pressed_key == ord(' '):
                is_paused = not is_paused

            if pressed_key == ord('r'):
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if pressed_key == ord('a'):
                self.__move_frames(-30)

            if pressed_key == ord('d'):
                self.__move_frames(30)

            if not is_paused:
                self.__read_and_show_frame()

        # When everything done, release the video capture object
        self.capture.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def __move_frames(self, frames):
        current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        # Calculate the frame number to rewind to
        new_frame = max(0, current_frame + frames)
        # Set the video to the new frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.__read_and_show_frame()

    def __read_and_show_frame(self):
        # Capture frame-by-frame
        ret, frame = self.capture.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('DeepSORT object tracking', frame)
