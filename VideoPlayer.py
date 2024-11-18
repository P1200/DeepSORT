import cv2


class VideoPlayer:

    def __init__(self):
        self.length = 0
        self.current_index = 0
        self.capture = None

    def play(self, capture):
        self.capture = capture
        is_paused = False
        self.length = self.capture.__len__()

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
                self.current_index = 0

            if pressed_key == ord('a'):
                self.__move_frames__(-30)

            if pressed_key == ord('d'):
                self.__move_frames__(30)

            if not is_paused:
                self.__read_and_show_frame__()

        # Closes all the frames
        cv2.destroyAllWindows()

    def __move_frames__(self, frames):
        self.current_index += frames

    def __read_and_show_frame__(self):
        if self.current_index != self.length - 1:
            cv2.imshow('DeepSORT object tracking', self.capture[self.current_index])
            self.current_index += 1
