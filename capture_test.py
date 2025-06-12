import threading
import queue
import time
import cv2
import numpy as np
import argparse
import json



def task_readframe(buffer_in, source):
    stream = cv2.VideoCapture(source)
    try:
        while stream.isOpened():
            ret, frame = stream.read()
            queue_in.put(frame) # put frame into queue
            time.sleep(0.1)  # simulate blocking delay
        stream.release()
    except KeyboardInterrupt:
        pass

def task_process(queue_in, queue_out):
    try:
        while True:
            # read input buffer
            frame_in = queue_in.get()

            # processing
            # still do nothing
            time.sleep(0.1)  # simulate blocking delay

            # write to output buffer after finished the process
            queue_out.put(frame_in)
    
    except KeyboardInterrupt:
        pass

def task_display(queue_out, displayflag):
    try:
        while True:
            if not queue_out.empty():
                frame = queue_out.get()
                if not frame is None:
                    if displayflag == 'true':
                        frame = cv2.resize(frame, (960, 540))
                        cv2.imshow("output frame", frame)
            # Press Q on keyboard to  exit
            cv2.waitKey(10)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()





if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-d", "--displayflag", help="display image with cv or not (true, false)", default='false')

    # Read arguments from command line
    args = parser.parse_args()

    # URL for video or camera source
    source = "/dev/video11"

    # Queue for frame buffering in image processing
    queue_in   = queue.Queue()
    queue_out  = queue.Queue()

    # config tasks
    t1 = threading.Thread(target=task_readframe, args=(queue_in, source))
    t2 = threading.Thread(target=task_process, args=(queue_in, queue_out))
    t3 = threading.Thread(target=task_display, args=(queue_out, args.displayflag))
    
    # start tasks
    t1.start()
    t2.start()
    t3.start()
    # wait for all threads to finish
    t1.join()
    t2.join()
    t3.join()
