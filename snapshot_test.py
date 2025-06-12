import gi
import sys
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst
# initialize GStreamer
Gst.init(sys.argv)

pipeline = Gst.parse_launch ("v4l2src device=/dev/video11 io-mode=dmabuf num-buffers=1 ! video/x-raw,format=NV12,width=640,height=480 ! mppjpegenc ! filesink location=" + sys.argv[1])
pipeline.set_state(Gst.State.PLAYING)
time.sleep(1)
pipeline.set_state(Gst.State.NULL)
