import cv2 as cv2
from modules.filters import draw_dot
import pathlib
import sys
import tempfile
import subprocess
import os

if __name__ == "__main__":
    if len(sys.argv) == 4:
        capture = cv2.VideoCapture(sys.argv[1])
        
        with open(sys.argv[2]) as f:
            predictions = f.read().splitlines()

        outpath = pathlib.Path(sys.argv[3])
        outpath.parent.mkdir(parents=True, exist_ok=True)

        fd, tempfile = tempfile.mkstemp(".mp4")
        os.close(fd)
        tempout = None

        ret = True
        count = 0
        while True:
            ret, frame = capture.read()

            if ret:
                color = (0,0,0)
                if predictions[count] == "0":
                    color = (0,0,255)
                elif predictions[count] == "1":
                    color = (0,255,0)
                    
                frame = draw_dot(frame, color)

                if tempout is None:
                    tempout = cv2.VideoWriter(tempfile, cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

                tempout.write(frame)
            else:
                break

            count += 1

        capture.release()
        tempout.release()

        try:
            cmd = [ 'ffmpeg', '-i', tempfile, '-i', sys.argv[1], '-y', '-hide_banner', '-loglevel', 'error', '-nostats', 
                    '-map', '0:0', '-map', '1:1', '-c:v', 'copy', '-c:a', 'copy', '-shortest', sys.argv[3] ]
            returned_output = subprocess.call(cmd)
            os.remove(tempfile)
        except:
            pathlib.Path(tempfile).replace(sys.argv[3])

    else:
        print("Usage: visualize.py <video> <predictions> <output>")