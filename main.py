import cv2
import insightface
from threading import Thread
from tqdm import tqdm
import onnxruntime as rt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='face', default="face.jpg")
parser.add_argument('-t', '--target', help='replace this face. If camera, use integer like 0',default=0, dest='target_path')
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 8
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
face_swapper = insightface.model_zoo.get_model("inswapper_128.onnx", session_options=sess_options)
face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
face_analyser.prepare(ctx_id=0, det_size=(640, 640))
try:
    input_face = cv2.imread(args['face'])
    source_face = sorted(face_analyser.get(input_face), key=lambda x: x.bbox[0])[0]
except:
    print("You forgot to add the input face")
    exit()
cap = cv2.VideoCapture(args['target_path'])
def face_analyser_thread(frame):
    faces = face_analyser.get(frame)
    for face in faces:
        frame = face_swapper.get(frame, face, source_face, paste_back=True)    
    return frame
with tqdm() as progressbar:
    temp = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame,)))
        temp[-1].start()
        while len(temp) >= 3:
            frame = temp.pop(0).join()
        cv2.imshow('Face Detection', frame)
        progressbar.update(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()