import insightface
import core.globals

FACE_ANALYSER = None


def get_face_analyser(det_thresh=0.5):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
    return FACE_ANALYSER


def get_face(img_data):
    faces = get_face_analyser().get(img_data)
    try:
        return sorted(faces, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None


def get_faces(img_data):
    faces = get_face_analyser().get(img_data)
    try:
        return sorted(faces, key=lambda x: x.bbox[0])
    except IndexError:
        return None
