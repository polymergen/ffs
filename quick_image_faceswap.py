"""
Quick image face swap: pick a source face photo and a target image, save the result.
Uses the same InsightFace + INSwapper pipeline as the main FastFaceSwap app.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("OMP_NUM_THREADS", "1")

import globalsz


def _default_args():
    return {
        "face": "",
        "target_path": "",
        "output": "",
        "camera_fix": False,
        "resolution": "1920x1080",
        "threads": "1",
        "image": True,
        "cli": True,
        "face_enhancer": "none",
        "no_faceswap": False,
        "experimental": False,
        "nocuda": False,
        "lowmem": False,
        "batch": "",
        "extract_output": "",
        "codeformer_fidelity": 0.1,
        "alpha": 1.0,
        "alpha2": 1.0,
        "codeformer_skip_if_no_face": False,
        "codeformer_face_upscale": False,
        "codeformer_background_enhance": False,
        "codeformer_upscale": 1,
        "selective": "",
        "optimization": "fp32",
        "fastload": False,
        "bbox_adjust": "50x50x50x50",
        "vcam": False,
        "apple": False,
        "occluder": False,
        "rembg": False,
        "advanced_search": False,
        "grim": False,
    }


globalsz.args = _default_args()
globalsz.lowmem = globalsz.args["lowmem"]
w, h = globalsz.args["resolution"].split("x")
globalsz.width, globalsz.height = int(w), int(h)

import cv2
from utils import (
    prepare_swappers_and_analysers,
    get_faces_adaptive_det_size,
    imread_bgr_with_exif,
)


def _primary_face(analyser, image_path):
    img = imread_bgr_with_exif(image_path)
    if img is None:
        raise ValueError(f"Could not read image:\n{image_path}")
    faces = get_faces_adaptive_det_size(analyser, img)
    if not faces:
        raise ValueError(f"No face detected in:\n{image_path}")
    return sorted(faces, key=lambda x: x.bbox[0])[0]


def _swap_all_faces(swapper, analyser, source_face, target_path, output_path):
    frame = imread_bgr_with_exif(target_path)
    if frame is None:
        raise ValueError(f"Could not read target image:\n{target_path}")
    faces = get_faces_adaptive_det_size(analyser, frame)
    if not faces:
        raise ValueError(f"No face detected in target image:\n{target_path}")
    for face in sorted(faces, key=lambda x: x.bbox[0]):
        frame = swapper.get(frame, face, source_face, paste_back=True)
    if not cv2.imwrite(output_path, frame):
        raise IOError(f"Could not write output file:\n{output_path}")
    return output_path


def _reveal_folder(path):
    folder = os.path.normpath(os.path.dirname(path))
    if sys.platform == "win32":
        os.startfile(folder)
    elif sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", folder], check=False)
    else:
        import subprocess
        subprocess.run(["xdg-open", folder], check=False)


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        messagebox.showinfo(
            "Quick Face Swap",
            "Step 1 of 3:\nChoose the SOURCE photo (the face you want to use).",
        )
        source_path = filedialog.askopenfilename(
            parent=root,
            title="Source face image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not source_path:
            return

        messagebox.showinfo(
            "Quick Face Swap",
            "Step 2 of 3:\nChoose the TARGET image (faces here will be replaced).",
        )
        target_path = filedialog.askopenfilename(
            parent=root,
            title="Target image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not target_path:
            return

        base, ext = os.path.splitext(target_path)
        default_name = os.path.basename(f"{base}_faceswap{ext or '.png'}")
        messagebox.showinfo(
            "Quick Face Swap",
            "Step 3 of 3:\nChoose where to save the result.",
        )
        output_path = filedialog.asksaveasfilename(
            parent=root,
            title="Save swapped image as",
            initialfile=default_name,
            initialdir=os.path.dirname(target_path),
            defaultextension=ext or ".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not output_path:
            return

        print("Loading face models (first run can take a minute)...")
        face_swappers, face_analysers = prepare_swappers_and_analysers(globalsz.args)
        if not face_swappers or face_swappers[0] is None:
            raise RuntimeError("Face swapper failed to load.")

        print("Detecting source face...")
        source_face = _primary_face(face_analysers[0], source_path)

        print("Swapping faces on target...")
        _swap_all_faces(
            face_swappers[0], face_analysers[0], source_face, target_path, output_path
        )
        print(f"Saved: {output_path}")

        open_folder = messagebox.askyesno(
            "Done",
            f"Face swap saved to:\n{output_path}\n\nOpen the folder containing the file?",
        )
        if open_folder:
            _reveal_folder(output_path)
    except Exception as exc:
        messagebox.showerror("Quick Face Swap – Error", str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
