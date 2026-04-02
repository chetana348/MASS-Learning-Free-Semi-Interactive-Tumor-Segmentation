import os
import numpy as np
import tifffile as tiff
import cv2
from glob import glob
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt


# ---------------- SHAPE MATCH ----------------
def match_shape_pad(gt, img_shape):

    gH, gW = gt.shape
    iH, iW = img_shape

    if (gH, gW) == (iH, iW):
        return gt

    new_gt = np.zeros((iH, iW), dtype=gt.dtype)

    y_offset = max((iH - gH) // 2, 0)
    x_offset = max((iW - gW) // 2, 0)

    y_end = min(y_offset + gH, iH)
    x_end = min(x_offset + gW, iW)

    gt_crop = gt[:y_end - y_offset, :x_end - x_offset]

    new_gt[y_offset:y_end, x_offset:x_end] = gt_crop

    return new_gt


# ---------------- PATHS ----------------
image_dir = r"T:\Labs\QMI\CK Data\prosx\cropped_padded\imagesTs"
gt_dir = r"T:\Labs\QMI\CK Data\prosx\cropped_padded\labelsTs"
out_dir = r"T:\Labs\QMI\CK Data\prosx\cropped_padded\scribblesTs"


os.makedirs(out_dir, exist_ok=True)

files = sorted(glob(os.path.join(gt_dir, "*.tif")))


# ---------------- PROCESS ----------------
for f in files:

    name = os.path.basename(f)
   
    out_path = os.path.join(out_dir, name)
    
    #  SKIP if already processed
    if os.path.exists(out_path):
        continue

    gt = tiff.imread(f)
    img = tiff.imread(os.path.join(image_dir, name))

    if gt.shape != img.shape:
        gt = match_shape_pad(gt, img.shape)

    mask = (gt >0).astype(np.uint8)
    scribble = np.zeros_like(mask, dtype=np.uint8)


    # ---------------- FG SCRIBBLE ----------------
    if np.any(mask):

        num_labels, labels = cv2.connectedComponents(mask)

        if num_labels > 1:
            areas = [(labels == i).sum() for i in range(1, num_labels)]
            largest = np.argmax(areas) + 1
            mask = (labels == largest).astype(np.uint8)

        ys, xs = np.where(mask > 0)

        pts = np.stack([xs, ys], axis=1)

        # -------- estimate scribble size BEFORE branch ----------
        if len(pts) > 2:
            pca_len = PCA(n_components=1)
            pca_len.fit(pts)
            proj = pts @ pca_len.components_[0]
            scribble_length = proj.max() - proj.min()
        else:
            scribble_length = 1

        # ---------------- SMALL TUMOR ----------------
        if scribble_length < 6:

            dt = distance_transform_edt(mask)
            cy, cx = np.unravel_index(np.argmax(dt), dt.shape)

            fg_scribble = np.zeros_like(mask)

            cv2.circle(fg_scribble, (cx, cy), 1, 1, -1)

            fg_scribble = fg_scribble * mask

        # ---------------- NORMAL TUMOR ----------------
        else:

            pca = PCA(n_components=2)
            pca.fit(pts)

            center = pca.mean_
            direction = pca.components_[0]

            length = max(mask.shape) * 2

            p1 = center - direction * length
            p2 = center + direction * length

            p1 = tuple(np.round(p1).astype(int))
            p2 = tuple(np.round(p2).astype(int))

            temp = np.zeros_like(mask)

            cv2.line(temp, p1, p2, 1, thickness=1)

            fg_scribble = temp * mask

    else:
        fg_scribble = np.zeros_like(mask)

    scribble[fg_scribble > 0] = 1


    # ---------------- ESTIMATE SIZE FROM SCRIBBLE ----------------
    ys, xs = np.where(fg_scribble > 0)

    if len(xs) > 1:

        pts = np.stack([xs, ys], axis=1)

        pca = PCA(n_components=1)
        pca.fit(pts)

        proj = pts @ pca.components_[0]

        scribble_length = proj.max() - proj.min()

    else:

        scribble_length = 1


    # ---------------- ADAPTIVE RING ----------------
    if scribble_length < 6:
        inner = 2
        outer = 6
    elif scribble_length < 15:
        inner = 4
        outer = 10
    else:
        inner = 6
        outer = 18


    # ---------------- RING REGION ----------------
    dist_fg = distance_transform_edt(1 - fg_scribble)

    ring = (dist_fg > inner + 1) & (dist_fg < outer)


    # ---------------- GRADIENT MAP ----------------
    img_float = img.astype(np.float32)

    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)


    # ---------------- CENTER ----------------
    ys, xs = np.where(fg_scribble > 0)

    if len(ys) == 0:
        ys, xs = np.where(mask > 0)

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))


    # ---------------- RADIAL SEARCH ----------------
    boundary = np.zeros_like(mask)

    H, W = mask.shape

    if scribble_length < 8:
        angles = np.linspace(0, 2*np.pi, 720)
    else:
        angles = np.linspace(0, 2*np.pi, 360)

    points = []

    for a in angles:
    
        dy = np.sin(a)
        dx = np.cos(a)
    
        best_val = -1
        best_y = None
        best_x = None
    
        fallback_y = None
        fallback_x = None
    
        for r in range(inner, outer):
    
            y = int(cy + dy*r)
            x = int(cx + dx*r)
    
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
    
            fallback_y = y
            fallback_x = x
    
            if not ring[y, x]:
                continue
            
            if fg_scribble[y, x] > 0:
                continue
    
            score = grad[y, x] - 0.2 * img_float[y, x]
    
            if score > best_val:
                best_val = score
                best_y = y
                best_x = x
    
        if best_y is None:
            best_y = fallback_y
            best_x = fallback_x
    
        if best_y is not None:
            points.append((best_x, best_y))
    boundary = np.zeros_like(mask)

    for i in range(len(points)-1):
        cv2.line(boundary, points[i], points[i+1], 1, 1)
    
    if len(points) > 2:
        cv2.line(boundary, points[-1], points[0], 1, 1)

    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3,3),np.uint8))
    kernel = np.ones((3,3),np.uint8)
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel)
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel)

    scribble[boundary > 0] = 2


    # ---------------- SAVE ----------------
    tiff.imwrite(os.path.join(out_dir, name), scribble.astype(np.uint8))

    print("Saved:", name)

print("\nDone.")
