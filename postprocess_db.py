import pyclipper
import cv2
import numpy as np

class DBPostProcess:
    """
    DB (Differentiable Binarization) postprocess:
    prob_map -> binary -> contours -> score -> unclip -> minAreaRect -> boxes
    """
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.5, min_size=3):
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.max_candidates = int(max_candidates)
        self.unclip_ratio = float(unclip_ratio)
        self.min_size = int(min_size)

    @staticmethod
    def _get_mini_boxes(contour):
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)  # (4,2) float
        points = sorted(list(points), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            idx1, idx4 = 0, 1
        else:
            idx1, idx4 = 1, 0
        if points[3][1] > points[2][1]:
            idx2, idx3 = 2, 3
        else:
            idx2, idx3 = 3, 2
        box = np.array([points[idx1], points[idx2], points[idx3], points[idx4]], dtype=np.float32)
        sside = min(rect[1])
        return box, sside

    @staticmethod
    def _box_score_fast(prob_map, box4):
        h, w = prob_map.shape[:2]
        box = box4.copy()
        xmin = max(0, int(np.floor(box[:, 0].min())))
        xmax = min(w - 1, int(np.ceil(box[:, 0].max())))
        ymin = max(0, int(np.floor(box[:, 1].min())))
        ymax = min(h - 1, int(np.ceil(box[:, 1].max())))
        if xmax <= xmin or ymax <= ymin:
            return 0.0

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)

        crop = prob_map[ymin:ymax + 1, xmin:xmax + 1]
        return float(cv2.mean(crop, mask)[0])

    def _unclip(self, box4):
        # distance = area * unclip_ratio / perimeter
        area = float(cv2.contourArea(box4.astype(np.float32)))
        peri = float(cv2.arcLength(box4.astype(np.float32), True))
        if peri < 1e-6 or area < 1e-6:
            return None

        distance = area * self.unclip_ratio / peri
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box4.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        expanded = max(expanded, key=lambda x: cv2.contourArea(np.array(x, dtype=np.float32)))
        return np.array(expanded, dtype=np.float32).reshape(-1, 1, 2)

    def __call__(self, prob_map):
        """
        prob_map: float32 [H,W]
        return: boxes (N,4,2) float32, scores (N,)
        """
        h, w = prob_map.shape[:2]
        binary = (prob_map > self.thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[: self.max_candidates]

        boxes = []
        scores = []

        for cnt in contours:
            box, sside = self._get_mini_boxes(cnt)
            if sside < self.min_size:
                continue

            score = self._box_score_fast(prob_map, box)
            if score < self.box_thresh:
                continue

            expanded = self._unclip(box)
            if expanded is None:
                continue

            box2, sside2 = self._get_mini_boxes(expanded)
            if sside2 < (self.min_size + 2):
                continue

            box2[:, 0] = np.clip(box2[:, 0], 0, w - 1)
            box2[:, 1] = np.clip(box2[:, 1], 0, h - 1)

            boxes.append(box2.astype(np.float32))
            scores.append(float(score))

        if len(boxes) == 0:
            return np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        return np.stack(boxes, axis=0), np.array(scores, dtype=np.float32)