# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from tqdm import tqdm
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.count_human = []
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.pbar = tqdm(total=self.last_idx) 

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, tracks):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            if(track.track_id not in self.count_human):
                self.count_human.append(track.track_id)

            ID = self.count_human.index(track.track_id) + 1

        return self.count_human

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1
            self.pbar.update(1)
        self.pbar.close()


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """
    
    def __init__(self, seq_info, update_ms):
        self.count_human = []
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.pbar = tqdm(total=self.last_idx)  

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            self.pbar.close()
            return False
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        self.pbar.update(1)
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes, points):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

        for key, values in points.items():
            for track_id, x, y in values:
                self.viewer.color = create_unique_color_uchar(track_id)
                self.viewer.points(x, y)
        

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            if(track.track_id not in self.count_human):
                self.count_human.append(track.track_id)

            ID = self.count_human.index(track.track_id) + 1

            self.viewer.color = create_unique_color_uchar(ID)
            self.viewer.rectangle(*track.to_tlwh().astype(np.int), label=str(ID))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)

        return self.count_human
#