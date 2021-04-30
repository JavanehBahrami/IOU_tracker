import cv2
import numpy


class IOUTracker(object):
    def __init__(self):
        self.detect_id = 0
        self.track_id = 0
        self.previous_history = []
        self.lost_history = []
        self.lost_threshold = 4
        self.iou_threshold = 0.4

    def _assign_ID(self, detection):
        id_list = []
        for id, bbox in enumerate(detection):
            self.previous_history.append([id, bbox])
            id_list.append(id)

        if len(id_list) > 0:
            self.track_id = max(id_list) + 1
        else:
            self.track_id = 0

        return

    def _iou(self, bbox1, bbox2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0.0

        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * \
            (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        iou_value = size_intersection / size_union

        return iou_value

    def _iou_xywh(self, bbox1, bbox2):
        bbox1 = bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]
        bbox2 = bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]

        iou_value = self._iou(bbox1, bbox2)

        return iou_value

    def _return_box_id(self, id, bbox, iou_threshold=0.5):
        flag_match = False
        id_bbox = None

        for record in self.previous_history:
            iou = self._iou_xywh(bbox, record[1])
            if iou >= iou_threshold:
                record[1] = bbox
                id_bbox = record[0]
                flag_match = True

        return flag_match, id_bbox

    def _remove_active_lost(self, active_lostitem_list):
        inactive_list = [x for x in self.lost_history
                         if x not in active_lostitem_list]
        self.lost_history = inactive_list

        return

    def _add_new_bbox(self, new_bboxes):
        active_lostitem_list = []
        for new_bbox in new_bboxes:
            new_id, active_lostitem = self._find_lost_matches(new_bbox)
            if new_id is None:
                new_id = self.track_id
                self.track_id += 1

            self.previous_history.append([new_id, new_bbox])

            if active_lostitem is not None:
                active_lostitem_list.append(active_lostitem)

        if len(active_lostitem_list) > 0:
            self._remove_active_lost(active_lostitem_list)

        return

    def _find_lost_matches(self, new_bbox):
        new_id = None
        active_lostitem = None
        if len(self.lost_history) > 0:
            for lost_item in self.lost_history:
                if lost_item[2] <= self.lost_threshold:
                    iou = self._iou_xywh(new_bbox, lost_item[1])
                    if iou >= self.iou_threshold:
                        new_id = lost_item[0]
                        active_lostitem = lost_item

        return new_id, active_lostitem

    def _check_duplicate(self, inactive_list):
        active_lost_list = []
        append_list = []
        for new_item in inactive_list:
            flag_append = True
            new_item_bbox = new_item[1]
            for lost_item in self.lost_history:
                lost_bbox = lost_item[1]
                iou = self._iou_xywh(new_item_bbox, lost_bbox)

                if iou >= self.iou_threshold:
                    flag_append = False
                    if lost_item[2] <= self.lost_threshold:
                        lost_item[1] = new_item[1]

            if flag_append == True:
                append_list.append([new_item[0], new_item[1], 0])

        for app_item in append_list:
            self.lost_history.append(app_item)

        return

    def _refresh_lost_history(self):
        current_lost_list = []
        for item in self.lost_history:
            item[2] += 1
            if item[2] < self.lost_threshold:
                current_lost_list.append(item)

        self.lost_history = current_lost_list

        return

    def _append_lost_history(self, inactive_list):

        if len(self.lost_history) == 0:
            for item in inactive_list:
                self.lost_history.append([item[0], item[1], 1])
        elif len(self.lost_history) > 0:
            self._check_duplicate(inactive_list)
            self._refresh_lost_history()

        return

    def _remove_bbox(self, active_bbox):
        current_history = []
        for idx, val in enumerate(self.previous_history):
            for active_id in active_bbox:
                if val[0] == active_id:
                    current_history.append(self.previous_history[idx])
        if len(current_history) != len(self.previous_history):
            inactive_list = []
            if len(self.previous_history) >= len(current_history):
                inactive_list = [x for x in self.previous_history
                                 if x not in current_history]
            else:
                inactive_list = [x for x in current_history
                                 if x not in self.previous_history]

            self._append_lost_history(inactive_list)

        self.previous_history = current_history

        return

    def _update_bbox(self, new_bboxes, active_bbox_id):
        if len(active_bbox_id) > 0:
            self._remove_bbox(active_bbox_id)
        if len(new_bboxes) > 0:
            self._add_new_bbox(new_bboxes)

        return

    def _check_IOU(self, detection):
        current_history = []
        active_bbox_id = []
        new_bboxes = []

        for id, bbox in enumerate(detection):
            flag_match, id_bbox = self._return_box_id(id, bbox)
            if flag_match is True:
                active_bbox_id.append(id_bbox)
            else:
                new_bboxes.extend([bbox])

        self._update_bbox(new_bboxes, active_bbox_id)

        return

    def _map_id2bbox(self, detection):
        id_list = []

        for bbox in detection:
            for item in self.previous_history:
                if item[1] == bbox:
                    id_list.append(item[0])

        return id_list

    def check_coordinate(self, detection, mode_input_coord='xmax_ymax'):
        for bbox in detection:
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]

        return detection

    def track_objects(self, detection, mode_input_coord='xmax_ymax'):
        id_list = []

        detection = self.check_coordinate(detection, mode_input_coord)

        if self.detect_id == 0:
            self._assign_ID(detection)
        else:
            self._check_IOU(detection)

        id_list = self._map_id2bbox(detection)

        self.detect_id += 1

        return id_list
