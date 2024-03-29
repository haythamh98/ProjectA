import os.path

from xml.dom import minidom
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config  import PatchTag as MetastasisType
from enum import Enum
import numpy as np

MICRO_THRESHHOLD = np.Inf # TODO: put real num



# import matplotlib.pyplot as plt TODO: fix this error

class XMLAnnotationHandler:
    """
    ana wa7sh
    rose is lucky to have me as partner
    """

    def __init__(self, xml_file_path: str,level0_real_ratio=1):
        self.xml_file_path = xml_file_path
        self.level0_real_ratio = level0_real_ratio
        self.polygons = []
        self.polygons_metastasis_tag = []
        if os.path.isfile(xml_file_path):
            self._parse_xml_file()

        else:
            pass  # negative/itc only mode

    def _parse_xml_file(self):
        document = minidom.parse(self.xml_file_path)
        annotations = document.getElementsByTagName("Annotation")
        assert annotations is not None
        # read XML file
        for annotation in annotations:
            coordinates = annotation.getElementsByTagName("Coordinates")
            assert coordinates is not None and len(coordinates) == 1  # TODO: check this
            cur_polygon = []
            for shape_coordinates in coordinates:
                for point in shape_coordinates.getElementsByTagName("Coordinate"):
                    x = float(point.getAttribute("X"))
                    y = float(point.getAttribute("Y"))
                    cur_polygon.append(Point(x, y))
            if len(cur_polygon) <= 2:
                print(f"Annotation {self.xml_file_path} file has annotation with 2 or less points, skipping ...")
                continue
            self.polygons.append(Polygon(cur_polygon))
        # diffrentiate between macro and micro
        for poly in self.polygons:
            box = poly.minimum_rotated_rectangle  # TODO: this is naive way and not right
            x, y = box.exterior.coords.xy
            axis = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
            major_axis = max(axis)
            if major_axis*self.level0_real_ratio >= MICRO_THRESHHOLD:
                self.polygons_metastasis_tag.append(MetastasisType.MICRO)
            else:
                self.polygons_metastasis_tag.append(MetastasisType.MACRO)


    def has_metastasis(self, x: float, y: float):
        return self.point_has_metastasis(Point(x, y))

    def point_has_metastasis(self, point: Point):
        for polygon in self.polygons:
            if polygon.contains(point):
                return True
        return False

    def get_polygon_metastasis(self, polygon: Polygon):
        for self_polygon_type,self_polygon in zip(self.polygons_metastasis_tag,self.polygons):
            if self_polygon.intersects(polygon):
                return self_polygon_type
        return MetastasisType.NEGATIVE

    def get_rectangle_metastasis(self, x1 : float, y1: float, x2: float, y2: float,
                            x3: float, y3: float, x4: float, y4: float):
        return self.get_polygon_metastasis(Polygon([Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4)]))


