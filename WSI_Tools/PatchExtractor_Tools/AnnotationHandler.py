from xml.dom import minidom
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PatchExtractor import PatchTag as MetastasisType
from enum import Enum

MICRO_THRESHHOLD = 1 # TODO: put real num, as fucntion of wsi.level?



# import matplotlib.pyplot as plt TODO: fix this error

class XMLAnnotationHandler:
    """
    ana wa7sh
    rose is lucky to have me as partner
    """

    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.polygons = []
        self._parse_xml_file()

    def _parse_xml_file(self):
        document = minidom.parse(self.xml_file_path)
        annotations = document.getElementsByTagName("Annotation")
        assert annotations is not None
        for annotation in annotations:
            coordinates = annotation.getElementsByTagName("Coordinates")
            assert coordinates is not None and len(coordinates) == 1  # TODO: check this
            cur_polygon = []
            for shape_coordinates in coordinates:
                for point in shape_coordinates.getElementsByTagName("Coordinate"):
                    x = float(point.getAttribute("X"))
                    y = float(point.getAttribute("Y"))
                    cur_polygon.append(Point(x, y))
            self.polygons.append(Polygon(cur_polygon))

    def has_metastasis(self, x: float, y: float):
        return self.point_has_metastasis(Point(x, y))

    def point_has_metastasis(self, point: Point):
        for polygon in self.polygons:
            if polygon.contains(point):
                return True
        return False

    def get_polygon_metastasis(self, polygon: Polygon):
        for self_polygon in self.polygons:
            if self_polygon.intersects(polygon):
                box = polygon.minimum_rotated_rectangle  # TODO: this is naive way and not right
                x, y = box.exterior.coords.xy
                axis = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
                major_axis = max(axis)
                if major_axis >= MICRO_THRESHHOLD:
                    return MetastasisType.MICRO
                else:
                    return MetastasisType.MACRO
        return MetastasisType.NEGATIVE

    def get_rectangle_metastasis(self, x1 : float, y1: float, x2: float, y2: float,
                            x3: float, y3: float, x4: float, y4: float):
        return self.get_polygon_metastasis(Polygon([Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4)]))

    def visualize(self):
        raise NotImplementedError  # TODO
        # m = self.polygons[0].exterior.xy
        # print(m)
        # x, y = self.polygons[0].exterior.xy
        # plt.plot(range(1, 10), range(1, 10))
        # plt.show()
