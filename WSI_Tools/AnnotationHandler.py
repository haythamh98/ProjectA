from xml.dom import minidom
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# import matplotlib.pyplot as plt TODO: fix this error

class AnnotationHandler:
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

    def has_tumor(self, x: float, y: float):
        return self.point_has_tumor(Point(x, y))

    def point_has_tumor(self, point: Point):
        for polygon in self.polygons:
            if polygon.contains(point):
                return True
        return False

    def polygon_has_tumor(self, polygon: Polygon):
        for self_polygon in self.polygons:
            if self_polygon.intersects(polygon):
                return True
        return False

    def rectangle_has_tumor(self, x1 : float, y1: float, x2: float, y2: float,
                            x3: float, y3: float, x4: float, y4: float):
        return self.polygon_has_tumor(Polygon([Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4)]))

    def visualize(self):
        raise NotImplementedError  # TODO
        # m = self.polygons[0].exterior.xy
        # print(m)
        # x, y = self.polygons[0].exterior.xy
        # plt.plot(range(1, 10), range(1, 10))
        # plt.show()
