from models.objects import MujocoXMLObject
from utils.mjcf_utils import xml_path_completion


class CubeObject(MujocoXMLObject):
    """
    Round Gear
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cube.xml"))


class BasePartObject(MujocoXMLObject):
    """
    Base of the assembly
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/basepart.xml"))

class CylObject(MujocoXMLObject):
    """
    Clutter
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cyl.xml"))

class Cyl2Object(MujocoXMLObject):
    """
    Clutter
    """

    def __init__(self):
        super().__init__(xml_path_completion("objects/cyl2.xml"))
