import copy
import xml.etree.ElementTree as ET
import numpy as np
from models.base import MujocoXML
from utils.mjcf_utils import string_to_array, array_to_string


class MujocoObject:
    """
    Base class for all objects.

    We use Mujoco Objects to implement all objects that
        1) may appear for multiple times in a task
        2) can be swapped between different tasks

    Typical methods return copy so the caller can all joints/attributes as wanted

    Attributes:
        asset (TYPE): Description
    """

    def __init__(self):
        self.asset = ET.Element("asset")

    def get_horizontal_radius(self):
        """
        Returns scalar
        If object a,b has horizontal distance d
        a.get_horizontal_radius() + b.get_horizontal_radius() < d
        should mean that a, b has no contact

        Helps us put objects programmatically without them flying away due to
        a huge initial contact force

        Returns:
            Float: radius

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError
        # return 2

    def get_collision(self, name=None, site=False):
        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision related fields
        of this object.

        Return is a copy

        Args:
            name (None, optional): Assign name to body
            site (False, optional): Add a site (with name @name
                 when applicable) to the returned body

        Returns:
            ET.Element: body

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_site_attrib_template(self):
        """
        Returns attribs of spherical site used to mark body origin

        Returns:
            Dictionary of default site attributes
        """
        return {
            "pos": "0 0 0",
            "size": "0.006 0.006 0.006",
            "rgba": "1 0 0 1",
            "type": "sphere",
        }


class MujocoXMLObject(MujocoXML, MujocoObject):
    """
    MujocoObjects that are loaded from xml files
    """

    def __init__(self, fname):
        """
        Args:
            fname (TYPE): XML File path
        """
        MujocoXML.__init__(self, fname)

    def get_horizontal_radius(self):
        horizontal_radius_site = self.worldbody.find("./body/site[@name='horizontal_radius_site']")
        return string_to_array(horizontal_radius_site.get("pos"))[0]

    def get_collision(self, name=None, site=False):

        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            if len(geoms) == 1:
                geoms[0].set("name", name)
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(name, i))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 1 0 1"
            if name is not None:
                template["name"] = name
            collision.append(ET.Element("site", attrib=template))
        return collision
