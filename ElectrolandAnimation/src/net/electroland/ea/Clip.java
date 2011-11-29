package net.electroland.ea;

import java.awt.Image;
import java.util.Map;

import net.electroland.utils.ParameterMap;

public abstract class Clip {

    int x, width, height;

    // configuration parameters from the properties file
    abstract public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams);

    // whatever the Conductor decided you needed when you start an instance
    abstract public void init(Map<String,Object> context);

    // return false when it is time for this scene to die.
    abstract public boolean isDone();

    // return an Image for the current frame.
    abstract public Image getFrame(int width, int height);
}