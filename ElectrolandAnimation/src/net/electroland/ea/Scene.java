package net.electroland.ea;

import java.awt.Image;
import java.util.Map;

import net.electroland.utils.ParameterMap;

public interface Scene {

    // configuration parameters from the properties file
    public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams);

    // whatever the Conductor decided you needed when you start an instance
    public void init(Map<String,Object> context);

    // return false when it is time for this scene to die.
    public boolean isDone();

    // return an Image for the current frame.
    public Image getFrame(int width, int height);
}