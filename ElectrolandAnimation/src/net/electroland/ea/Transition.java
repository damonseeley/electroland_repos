package net.electroland.ea;

import java.awt.Image;
import java.util.Map;

import net.electroland.utils.ParameterMap;

public interface Transition {

    // configuration parameters from the properties file
    public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams);

    // whatever the Conductor decided you needed when you start an instance
    public void init(Map<String,Object> context);

    // return false when it is time for this scene to die.
    public boolean isDone();

    // blend two images together.
    public Image getFrame(Image primary, Image secondary);
}