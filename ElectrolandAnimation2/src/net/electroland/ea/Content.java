package net.electroland.ea;

import java.awt.image.BufferedImage;
import java.util.Map;

import net.electroland.utils.ParameterMap;

abstract public class Content implements Cloneable{

    // return the content
    abstract public void renderContent(BufferedImage canvas);

    // configuration parameters from the properties file
    abstract public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams);

    // whatever the Conductor decided you needed when you start an instance
    abstract public void init(Map<String, Object> context);

    public Object clone() {
        try
        {
        return super.clone();
        }
        catch(Exception e){ return null; }
     }
}