package net.electroland.ea;

import java.awt.Dimension;
import java.awt.Image;
import java.util.Hashtable;
import java.util.Map;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;

public class AnimationManager {

    private Dimension stageDim;
    private int fps = 33;
    Map<String, Scene> scenes;
    Map<String, Transition> transitions;

    public void setContext(Map<String, Object> context)
    {
        
    }
    public Map<String, Object> getContext()
    {
        return null;
    }

    public void addSceneListener(SceneListener sl)
    {
        
    }
    
    /************* Thread management ********************/
    public void setDesiredFPS()
    {
        
    }
    public int getMeasuredFPS()
    {
        return 0;
    }
    public void start()
    {
        
    }
    public void stop()
    {
        
    }

    /************************* Scene management ******************************/
    /**
     * 
     * @param scene -
     * @param t - Transition to use (applied between the new image and the prior 
     *        scene).  Can be null for "apply immediately".
     * @param duration -1 for infinite (or until the scene kills itself)
     * @param delay milliseconds to wait BEFORE running this animation
     * @return
     */
    public int startScene(String scene, String transition, long duration, long delay)
    {
        return 0;
    }

    /** Transition to a static color.  e.g, for fade out. **/
    public int transitionOut(String hexColor, String transition, long duration, long delay)
    {
        return 0;
    }

    /** IMMEDIATELY kill a scene based on it's ID. **/
    public Scene killScene(int i)
    {
        return null;
    }

    /************************** Stage managment *******************************/
    public Dimension getStageDimensions()
    {
        return null;
    }
    public void setStageDimensions(Dimension d)
    {
        
    }
    public Image getStage()
    {
        return null;
    }
    public int[] getStagePixels(int x, int y, int width, int height)
    {
        return null;
    }

    /************************ parse props *************************************/
    public void config(String filename)
    {
        ElectrolandProperties p = new ElectrolandProperties(filename);

        // fps, stage width & height
        Integer fpsProp = p.getOptionalInt("settings", "global", "fps");
        this.fps = fpsProp == null ? this.fps : fpsProp;

        Integer widthProp = p.getOptionalInt("settings", "global", "width");
        if (widthProp != null)
            stageDim.width = widthProp;

        Integer heightProp = p.getOptionalInt("settings", "global", "height");
        if (heightProp != null)
            stageDim.height = heightProp;

        // scenes
        scenes = new Hashtable<String,Scene>();
        Map<String, ParameterMap> sceneParams = p.getObjects("scene");
        for (String s : sceneParams.keySet()){
            ParameterMap universalParams = sceneParams.get(s);
            Map<String, ParameterMap> extendedParams = p.getObjects(s);

            Scene scene = (Scene)(universalParams.getRequiredClass("class"));
            scene.config(universalParams, extendedParams);
            scenes.put(s, scene);
        }

        // transitions
        transitions = new Hashtable<String, Transition>();
        Map<String, ParameterMap> transitionParams = p.getObjects("transition");
        for (String s : transitionParams.keySet()){
            ParameterMap universalParams = transitionParams.get(s);
            Map<String, ParameterMap> extendedParams = p.getObjects(s);

            Transition transition = (Transition)(universalParams.getRequiredClass("class"));
            transition.config(universalParams, extendedParams);
            transitions.put(s, transition);
        }
    }
}