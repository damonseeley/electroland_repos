package net.electroland.ea;

import java.awt.Dimension;
import java.awt.Image;
import java.util.Map;

public class AnimationManager {

    private Image one, two;

    public void config(String filename)
    {
        
    }

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
}