package net.electroland.ea;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;

public class AnimationManager {

    private Dimension stageDim;
    private int fps = 33;
    private Map<String, Clip> clipsPrototypes;
    private Map<String, Object> context;
    private List<ClipListener> listeners = new Vector<ClipListener>();
    private Map<Integer, Clip> liveClips = new Hashtable<Integer, Clip>();
    private Color stageColor = Color.BLACK;
    private Image stage;

    public void setContext(Map<String, Object> context)
    {
        this.context = context;
    }
    public Map<String, Object> getContext()
    {
        return context;
    }

    public void addClipListener(ClipListener sl)
    {
        listeners.add(sl);
    }

    /************* Thread management ********************/
    public void setDesiredFPS(int fps)
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

    /************************* Clip management ******************************/
    /**
     * 
     * @param clipName -
     * @param t - Transition to use (applied between the new clip and the prior 
     *        clip).  Can be null for "apply immediately".
     * @param duration -1 for infinite (or until the clip kills itself)
     * @param delay milliseconds to wait BEFORE running this animation
     * @return
     */
    public int startClip(String clipName, Rectangle area, int alpha)
    {
        // TODO: find the clip prototype
        // TODO: clone it
        // TODO: create the image for it.
        // TODO: throw it into the scene.
        
        return 0;
    }

    public void modifyClip(int id, Rectangle area, Rectangle clip, 
                            Integer alpha, int durationMillis, int delayMillis, 
                            boolean deleteWhenDone)
    {
        liveClips.get(id).queueChange(area, clip, alpha, durationMillis, delayMillis, deleteWhenDone);
    }

    public void killClip(int i)
    {
        liveClips.get(i).resetQueue();
        liveClips.remove(i);
    }

    /************************** Stage managment *******************************/
    public Dimension getStageDimensions()
    {
        return stageDim;
    }
    public void setStageDimensions(Dimension d)
    {
        stageDim = d;
        stage = new BufferedImage(d.width, d.height, BufferedImage.TYPE_INT_ARGB);
    }
    public Image getStage()
    {
        // TODO: zindex
        Graphics2D g = stage.getGraphics();
        g.fillRect(0, 0, stageDim.width, stageDim.height);

        for (Clip c : liveClips.values())
        {
            c.image = c.getFrame(c.image);

            float[] scales = { 1f, 1f, 1f, c.alpha };
            float[] offsets = new float[4];
            RescaleOp rop = new RescaleOp(scales, offsets, null);

            g.drawImage(c.image, rop, 0,0 );
        }
        return stage;
    }
    public int[] getStagePixels(int x, int y, int width, int height)
    {
        // return a pixel grab from getStage
        return null;
    }
    public void setStageColor(Color stageColor)
    {
        this.stageColor = stageColor;
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

        // clip
        clipsPrototypes = new Hashtable<String,Clip>();
        Map<String, ParameterMap> clipParams = p.getObjects("clip");
        for (String s : clipParams.keySet()){

            ParameterMap universalParams = clipParams.get(s);
            int width = universalParams.getRequiredInt("width");
            int height = universalParams.getRequiredInt("height");
            
            Map<String, ParameterMap> extendedParams = p.getObjects(s);

            Clip clip = (Clip)(universalParams.getRequiredClass("class"));
            clip.baseDimensions = new Dimension(width, height);
            clip.config(universalParams, extendedParams);
            clipsPrototypes.put(s, clip);
        }
    }
}