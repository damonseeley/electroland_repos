package net.electroland.ea;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class AnimationManager {

    private static Logger logger = Logger.getLogger(AnimationManager.class);
    private Dimension stageDim;
    private int fps = 33;
    private Map<String, Clip> clipsPrototypes;
    private Map<String, Object> context;
    private List<ClipListener> listeners = new Vector<ClipListener>();
    private Map<Integer, Clip> liveClips = new ConcurrentHashMap<Integer, Clip>();
    private Color stageColor = Color.BLACK;
    private Image stage;
    private int id = 0;

    public void setContext(Map<String, Object> context)
    {
        this.context = context;
    }
    public Map<String, Object> getContext()
    {
        return context;
    }

    public void addClipListener(ClipListener cl)
    {
        listeners.add(cl);
    }

    private void notifyListeners(ClipEvent e)
    {
        switch(e.type){
        case(ClipEvent.STARTED):
            for (ClipListener listener : listeners)
            {
                listener.clipStarted(e);
            }
            break;
        case(ClipEvent.ENDED):
            for (ClipListener listener : listeners)
            {
                listener.clipEnded(e);
            }
            break;
        }
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
    public int startClip(String clipName, Rectangle area, double alpha)
    {
        Clip p = clipsPrototypes.get(clipName);
        if (p != null){
            Clip c = (Clip)p.clone();
            c.init(context);
            c.image = new BufferedImage(c.baseDimensions.width, 
                                        c.baseDimensions.height, 
                                        BufferedImage.TYPE_INT_ARGB);
            c.id = id;
            liveClips.put(id, c);
            c.area = area;
            c.alpha = alpha;
            c.background = this.stageColor;
            ClipEvent e = new ClipEvent(this);
            e.clipId = c.id;
            e.clip = c;
            e.type = ClipEvent.STARTED;
            notifyListeners(e);
            return id++;
        }else{
            return -1;
            
        }
    }
    /**
     * 
     * @param id - (int) Clip ID
     * @param area - (Rectangle) destination rect
     * @param clip - (Rectangle) clipping rect
     * @param alpha - (Double) destination alpha value
     * @param durationMillis - (int) change duration in milliseconds
     * @param delayMillis - (int) delay duration in milliseconds
     * @param deleteWhenDone - (boolean) kill this clip when complete?
     */

    public void queueClipChange(int id, Rectangle area, Rectangle clip, 
                            Double alpha, int durationMillis, int delayMillis, 
                            boolean deleteWhenDone)
    {
        liveClips.get(id).queueChange(area, clip, alpha, durationMillis, 
                                      delayMillis, deleteWhenDone);
    }

    public void killClip(int i)
    {
        liveClips.get(i).resetQueue();
        Clip removed = liveClips.remove(i);
        ClipEvent e = new ClipEvent(this);
        e.clipId = removed.id;
        e.clip = removed;
        e.type = ClipEvent.ENDED;
        notifyListeners(e);
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
        Graphics g = stage.getGraphics();
        g.setColor(stageColor);
        g.fillRect(0, 0, stageDim.width, stageDim.height);

        for (Clip c : liveClips.values())
        {
            if (c.isDeleted() || c.isDone()){
                killClip(c.id);
            }else{
                c.processChanges();
                c.image = c.getFrame(new BufferedImage(c.baseDimensions.width, 
                                        c.baseDimensions.height, 
                                        BufferedImage.TYPE_INT_ARGB));
                BufferedImage alpha = createScaledAlpha(c.image, 
                                                        c.area.width,
                                                        c.area.height,
                                                        (float)c.alpha);
                g.drawImage(alpha, c.area.x, c.area.y, c.area.width, c.area.height, null);
            }
        }
        g.dispose();
        return stage;
    }
    
    public static int[] toPixels(Image stage, int width, int height)
    {
        if (stage instanceof BufferedImage){
            int[] pixels = new int[width * height];
            ((BufferedImage)stage).getRGB(0, 0, width, height, pixels, 0, width);
            return pixels;
        }else{
            throw new RuntimeException("Sorry, only works with BufferedImage.");
        }
    }
    
    public void setStageColor(Color stageColor)
    {
        this.stageColor = stageColor;
        for (Clip c : liveClips.values())
        {
            c.background = stageColor;
        }
    }

    public int getFps() {
        return fps;
    }
    public void setFps(int fps) {
        this.fps = fps;
    }
    /************************ parse props *************************************/
    public void config(String filename)
    {
        logger.info("loading " + filename);
        ElectrolandProperties p = new ElectrolandProperties(filename);

        // fps, stage width & height
        this.fps = p.getRequiredInt("settings", "global", "fps");

        stageDim = new Dimension(p.getRequiredInt("settings", "global", "width"),
                                p.getRequiredInt("settings", "global", "height"));
        setStageDimensions(stageDim);

        // clip
        clipsPrototypes = new Hashtable<String,Clip>();
        Map<String, ParameterMap> clipParams = p.getObjects("clip");
        for (String s : clipParams.keySet()){
            logger.info("loading clip '" + s + "'");
            ParameterMap universalParams = clipParams.get(s);
            int width = universalParams.getRequiredInt("width");
            int height = universalParams.getRequiredInt("height");

            Map<String, ParameterMap> extendedParams = null;
            try{
                extendedParams = p.getObjects(s);
            }catch(OptionException e){
                // not a problem.  There might not be any extended params.
            }

            Clip clip = (Clip)(universalParams.getRequiredClass("class"));
            clip.baseDimensions = new Dimension(width, height);
            clip.config(universalParams, extendedParams);
            clipsPrototypes.put(s, clip);
        }
    }

    private static BufferedImage createScaledAlpha(Image image, int width, int height, float transperancy) {
        // buffer for the original (scaled)
        if (width < 1) width = 1;
        if (height < 1) height = 1;

        BufferedImage img = new BufferedImage(width, height, BufferedImage.TRANSLUCENT);
        Graphics g = img.getGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();

        // for the alpha version
        BufferedImage aimg = new BufferedImage(width, height, BufferedImage.TRANSLUCENT);
        Graphics2D g2d = aimg.createGraphics();
        g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, transperancy));
        g2d.drawImage(img, null, 0, 0);
        g2d.dispose();

        // Return the image
        return aimg;
    }
}