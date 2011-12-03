package net.electroland.ea;

import java.awt.Dimension;
import java.awt.Image;
import java.awt.Rectangle;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.utils.ParameterMap;

public abstract class Clip implements Cloneable{

    // values used to add this clip to the scene
    protected Dimension baseDimensions;
    protected Rectangle area, clip;
    protected int alpha;
    protected Image image;
    protected int id;
    private Queue<Change> changes;
    private Change currentChange;
    private boolean isDeleted = false;

    // configuration parameters from the properties file
    abstract public void config(ParameterMap primaryParams, Map<String, ParameterMap> extendedParams);

    // whatever the Conductor decided you needed when you start an instance
    abstract public void init(Map<String, Object> context);

    // return false when it is time for this scene to die.
    abstract public boolean isDone();

    // Will be passed a baseSize.width x baseSize.height image.  
    // manager is expecting to receive the same back (should properly implement
    // manager as an image observer instead)
    abstract public Image getFrame(Image image);


    public boolean isDeleted(){
        return isDeleted;
    }
    public Image getImage() {
        return image;
    }
    public int getId() {
        return id;
    }
    public Dimension getBaseDimensions()
    {
        return baseDimensions;
    }

    protected void processChanges()
    {
        if (currentChange != null)
        {
            if (currentChange.isComplete()){
                
                if (currentChange.deleteWhenDone){
                    isDeleted = true;
                }
                
                if (currentChange.started){
                    // apply the final area and alpha
                    this.alpha = currentChange.targetAlpha;
                    this.area = currentChange.targetArea;
                }
                currentChange = null;
            }else{
                if (currentChange.started){
                    // percent complete
                    double pc = (System.currentTimeMillis() - currentChange.start)
                                    /(double)currentChange.duration;
                    
                    if (currentChange.initArea != currentChange.targetArea){
                        alpha = Change.d(currentChange.initAlpha, currentChange.targetAlpha, pc);
                    }
                    if (currentChange.initAlpha != currentChange.targetAlpha){
                        area.x = Change.d(currentChange.initArea.x, currentChange.targetArea.x, pc);
                        area.y = Change.d(currentChange.initArea.y, currentChange.targetArea.y, pc);
                        area.width = Change.d(currentChange.initArea.width, currentChange.targetArea.width, pc);
                        area.height = Change.d(currentChange.initArea.height, currentChange.targetArea.height, pc);
                         
                    }
                }else{
                    if (System.currentTimeMillis() > currentChange.start){
                        // why don't you start me up! oh yeeeah
                        currentChange.initArea = new Rectangle(area);
                        currentChange.initAlpha = new Integer(alpha);
                        if (currentChange.targetArea == null)
                            currentChange.targetArea = currentChange.initArea;
                        if (currentChange.targetAlpha == null)
                            currentChange.targetAlpha = currentChange.initAlpha;
                    }
                }
            }
        }
        if (currentChange == null && !changes.isEmpty()){
            // get the next change
            currentChange = changes.remove();
            // configure it
            currentChange.start = System.currentTimeMillis() + currentChange.delay;
            currentChange.finish = currentChange.start + currentChange.duration;
        }
    }

    protected void queueChange(Rectangle area, Rectangle clip, Integer alpha, int durationMillis, int delayMillis, boolean deleteWhenDone)
    {
        if (changes == null)
        {
            resetQueue();
        }
        changes.add(new Change(area, alpha, durationMillis, delayMillis, deleteWhenDone));
    }
    protected void resetQueue()
    {
        changes = new ConcurrentLinkedQueue<Change>();
    }
    public Object clone() {
        try
        {
        return super.clone();
        }
        catch(Exception e){ return null; }
     }
}

class Change{

    long start, finish;
    int delay;
    int duration;
    Rectangle initArea, targetArea;
    int initAlpha;
    Integer targetAlpha;
    boolean started = false;
    boolean deleteWhenDone = false;

    public Change(Rectangle area, Integer alpha, int duration, int delay, boolean deleteWhenDone){
        this.targetArea = area;
        this.targetAlpha = alpha;
        this.duration = duration;
        this.delay = delay;
        this.deleteWhenDone = deleteWhenDone;
    }

    public boolean isComplete(){
        return System.currentTimeMillis() > finish;
    }
    
    public static int d(int init, int target, double complete){
        return (int)((complete * (target - init)) + init);
    }
}