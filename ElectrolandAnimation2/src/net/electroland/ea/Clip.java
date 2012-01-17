package net.electroland.ea;

import java.awt.AlphaComposite;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.ea.changes.DelayedInstantChange;

public class Clip implements Cloneable{

    private boolean isRemoved = false;
    private List<Clip>children;
    private State initialState; // state when instantiated
    private State currentState; // current state
    private Queue<QueuedChange>changes; // queued changes
    private QueuedChange currentChange;
    protected Content content;

    public Clip(Content content, int top, int left, int width, int height, double alpha)
    {
        this.content = content;
        this.initialState = new State(top, left, width, height, alpha);
        this.currentState = new State(top, left, width, height, alpha);
        children = Collections.synchronizedList(new ArrayList<Clip>());
        changes = new ConcurrentLinkedQueue<QueuedChange>();
    }
    public Clip addClip(Content content, int top, int left, int width, int height, double alpha){
        Clip newClip = new Clip(content, top, left, width, height, alpha);
        children.add(newClip);
        return newClip;
    }
    public Clip addClip(Clip clip, int top, int left, int width, int height, double alpha){
        clip.currentState = new State(top, left, width, height, alpha);
        children.add(clip);
        return clip;
    }
    public Clip addClip(Content content, int top, int left, int width, int height, double alpha, int delay){
        Clip newClip = new Clip(content, top, left, width, height, 0);
        newClip.queueChange(new DelayedInstantChange().alphaTo(alpha), delay);
        children.add(newClip);
        return newClip;
    }
    protected void processChanges(){

        // process changes in children first.
        Iterator<Clip> clips = children.iterator();
        while (clips.hasNext()){
            Clip child = clips.next();
            // any deletions?
            if (child.isRemoved){
                clips.remove();
            }
            else{
                child.processChanges();
            }
        }

        if (currentChange != null){
            long now = System.currentTimeMillis();

            if (currentChange.started && now > currentChange.endTime)
            {
                switch(currentChange.type){
                case(QueuedChange.DELETE):
                    this.remove();
                    break;
                case(QueuedChange.DELETE_CHILDREN):
                    this.removeChildren();
                    break;
                case(QueuedChange.CHANGE):
                    // move it to final state.
                    currentState = currentChange.change.getTargetState(initialState);
                    break;
                }
                currentChange = null;
            }else if (now > currentChange.startTime)
            {
                currentChange.started = true;
                if(currentChange.type == QueuedChange.CHANGE){
                    double percentComplete = 0.0;
                    // divide by zero possibility here.
                    if (currentChange.endTime != currentChange.startTime){
                        percentComplete = (now - currentChange.startTime) / (double)(currentChange.endTime - currentChange.startTime);
                        // occurs when "now" overshoots
                        if (percentComplete > 1.0){
                                percentComplete = 1.0;
                        }
                    }else{
                        percentComplete = 1.0;
                    }
                    currentState = currentChange.change.nextState(initialState, percentComplete);
                }
            }
        }else{
            if (!changes.isEmpty())
            {
                // get the next change from the queue
                initialState = currentState;
                currentChange = changes.remove();
                currentChange.startTime = System.currentTimeMillis() + currentChange.delay;
                currentChange.endTime = currentChange.startTime + currentChange.duration;
            }
        }
        
    }
    // generates this Clip: current problem scaling isn't working because
    // children don't know how to scale x,y.
    protected BufferedImage getImage(BufferedImage stage, double wScale, double hScale)
    {
        double myWScale = (this.currentState.geometry.width / (double)this.initialState.geometry.width) * wScale;
        double myHScale = (this.currentState.geometry.height / (double)this.initialState.geometry.height) * hScale;

        int width = (int)(currentState.geometry.width * wScale);
        int height = (int)(currentState.geometry.height * hScale);
        int left = (int)(currentState.geometry.x * wScale);
        int top = (int)(currentState.geometry.y * hScale);

        BufferedImage substage = new BufferedImage(width,
                                                   height,
                                                   BufferedImage.TYPE_INT_ARGB);
        if (content != null)
        {
            content.renderContent(substage);
        }
        Graphics g = substage.getGraphics();

        for (Clip child : children)
        {
            BufferedImage childImage = child.getImage(substage, myWScale, myHScale);
            g.drawImage(childImage, 0, 0, null);
        }
        g.dispose();

        BufferedImage complete = Clip.createScaledAlpha(substage, 
                                          width, 
                                          height, 
                                          (float)currentState.alpha);

        Graphics g2 = stage.getGraphics();
        g2.drawImage(complete, left, top, width, height, null);
        g2.dispose();
        return stage;
    }
    public Clip queueChange(Change change, int duration)
    {
        if (duration < 0)
        {
            throw new RuntimeException("Change duration must be > 0.");
        }
        if (change == null)
        {
            throw new RuntimeException("Attempt to queue a null Change.");
        }
        QueuedChange qc = new QueuedChange();
        qc.type = QueuedChange.CHANGE; 
        qc.change = change;
        qc.duration = duration;
        changes.add(qc);
        return this;
    }
    public Clip delay(int millis)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        System.out.println("delay queued");
        QueuedChange delay = new QueuedChange();
        delay.type = QueuedChange.DELAY; 
        delay.delay = millis;
        changes.add(delay);
        return this;
    }
    private void remove()
    {
        isRemoved = true;
        for (Clip child : children)
        {
            child.remove();
        }
    }
    private void removeChildren()
    {
        for (Clip child : children)
        {
            child.remove();
        }
    }
    public void delete()
    {
        QueuedChange delete = new QueuedChange();
        delete.type = QueuedChange.DELETE; 
        changes.add(delete);
    }
    public void deleteChildren()
    {
        QueuedChange delete = new QueuedChange();
        delete.type = QueuedChange.DELETE_CHILDREN; 
        changes.add(delete);
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
    public Object clone() {
        try
        {
        return super.clone();
        }
        catch(Exception e){ return null; }
     }

}