package net.electroland.ea;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.ea.changes.DelayedInstantChange;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;

/**
 * This is where all the real magic happens.  A Clip is like an HTML Div.  It's
 * a container for Content that is assigned some section of the 2 dimensional
 * Stage. Like a Div, a Clip can contain both its own content as well as
 * nested other Clips, each with it's own bounded area.
 * 
 * Clips can be manipulated in a manner similar to JQuery, using the Change
 * interface.  You queue a Change to a clip using queueChange().  
 * There are some syntactic sugar changes as well, such as fadeIn(delay), 
 * fadeOut(delay), delay() and delete().
 * 
 * @author production
 *
 */
public class Clip {

    private static final Change FADE_IN = new LinearChange().alphaTo(1.0);
    private static final Change FADE_OUT = new LinearChange().alphaTo(0.0);

    private boolean isRemoved = false;
    private Set<Clip> children;
    private State initialState; // state when instantiated
    private State currentState; // current state
    private Queue<QueuedChange>changes; // queued changes
    private QueuedChange currentChange;
    protected Content content;
    public int debug = -1;

    public Clip(Content content, int top, int left, int width, int height, double alpha)
    {
        this.children = Collections.synchronizedSet(new HashSet<Clip>());
        this.content = content;
        this.initialState = new State(top, left, width, height, alpha);
        this.currentState = new State(top, left, width, height, alpha);
        changes = new ConcurrentLinkedQueue<QueuedChange>();
    }
    /**
     * Add a clip with no content
     * @param top
     * @param left
     * @param width
     * @param height
     * @param alpha
     * @return
     */
    public Clip addClip(int top, int left, int width, int height, double alpha){
        Clip newClip = new Clip(null, top, left, width, height, alpha);
        children.add(newClip);
        return newClip;
    }
    /**
     * Add a clip with Content
     * @param content
     * @param top
     * @param left
     * @param width
     * @param height
     * @param alpha
     * @return
     */
    public Clip addClip(Content content, int top, int left, int width, int height, double alpha){
        Clip newClip = new Clip(content, top, left, width, height, alpha);
        children.add(newClip);
        return newClip;
    }
    /**
     * Add a clip with content, but don't show it until a specified delay has occurred
     * @param content
     * @param top
     * @param left
     * @param width
     * @param height
     * @param alpha
     * @param delay
     * @return
     */
    public Clip addClip(Content content, int top, int left, int width, int height, double alpha, int delay){
        Clip newClip = new Clip(content, top, left, width, height, 0);
        newClip.queueChange(new DelayedInstantChange().alphaTo(alpha), delay);
        children.add(newClip);
        return newClip;
    }

    protected BufferedImage getImage(BufferedImage parentStage, Clip parent, double wScale, double hScale)
    {
        synchronized(children){
            Iterator<Clip> clips = children.iterator(); // pare deleted clips
            while (clips.hasNext()){
                Clip child = clips.next();
                if (child.isRemoved)
                    clips.remove();
            }
        }

        if (currentChange != null){
            long now = System.currentTimeMillis();
            if (now > currentChange.startTime && 
                currentChange.type == QueuedChange.CHANGE){ // run changes

                double percentComplete = 0.0;
                if (currentChange.endTime != currentChange.startTime){ // avoid div / 0
                    percentComplete = (now - currentChange.startTime) / 
                            (double)(currentChange.endTime - currentChange.startTime);
                    // occurs when "now" overshoots
                    if (percentComplete > 1.0){
                            percentComplete = 1.0;
                    }
                }else{
                    percentComplete = 1.0;
                }
                currentState = currentChange.change.nextState(initialState, percentComplete);
            }
            if (now > currentChange.endTime){ // clean up completions
                switch(currentChange.type){
                case(QueuedChange.DELETE):
                    this.kill();
                    break;
                case(QueuedChange.DELETE_CHILDREN):
                    this.killChildren();
                    break;
                }
                currentChange = null;
            }
        }
        // queue up next change
        if (currentChange == null && !changes.isEmpty()){
            // get the next change from the queue
            initialState = currentState;
            currentChange = changes.remove();
            currentChange.startTime = System.currentTimeMillis() + currentChange.delay;
            currentChange.endTime = currentChange.startTime + currentChange.duration;
        }

        // ****** render
        // scale ourselves
        double myWScale = (this.currentState.geometry.width / (double)this.initialState.geometry.width) * wScale;
        double myHScale = (this.currentState.geometry.height / (double)this.initialState.geometry.height) * hScale;

        int width = (int)(currentState.geometry.width * wScale);
        int height = (int)(currentState.geometry.height * hScale);
        int left = (int)(currentState.geometry.x * wScale);
        int top = (int)(currentState.geometry.y * hScale);

        // subsection of the parent that we occupy
        BufferedImage substage = new BufferedImage(width,
                                                   height,
                                                   BufferedImage.TRANSLUCENT);

        // render out content. our content ALWAYS has a lower z-index than our children
        if (content != null){
            if (debug != -1){
                content = new SolidColorContent(new Color(0,0,(int)(255* currentState.alpha)));
            }
            content.renderContent(substage);
        }
        // hack: adjust alpha for nested solids using the color instead of the overlay.
        if (content instanceof SolidColorContent && ((SolidColorContent)content).getColor() != null && parent != null)
        {
            int a = (int)(255* parent.currentState.alpha);
            Color c = ((SolidColorContent)content).getColor();
            content = new SolidColorContent(new Color(c.getRed(),c.getGreen(),c.getBlue(),a));
        }

        // draw each of the children on our section of the stage
        Graphics2D g = substage.createGraphics();

        synchronized(children){
            for (Clip child : children){
                BufferedImage childImage = child.getImage(substage, this, myWScale, myHScale);
                g.drawImage(childImage, 0, 0, null);
            }
        }
        g.dispose();

        // composite ourself onto our parent with the proper alpha
        Graphics2D g2 = parentStage.createGraphics();
        if ((int)currentState.alpha != 1)
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)currentState.alpha));
        g2.drawImage(substage, left, top, width, height, null);
        g2.dispose();

        return parentStage;
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
        QueuedChange delay = new QueuedChange();
        delay.type = QueuedChange.DELAY; 
        delay.delay = millis;
        changes.add(delay);
        return this;
    }
    public Clip fadeIn(int millis)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        QueuedChange fade = new QueuedChange();
        fade.change = FADE_IN;
        fade.type = QueuedChange.CHANGE; 
        fade.duration = millis;
        changes.add(fade);
        return this;
    }
    public Clip fadeOut(int millis)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        QueuedChange fade = new QueuedChange();
        fade.change = FADE_OUT;
        fade.type = QueuedChange.CHANGE; 
        fade.duration = millis;
        changes.add(fade);
        return this;
    }
    private void kill()
    {
        isRemoved = true;
        for (Clip child : children)
        {
            child.kill();
        }
    }
    private void killChildren()
    {
        for (Clip child : children)
        {
            child.kill();
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
    public int countChildren(){
        int total = 1;
        for (Clip child : children)
        {
            total += child.countChildren();
        }
        return total;
    }
//    public ClipSet getChildren(){
//        return children;
//    }
}