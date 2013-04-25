package net.electroland.ea;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.ea.easing.Linear;

/**
 * This is where all the real magic happens.  A Clip is like an HTML Div.  It's
 * a container for Content that is assigned some section of the "stage".
 * Like a Div, a Clip can contain both its own content as well as other nested
 * Clips.
 * 
 * Clips can be manipulated in a manner similar to JQuery, using the Tween
 * interface.  You queue a Tween to a Clip using Clip.queue(...).
 * 
 * There are some syntactic sugar Tweens available, such as fadeIn(millis) and 
 * fadeOut(millis).  Also, there are some special directives like delay() and 
 * delete().
 * 
 * @author production
 *
 */
public class Clip implements Comparable<Clip>{

    private boolean             isRequestedForKill = false;
    private List<Clip>          children;
    protected ClipState           initialState;
    protected ClipState           currentState;
    private Queue<QueuedActionState>  queuedTweens;
    private QueuedActionState         tweenInProgress;
    private Object              message;
    protected Animation         animationManager;
    protected Content           content;
    public int                  zIndex = 0;
    public long                 createTime;

    public Clip(Content content, Color bgcolor, int top, int left, int width, int height, float alpha)
    {
        this.children       = Collections.synchronizedList(new ArrayList<Clip>());
        this.content        = content;
        this.initialState   = new ClipState(top, left, width, height, alpha, bgcolor);
        this.currentState   = new ClipState(top, left, width, height, alpha, bgcolor);
        queuedTweens        = new ConcurrentLinkedQueue<QueuedActionState>();
        createTime          = System.currentTimeMillis();
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
    public Clip addClip(int top, int left, int width, int height, float alpha){
        Clip newClip = new Clip(null, null, top, left, width, height, alpha);
        newClip.animationManager = this.animationManager;
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
    public Clip addClip(Color bgcolor, int top, int left, int width, int height, float alpha){
        Clip newClip = new Clip(null, bgcolor, top, left, width, height, alpha);
        newClip.animationManager = this.animationManager;
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
    public Clip addClip(Content content, int top, int left, int width, int height, float alpha){
        Clip newClip = new Clip(content, null, top, left, width, height, alpha);
        newClip.animationManager = this.animationManager;
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
    public Clip addClip(Content content, Color bgcolor, int top, int left, int width, int height, float alpha){
        Clip newClip = new Clip(content, bgcolor, top, left, width, height, alpha);
        newClip.animationManager = this.animationManager;
        children.add(newClip);
        return newClip;
    }

    /**
     * this doesn't work.
     * @param bgcolor
    public void setBackground(Color bgcolor){
        this.initialState.bgcolor = bgcolor;
    }
     */

    public Clip queue(Sequence sequence){
        for (Tween tween : sequence.sequence){
            queue(tween, tween.durationMillis);
        }
        return this;
    }

    /**
     * Request that a tweening directive be queued up.  The Tween will take
     * place over the specified duration.
     * 
     * @deprecated - use queue(Sequence sequence)
     */
    public Clip queue(Tween change, int millis)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Change duration must be > 0.");
        }
        if (change == null)
        {
            throw new RuntimeException("Attempt to queue a null Change.");
        }
        QueuedActionState qc = new QueuedActionState();
        qc.type = QueuedActionState.CHANGE; 
        qc.change = change;
        qc.duration = qc.change.durationMillis;
        queuedTweens.add(qc);
        return this;
    }
    /** when the Clip gets to this directive in the queue, just delay for the
     * specified milliseconds before moving on to the next directive in the
     * queue.
     * 
     * @param millis
     * @return
     */
    public Clip pause(int millis)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        QueuedActionState delay = new QueuedActionState();
        delay.type = QueuedActionState.DELAY; 
        delay.delay = millis;
        queuedTweens.add(delay);
        return this;
    }
    public Clip announce(Object message)
    {
        this.message = message;
        QueuedActionState state = new QueuedActionState();
        state.type = QueuedActionState.MESSAGE; 
        state.delay = 0;
        queuedTweens.add(state);
        return this;
    }
    public Clip fadeIn(int millis)
    {
        return fadeIn(millis, new Linear());
    }
    public Clip fadeIn(int millis, EasingFunction ef)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        QueuedActionState fade = new QueuedActionState();
        fade.change = new Tween(ef).alphaTo(1.0f);;
        fade.type = QueuedActionState.CHANGE; 
        fade.duration = millis;
        queuedTweens.add(fade);
        return this;
    }
    public Clip fadeOut(int millis)
    {
        return fadeOut(millis, new Linear());
    }
    public Clip fadeOut(int millis, EasingFunction ef)
    {
        if (millis < 0)
        {
            throw new RuntimeException("Delay duration must be > 0.");
        }
        QueuedActionState fade = new QueuedActionState();
        fade.change = new Tween(ef).alphaTo(0.0f);;
        fade.type = QueuedActionState.CHANGE; 
        fade.duration = millis;
        queuedTweens.add(fade);
        return this;
    }
    /** When the Clip gets to this directive in the queue, delete this
     * clip.
     */
    public void deleteWhenDone()
    {
        QueuedActionState delete = new QueuedActionState();
        delete.type = QueuedActionState.DELETE;
        queuedTweens.add(delete);
    }
    /**
     * When the Clip gets to this directive in the queue, delete all of its
     * children but leave it alone.
     */
    public void deleteChildrenWhenDone()
    {
        QueuedActionState delete = new QueuedActionState();
        delete.type = QueuedActionState.DELETE_CHILDREN;
        queuedTweens.add(delete);
    }

    protected BufferedImage getImage()
    {
        synchronized(children){
            Iterator<Clip> clips = children.iterator(); // pare deleted clips
            while (clips.hasNext()){
                Clip child = clips.next();
                if (child.isRequestedForKill)
                    clips.remove();
            }
        }

        if (tweenInProgress != null){
            long now = System.currentTimeMillis();
            if (now > tweenInProgress.startTime && 
                tweenInProgress.type == QueuedActionState.CHANGE){ // run changes

                float percentComplete = 0.0f;
                if (tweenInProgress.endTime != tweenInProgress.startTime){ // avoid div / 0
                    percentComplete = (now - tweenInProgress.startTime) / 
                            (float)(tweenInProgress.endTime - tweenInProgress.startTime);
                    // occurs when "now" overshoots
                    if (percentComplete > 1.0){
                            percentComplete = 1.0f;
                    }
                }else{
                    percentComplete = 1.0f;
                }
                currentState = tweenInProgress.change.nextFrame(initialState, percentComplete);
            }
            if (now > tweenInProgress.endTime){ // clean up completions
                switch(tweenInProgress.type){
                case(QueuedActionState.DELETE):
                    this.kill();
                    break;
                case(QueuedActionState.DELETE_CHILDREN):
                    this.killChildren();
                    break;
                case(QueuedActionState.MESSAGE):
                    animationManager.announce(message);
                    break;
                }
                tweenInProgress = null;
            }
        }
        // queue up next change
        if (tweenInProgress == null && !queuedTweens.isEmpty()){
            // get the next change from the queue
            initialState = currentState;
            tweenInProgress = queuedTweens.remove();
            tweenInProgress.startTime = System.currentTimeMillis() + tweenInProgress.delay;
            tweenInProgress.endTime = tweenInProgress.startTime + tweenInProgress.duration;
        }

        // subsection of the parent that we occupy
        BufferedImage clipImage = new BufferedImage(currentState.geometry.width,
                                                    currentState.geometry.height,
                                                    BufferedImage.TRANSLUCENT);

        // render background. our content ALWAYS has a lower z-index than our children
        if (currentState.bgcolor != null){
            Graphics g = clipImage.getGraphics();
            g.setColor(currentState.bgcolor);
            g.fillRect(0, 0, clipImage.getWidth(), clipImage.getHeight());
            g.dispose();
        }
        // render our content. our content ALWAYS has a lower z-index than our children
        if (content != null){
            content.renderContent(clipImage);
        }

        // draw each of the children on our section of the stage
        Graphics2D g = clipImage.createGraphics();

        synchronized(children){

            java.util.Collections.sort(children);
            for (Clip child : children){

                int childX = child.currentState.geometry.x;
                int childY = child.currentState.geometry.y;
                int childW = child.currentState.geometry.width;
                int childH = child.currentState.geometry.height;
                float childA = child.currentState.alpha;
                if (childA > 1)
                    childA = 1.0f;
                if (childA < 0)
                    childA = 0.0f;

                BufferedImage childImage = child.getImage();
                g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)childA));

                g.drawImage(childImage, childX, childY, childW, childH, null);
            }
        }
        g.dispose();

        return clipImage;
    }
    private void kill()
    {
        isRequestedForKill = true;
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
    public int countChildren(){
        int total = 1;
        for (Clip child : children)
        {
            total += child.countChildren();
        }
        return total;
    }
    @Override
    public int compareTo(Clip clip) {
        return this.zIndex - clip.zIndex;
    }
}