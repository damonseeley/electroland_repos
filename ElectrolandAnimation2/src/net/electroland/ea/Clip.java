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

public class Clip implements Cloneable{

    private boolean isRemoved = false;
    private List<Clip>children;
    private State initialState; // state when instantiated
    private State currentState; // current state
    private Queue<QueuedChange>changes; // queued changes
    private QueuedChange currentChange;
    private Content content;

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
    protected void processChanges(){
        
        Iterator<Clip> clips = children.iterator();
        while (clips.hasNext()){
            Clip child = clips.next();
            if (child.isRemoved)
                clips.remove();
            else
                child.processChanges();
        }

        if (currentChange != null){
            long now = System.currentTimeMillis();

            if (currentChange.started && now > currentChange.endTime)
            {
                System.out.println("passed end");
                switch(currentChange.type){
                case(QueuedChange.DELETE):
                    System.out.println("delete");
                    this.remove();
                    break;
                case(QueuedChange.DELAY):
                    break;
                case(QueuedChange.CHANGE):
                    currentState = currentChange.change.nextState(initialState, 1.0);
                    break;
                }
                currentChange = null;
            }else if (now > currentChange.startTime)
            {
                currentChange.started = true;
                switch(currentChange.type){
                case(QueuedChange.CHANGE):
                    double percentComplete = (now - currentChange.startTime) / (double)(currentChange.endTime - currentChange.startTime);
                    if (percentComplete > 1.0){
                        percentComplete = 1.0;
                    }
                    currentState = currentChange.change.nextState(initialState, percentComplete);
                    break;
                }
            }
        }else{
            if (!changes.isEmpty())
            {
                initialState = currentState;
                currentChange = changes.remove();
                currentChange.startTime = System.currentTimeMillis() + currentChange.delay;
                currentChange.endTime = currentChange.startTime + currentChange.duration;
            }
        }
        
    }
    protected BufferedImage getImage(BufferedImage stage)
    {
        int width = currentState.geometry.width;
        int height = currentState.geometry.height;

        BufferedImage substage = new BufferedImage(width,
                                                   height,
                                                   BufferedImage.TYPE_INT_ARGB);
        content.renderContent(substage);
        Graphics g = substage.getGraphics();

        for (Clip child : children)
        {
            BufferedImage childImage = child.getImage(substage);
            g.drawImage(childImage, 0, 0, null);
        }
        g.dispose();

        BufferedImage complete = Clip.createScaledAlpha(substage, 
                                          width, 
                                          height, 
                                          (float)currentState.alpha);

        Graphics g2 = stage.getGraphics();
        int left = currentState.geometry.x;
        int top = currentState.geometry.y;
        g2.drawImage(complete, left, top, width, height, null);
        g2.dispose();
        return stage;
    }
    public Clip queueChange(Change change, int duration)
    {
        System.out.println("change queued: " + change);
        QueuedChange qc = new QueuedChange();
        qc.type = QueuedChange.CHANGE; 
        qc.change = change;
        qc.duration = duration;
        changes.add(qc);
        return this;
    }
    public Clip delay(int millis)
    {
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
    public void delete()
    {
        System.out.println("delete queued");
        QueuedChange delete = new QueuedChange();
        delete.type = QueuedChange.DELETE; 
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