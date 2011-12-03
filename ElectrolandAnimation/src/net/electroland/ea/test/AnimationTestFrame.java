package net.electroland.ea.test;

import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.ClipEvent;
import net.electroland.ea.ClipListener;

import org.apache.log4j.Logger;

public class AnimationTestFrame extends JFrame implements ClipListener{

    private static Logger logger = Logger.getLogger(AnimationTestFrame.class);
    private static final long serialVersionUID = 1L;

    public static void main(String args[])
    {
        // create a context
        Hashtable<String, Object> context = new Hashtable<String, Object>();
        // just putting some bullshit object into the context.  normally, you'd
        // put something useful in, like "sound_manager", soundManager
        context.put("random_rectangle", new Rectangle());

        // create an AnimationManager
        AnimationManager anim = new AnimationManager();
        anim.setContext(context);
        anim.config(args.length > 0 ? args[1] : "animation.properties");
        int w = anim.getStageDimensions().width / 2;
        int h = anim.getStageDimensions().height / 2;

        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // have the frame listen for clip events
        anim.addClipListener(f);

        // play a clip in the top/left quadrant
        int clipId = anim.startClip("testClip", new Rectangle(0,0,w,h), 1.0);

        // delay 2 seconds, and then slowly expand it to full screen
        anim.queueClipChange(clipId, new Rectangle(0,0,w*2,h*2), null, null, 1500, 1000, false);

        // delay 2 seconds, and then slowly make it the bottom left quadrant
        anim.queueClipChange(clipId, new Rectangle(w,h,w,h), null, null, 500, 1000, false);
        
        // delay 2 seconds, and then fade it out
        anim.queueClipChange(clipId, null, null, 0.0, 1000, 1500, true);

        while (true){
            f.getGraphics().drawImage(anim.getStage(), 0, 0, f.getWidth(), f.getHeight(), null);

            // should sync ELU here.

            try {
                Thread.sleep(1000/anim.getFps());
            } catch (InterruptedException e) {
                System.exit(0);
            }
        }
    }

    @Override
    public void clipEnded(ClipEvent e) {
        logger.info("clip " + e.clipId + " of type " + e.clipId + " ended.");
    }

    @Override
    public void clipStarted(ClipEvent e) {
        logger.info("clip " + e.clipId + " of type " + e.clipId + " started.");
    }
}