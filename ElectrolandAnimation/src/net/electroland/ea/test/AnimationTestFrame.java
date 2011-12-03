package net.electroland.ea.test;

import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;

public class AnimationTestFrame extends JFrame{

    /**
     * 
     */
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
        // play a clip in the top/left quadrant
        anim.startClip("testClip", new Rectangle(0,0,w,h), 1.0);
        
        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        while (true){
            f.getGraphics().drawImage(anim.getStage(), 0, 0, f.getWidth(), f.getHeight(), null);
        }
    }
}