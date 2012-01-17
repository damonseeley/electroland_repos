package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;

public class AnimationTestFrame extends JFrame{

    private static final long serialVersionUID = 1L;
    AnimationManager anim;

    public static void main(String args[])
    {
        // create a context
        Hashtable<String, Object> context = new Hashtable<String, Object>();
        // just putting some bullshit object into the context.  normally, you'd
        // put something useful in, like "sound_manager", soundManager
        context.put("random_rectangle", new Rectangle());

        // create an AnimationManager
        AnimationManager anim = new AnimationManager("animation.properties");
        int w = anim.getStageDimensions().width / 2;
        int h = anim.getStageDimensions().height / 2;

        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        Clip one = anim.addClip(anim.getContent("slowImage"), w, h, w, h, 1.0);
        // nested clip
        one.addClip(anim.getContent("fastImage"), 50, 50, 100, 100, .5);

        SolidColorContent c = new SolidColorContent(Color.RED);
        Clip two = anim.addClip(c, 0, 0, w, h, .75);
        Change change = new LinearChange().toAlpha(0.0).toLeft(w);
        two.queueChange(change, 2000);
        Change change2 = new LinearChange().toAlpha(0.25).toLeft(0);
        Change change3 = new LinearChange().toAlpha(1.0).toLeft(w);
        Change change4 = new LinearChange().scaleHeight(.5);
        
        one.queueChange(change2, 1000).delay(500).queueChange(change3, 750).queueChange(change4, 1000).delete();

        while (true){
            f.getGraphics().drawImage(anim.getStage(), 0, 0, f.getWidth(), f.getHeight(), null);
            try {
                Thread.sleep(1000/anim.getFps());
            } catch (InterruptedException e) {
                System.exit(0);
            }
        }
    }
}