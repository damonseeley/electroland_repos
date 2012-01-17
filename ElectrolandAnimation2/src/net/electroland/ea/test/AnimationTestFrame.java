package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.changes.DelayedInstantChange;
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
        anim.setStageColor(Color.WHITE);
        int w = anim.getStageDimensions().width / 2;
        int h = anim.getStageDimensions().height / 2;

        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // one animation clip with another nested.
        Clip one = anim.addClip(anim.getContent("slowImage"), w, h, w, h, 1.0);
        Clip two = one.addClip(anim.getContent("fastImage"), 50, 50, 100, 100, .5);
        Clip wow = two.addClip(anim.getContent("fastImage"), 50, 50, 50, 50, .5);

        // second copy  of the nested stuff(woo hoo!)
        Clip cop = (Clip)one.clone();
        anim.addClip(cop, 0, 0, w, h, 1.0);
        wow.delete();

        // red box clip
        SolidColorContent c = new SolidColorContent(Color.RED);
        Clip red = anim.addClip(c, 0, 0, w, h, 0);

        Change change0 = new DelayedInstantChange().alphaTo(1.0).xTo(w);
        Change change1 = new LinearChange().alphaTo(1.0).yTo(h);
        Change change2 = new LinearChange().alphaTo(.75).xBy(-10).yBy(-10);
        Change change3 = new LinearChange().alphaTo(0.0).scaleHeight(.5).scaleWidth(.5);
        red.queueChange(change0, 0).queueChange(change1, 1000).delay(500).queueChange(change2, 1000).queueChange(change3, 1000);
        Change change4 = new LinearChange().alphaTo(0.25).xTo(0);
        Change change5 = new LinearChange().alphaTo(1.0).xTo(w).yTo(0);
        Change change6 = new LinearChange().scaleHeight(1.25);
        // brokenChange change6 = new LinearChange().scaleHeight(.5);

        one.queueChange(change4, 0).delay(500).queueChange(change5, 750);
        one.delay(2000).queueChange(change6, 1000);

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