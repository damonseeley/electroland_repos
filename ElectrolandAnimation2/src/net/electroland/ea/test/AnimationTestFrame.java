package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.easing.QuinticIn;
import net.electroland.ea.easing.QuinticInOut;
import net.electroland.ea.easing.QuinticOut;

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

        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // one animation clip with another nested.
        Clip one = anim.addClip(anim.getContent("stillImage"),  50, 50, 100, 100, 1.0);
        Clip two = anim.addClip(anim.getContent("stillImage"),  150, 50, 100, 100, 1.0);
        Clip thr = anim.addClip(anim.getContent("stillImage"),  250, 50, 100, 100, 1.0);

        one.queueChange(new Change(new QuinticIn()).yBy(100).scaleWidth(1), 2000).delete();
        two.queueChange(new Change(new QuinticOut()).yBy(100).scaleWidth(1), 2000).delete();
        thr.queueChange(new Change(new QuinticInOut()).yBy(100).scaleWidth(1), 2000).delete();

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