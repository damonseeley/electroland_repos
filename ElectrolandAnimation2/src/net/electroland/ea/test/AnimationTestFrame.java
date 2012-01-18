package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
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
        anim.setStageColor(Color.BLACK);
        int w = anim.getStageDimensions().width;
        int h = anim.getStageDimensions().height;

        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getStageDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        Clip stage = anim.addClip(new SolidColorContent(null), 0, 0, w, h, 1.0);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip trill1 = stage.addClip(simpleClip2, 0,              0, w / 4, h, 1.0, 0);
        Clip trill2 = stage.addClip(simpleClip2, w/4,            0, w / 4, h, 1.0, 170);
        Clip trill3 = stage.addClip(simpleClip2, w/2,            0, w / 4, h, 1.0, 375);
        Clip trill4 = stage.addClip(simpleClip2, (int)(w *.75),  0, w / 4, h, 1.0, 530);

        //fade em all out
        Change fadeOut = new LinearChange().alphaTo(0);
        trill1.delay(3500).queueChange(fadeOut, 500).delete();
        trill2.delay(3500).queueChange(fadeOut, 500).delete();
        trill3.delay(3500).queueChange(fadeOut, 500).delete();
        trill4.delay(3500).queueChange(fadeOut, 500).delete();

        

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