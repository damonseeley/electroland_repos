package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.Animation;
import net.electroland.ea.AnimationListener;
import net.electroland.ea.Clip;
import net.electroland.ea.Sequence;
import net.electroland.ea.easing.CubicOut;
import net.electroland.ea.easing.Linear;
import net.electroland.ea.easing.QuinticIn;

public class AnimationTestFrame extends JFrame implements AnimationListener{

    private static final long serialVersionUID = 1L;
    static Animation anim;

    public static void main(String args[])
    {
        // create a context
        Hashtable<String, Object> context = new Hashtable<String, Object>();
        // just putting some bullshit object into the context.  normally, you'd
        // put something useful in, like "sound_manager", soundManager
        context.put("random_rectangle", new Rectangle());

        // create an Animator
        anim = new Animation("animation.properties");
        anim.setBackground(Color.WHITE);

        // configure test app window.
        AnimationTestFrame f = new AnimationTestFrame();
        f.setSize(anim.getFrameDimensions());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // we're listening for messages.
        anim.addListener(f);

        
        
        // create a couple clips
        Clip one = anim.addClip(anim.getContent("stillImage"),  50, 50, 100, 100, 1.0f);
        Clip two = anim.addClip(anim.getContent("slowImage"),  150, 50, 100, 100, 1.0f);
        Clip thr = anim.addClip(anim.getContent("fastImage"), new Color(0,255,0), 250, 50, 100, 100, 1.0f);
        Clip fur = one.addClip(Color.getHSBColor(0.9f,0.8f,1.0f), 0, 0, 100, 100, 1.0f);

        System.out.println("one: " + one);
        System.out.println("two: " + two);
        System.out.println("thr: " + thr);
        System.out.println("fur: " + fur);

        //fur.keepAlive();
        
        Sequence bounce = new Sequence(); 

                 bounce.yTo(150).yUsing(new QuinticIn()) // would be nice to make easing functions static.
                       .xBy(100).xUsing(new Linear())
                       .scaleWidth(2.0f)
                       .hueBy(0.3f).hueUsing(new Linear())
                       .saturationBy(0.2f).saturationUsing(new CubicOut())
                       .duration(1000)
                .newState()
                       .yTo(75).yUsing(new CubicOut())
                       .xBy(100).xUsing(new Linear())
                       .hueBy(-0.3f).hueUsing(new Linear())
                       .saturationBy(-0.2f).saturationUsing(new CubicOut())
                       .scaleWidth(.5f)
                       .duration(1000);

        // three bouncing clips:
        Sequence hue = new Sequence().hueBy(.1f).duration(1000);

        two.pause(2000).queue(bounce).queue(bounce).queue(bounce).fadeOut(500).announce(two);

        thr.queue(bounce).queue(bounce).queue(bounce).fadeOut(500);

        fur.queue(hue).queue(hue).queue(hue).fadeOut(500);

        System.out.println(anim.countClips());
        
        // render forever at 33 fps
        while (true){
            f.getGraphics().drawImage(anim.getFrame(), 0, 0, f.getWidth(), f.getHeight(), null);
            try {
                Thread.sleep(33);
            } catch (InterruptedException e) {
                System.exit(0);
            }
        }
    }

    @Override
    public void messageReceived(Object message) {
        System.out.println(message);
        System.out.println(anim.countClips());
    }
}