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
    Animation anim;

    public static void main(String args[])
    {
        // create a context
        Hashtable<String, Object> context = new Hashtable<String, Object>();
        // just putting some bullshit object into the context.  normally, you'd
        // put something useful in, like "sound_manager", soundManager
        context.put("random_rectangle", new Rectangle());

        // create an Animator
        Animation anim = new Animation("animation.properties");
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
        Clip fur = anim.addClip(Color.getHSBColor(0.9f,0.8f,1.0f), 350, 50, 100, 100, 1.0f);

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
        one.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();

        two.pause(1000).queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();

        thr.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();

        fur.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();

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
    }
}