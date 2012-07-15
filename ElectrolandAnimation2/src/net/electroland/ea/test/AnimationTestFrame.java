package net.electroland.ea.test;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.Hashtable;

import javax.swing.JFrame;

import net.electroland.ea.Animation;
import net.electroland.ea.AnimationListener;
import net.electroland.ea.Clip;
import net.electroland.ea.Tween;
import net.electroland.ea.easing.CircularIn;
import net.electroland.ea.easing.CircularInOut;
import net.electroland.ea.easing.CircularOut;
import net.electroland.ea.easing.CubicIn;
import net.electroland.ea.easing.CubicInOut;
import net.electroland.ea.easing.CubicOut;
import net.electroland.ea.easing.ExponentialIn;
import net.electroland.ea.easing.Linear;
import net.electroland.ea.easing.QuadraticIn;
import net.electroland.ea.easing.QuadraticOut;
import net.electroland.ea.easing.QuarticIn;
import net.electroland.ea.easing.QuarticInOut;
import net.electroland.ea.easing.QuarticOut;
import net.electroland.ea.easing.QuinticIn;
import net.electroland.ea.easing.QuinticOut;

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
        Clip one = anim.addClip(anim.getContent("stillImage"),  50, 50, 100, 100, 0.0f);
        Clip two = anim.addClip(anim.getContent("slowImage"),  150, 50, 100, 100, 0.0f);
        Clip thr = anim.addClip(anim.getContent("fastImage"),  250, 50, 100, 100, 0.0f);

        // a couple tweening directives
        Tween d1 = new Tween(new QuarticIn()).yBy(100).scaleWidth(2.0f).widthUsing(new Linear()).alphaTo(1.0f);
        Tween d2 = new Tween(new QuarticOut()).yBy(100).alphaTo(1.0f);
        Tween d3 = new Tween(new QuarticInOut()).yBy(100).alphaTo(1.0f);

        Tween r1 = new Tween(new CircularIn()).xBy(100).scaleWidth(.5f).widthUsing(new ExponentialIn());
        Tween r2 = new Tween(new CircularOut()).xBy(100);
        Tween r3 = new Tween(new CircularInOut()).xBy(100);

        Tween u1 = new Tween(new CubicIn()).yBy(-100);
        Tween u2 = new Tween(new CubicOut()).yBy(-100).scaleHeight(.5f);
        Tween u3 = new Tween(new CubicInOut()).yBy(-100);

        Tween l1 = new Tween(new QuarticIn()).xBy(-100);
        Tween l2 = new Tween(new QuarticOut()).xBy(-100).scaleHeight(2.0f);
        Tween l3 = new Tween(new QuarticInOut()).xBy(-100);

        Tween c1 = new Tween().yTo(150).xBy(100).yUsing(new QuinticIn()).xUsing(new Linear());
        Tween c2 = new Tween().yTo(150).xBy(100).yUsing(new CubicIn()).xUsing(new Linear());
        Tween c3 = new Tween().yTo(150).xBy(100).yUsing(new QuadraticIn()).xUsing(new Linear());

        Tween c4 = new Tween().yTo(75).xBy(50).yUsing(new QuinticOut()).xUsing(new Linear()).scaleAlpha(.25f);
        Tween c5 = new Tween().yTo(75).xBy(50).yUsing(new CubicOut()).xUsing(new Linear()).scaleAlpha(.25f);
        Tween c6 = new Tween().yTo(75).xBy(50).yUsing(new QuadraticOut()).xUsing(new Linear()).scaleAlpha(.25f);

        // queue those tweening directives
        one.queue(d1, 2000).queue(r1, 2000).queue(u1, 2000).queue(l1, 2000).queue(c1, 1000).queue(c4, 1000).queue(c1, 1000).queue(c4, 1000).fadeOut(500).queue(c1, 1000).queue(c4, 1000).fadeOut(500).deleteWhenDone();
        two.queue(d2, 2000).queue(r2, 2000).queue(u2, 2000).queue(l2, 2000).queue(c2, 1000).queue(c5, 1000).queue(c2, 1000).queue(c5, 1000).fadeOut(500).fadeOut(500).deleteWhenDone();
        thr.queue(d3, 2000).queue(r3, 2000).queue(u3, 2000).queue(l3, 2000).queue(c3, 1000).queue(c6, 1000).queue(c3, 1000).queue(c6, 1000).fadeOut(500).fadeOut(500).deleteWhenDone();

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