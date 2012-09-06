ElectrolandAnimation
--
The root class for this library is Animation.  Animation generates frames of
animation in the form of images.  It doesn't create an animation thread. 
Instead, you _poll_ it from the nextFrame() method to get frames to render on
your own.

* An instance of an Animation contains any number of Clips on it's "stage".
* You animate a Clip by applying a Tween to it
* A Tween can use any form of EasingFunction
* A Sequence is a set of Tweens.  A typical animation pattern looks like this:
* An AnimationListener can subscribe to broadcsts from the Animation.

// create an Animation, loading the required resources
Animation anim = new Animation("animation.properties");
anim.setBackground(Color.WHITE);

// create a couple clips
Clip one = anim.addClip(anim.getContent("stillImage"),  50, 50, 100, 100, 1.0f);
Clip two = anim.addClip(anim.getContent("slowImage"),  150, 50, 100, 100, 1.0f);
Clip thr = anim.addClip(anim.getContent("fastImage"),  250, 50, 100, 100, 1.0f);

// create a sequence:
Sequence bounce = new Sequence(); 

         bounce.yTo(150).yUsing(new QuinticIn())
               .xBy(100).xUsing(new Linear())
               .scaleWidth(2.0f)
               .duration(1000)
        .newState()
               .yTo(75).yUsing(new CubicOut())
               .xBy(100).xUsing(new Linear())
               .scaleWidth(.5f)
               .duration(1000);

// apply the sequences:
// three bouncing clips:
one.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();
two.pause(1000).queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();
thr.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone().announce("I'm done");

// your render thread:
while (true){
    myPanel.getGraphics().drawImage(anim.getFrame(), 0, 0, 
                                    myPanel.getWidth(), myPanel.getHeight(), null);
    try {
        Thread.sleep(33);
    } catch (InterruptedException e) {
        System.exit(0);
    }
}

