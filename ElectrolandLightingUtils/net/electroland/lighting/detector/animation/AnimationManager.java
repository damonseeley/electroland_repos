package net.electroland.lighting.detector.animation;

import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;

public class AnimationManager implements Runnable {

	private DetectorManagerJPanel dmp;
	private Thread thread;
	private CopyOnWriteArrayList<AnimationRecipients>live;
	private CopyOnWriteArrayList<AnimationListener>listeners;	
	private boolean isRunning = false;
	private long delay;

	public AnimationManager(int fps)
	{
		this.delay = (long)(1000 / (double)fps);
		live = new CopyOnWriteArrayList<AnimationRecipients>();
		listeners = new CopyOnWriteArrayList<AnimationListener>();
		System.out.println("Delay=" + delay);
	}

	public AnimationManager(DetectorManagerJPanel dmp, int fps)
	{
		this.dmp = dmp;
		this.delay = (long)(1000 / (double)fps);
		live = new CopyOnWriteArrayList<AnimationRecipients>();
		listeners = new CopyOnWriteArrayList<AnimationListener>();
		System.out.println("Delay=" + delay);
	}

	final public void startAnimation(Animation a, Collection <Recipient> recipients){
		a.initialize();
		live.add(new AnimationRecipients(a, recipients));
		System.out.println("starting animation " + a);
	}
	final public void startAnimation(Animation a, Recipient r){
		Vector<Recipient> v = new Vector<Recipient>();
		v.add(r);
		startAnimation(a, v);
		System.out.println("starting animation " + a);
	}
	// HOW TO STOP AN ANIMATION BY FORCE???
	
	
//	final public void startAnimation(Animation a, Transition t, Collection <Recipient> fixtures){
//		// how the FUCK is this going to work.
//	}

	final public void addListener(AnimationListener listener){
		listeners.add(listener);
	}

	final public void removeListener(AnimationListener listener){
		listeners.remove(listener);
	}

	// start all animation (presuming any Animations are in the set)
	final public void goLive(){
		isRunning = true;
		if (thread == null){
			thread = new Thread(this);
			thread.start();
		}
	}

	// stop all animation
	final public void pause(){
		isRunning = false;
	}

	final public void run() {
		long startTime;
		while (isRunning){
			startTime = System.currentTimeMillis();

			// temp -----------------------------------------------------------
			// just iterator through each live show. sync what you have with
			// each fixture.
			Iterator<AnimationRecipients> animeRecips = live.iterator();
			while (animeRecips.hasNext())
			{
				AnimationRecipients ar = animeRecips.next();
				if (ar.animation.isDone())
				{	// if the animation is done, cleanup, kill it, and alert all listeners.
					ar.animation.cleanUp();
					live.remove(ar);
					Iterator<AnimationListener> list = listeners.iterator();
					while (list.hasNext())
					{
						list.next().animationComplete(ar.animation);
					}
				}else{
					// otherwise, update the animation and sync the fixtures.
					Raster r = ar.animation.getFrame();
					Iterator<Recipient> recips = ar.recipients.iterator();
					while (recips.hasNext())
					{
						recips.next().sync(r);
					}
				}
			}
			if (dmp != null)
			{
				dmp.repaint();
			}

			// end temp -------------------------------------------------------
			
			// do work

			// a:  call getFrame() on all animations. Create a List per fixture
			//     and store the result of each getFrame() on the proper lists..

			// deal with Transitions...

			// b: for each fixture, composite the frames, and call sync

			// c. (if dmp != null) call repaint() on the JPanel

			// c: figure out if isDone has been called on any Animation.

			// yes?  remove it and alert listeners.

			try 
			{
				long cycleDuration = System.currentTimeMillis() - startTime;
				Thread.sleep(cycleDuration >= delay ? 0 : delay - cycleDuration);
			} catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}
		thread = null;
	}
}
class AnimationRecipients{
	protected Animation animation;
	protected Collection<Recipient> recipients;
	public AnimationRecipients(Animation a, Collection<Recipient>r){
		this.animation = a;
		this.recipients = r;
	}
}