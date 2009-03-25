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
	private CopyOnWriteArrayList<CompletableRecipients>live;
	private CopyOnWriteArrayList<CompletionListener>listeners;	
	private boolean isRunning = false;
	private long delay;

	public AnimationManager(int fps)
	{
		this.delay = (long)(1000 / (double)fps);
		this.init(fps);
	}

	public AnimationManager(DetectorManagerJPanel dmp, int fps)
	{
		this.dmp = dmp;
		this.init(fps);
	}

	public void init(int fps)
	{
		this.delay = (long)(1000 / (double)fps);
		live = new CopyOnWriteArrayList<CompletableRecipients>();
		listeners = new CopyOnWriteArrayList<CompletionListener>();
	}

	final public void startAnimation(Completable c, Collection <Recipient> recipients)
	{
		c.initialize();
		// for each recipient to receive the new Animation, find any other 
		// instances of the recipient in the live collections, and remove them.
		Iterator <CompletableRecipients> liveAnimationRecipients = live.iterator();
		while (liveAnimationRecipients.hasNext())
		{
			liveAnimationRecipients.next().recipients.remove(recipients);
		}

		live.add(new CompletableRecipients(c, recipients));
		System.out.println("starting animation or transition " + c);
	}

	final public void startAnimation(Completable c, Recipient r)
	{
		Vector<Recipient> v = new Vector<Recipient>();
		v.add(r);
		startAnimation(c, v);
		System.out.println("starting animation or transition " + c);
	}

	final public void startAnimation(Completable c, Transition t, Collection <Recipient> r)
	{
		// no transition for now.
		startAnimation(c, r);
	}

	final public void addListener(CompletionListener listener)
	{
		listeners.add(listener);
	}

	final public void removeListener(CompletionListener listener)
	{
		listeners.remove(listener);
	}

	// start all animation (presuming any Animations are in the set)
	final public void goLive()
	{
		isRunning = true;
		if (thread == null)
		{
			thread = new Thread(this);
			thread.start();
		}
	}

	// stop all animation
	final public void pause()
	{
		isRunning = false;
	}

	final public void run()
	{
		long startTime;
		while (isRunning)
		{
			startTime = System.currentTimeMillis();

			// temp -----------------------------------------------------------
			// just iterator through each live show. sync what you have with
			// each fixture.
			Iterator<CompletableRecipients> animeRecips = live.iterator();
			while (animeRecips.hasNext())
			{
				CompletableRecipients ar = animeRecips.next();
				if (ar.completable.isDone())
				{	// if the animation is done, cleanup, kill it, and alert all listeners.
					ar.completable.cleanUp();
					live.remove(ar);
					Iterator<CompletionListener> list = listeners.iterator();
					while (list.hasNext())
					{
						list.next().completed(ar.completable);
					}
				}else{
					// otherwise, update the animation and sync the fixtures.
					if (ar.completable instanceof Animation)
					{
						Raster r = ((Animation)ar.completable).getFrame();
						Iterator<Recipient> recips = ar.recipients.iterator();
						while (recips.hasNext())
						{
							recips.next().sync(r);
						}
					}
				}
			}
			if (dmp != null)
			{
				dmp.repaint();
			}

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
class CompletableRecipients
{
	protected Completable completable;
	protected Collection<Recipient> recipients;
	public CompletableRecipients(Completable a, Collection<Recipient>r)
	{
		this.completable = a;
		this.recipients = r;
	}
}