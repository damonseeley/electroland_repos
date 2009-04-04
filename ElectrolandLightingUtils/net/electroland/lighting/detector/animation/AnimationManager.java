package net.electroland.lighting.detector.animation;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;

public class AnimationManager implements Runnable {

	private DetectorManagerJPanel dmp;
	private Thread thread;
	private HashMap<Animation, AnimationRecipients>animationRecipients;
	private HashMap<Recipient, RecipientState>recipientStates;
	private CopyOnWriteArrayList<AnimationListener>listeners;	
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
		animationRecipients = new HashMap<Animation, AnimationRecipients>();
		recipientStates = new HashMap<Recipient, RecipientState>();
		listeners = new CopyOnWriteArrayList<AnimationListener>();
	}

	// no transition = kill existing show on any overlapping recipient
	final public void startAnimation(Animation a, Collection <Recipient> recipients)
	{
		// initialize the new animation.
		a.initialize();

		synchronized (animationRecipients)
		{
			Iterator <Animation> currentItr = animationRecipients.keySet().iterator();
			while (currentItr.hasNext())
			{
				Animation currentAnimation = currentItr.next();
				AnimationRecipients ar = animationRecipients.get(currentAnimation);
				// if any of the recipients we are targetting currently is allocated
				// to another animation, take it from the other animation.
				ar.recipients.remove(recipients);
				// if the other animation has no recipients left, kill it off.
				if (ar.recipients.isEmpty())
				{
					this.killOff(currentAnimation);
				}
			}
			// store the show to recipient mappings
			animationRecipients.put(a, new AnimationRecipients(recipients));

			// set the current states
			Iterator <Recipient> recipientsItr = recipients.iterator();
			while (recipientsItr.hasNext())
			{
				recipientStates.put(recipientsItr.next(), new RecipientState(a));
			}			
		}

		System.out.println("starting animation " + a);
		this.printState(System.out);
	}

	final public void startAnimation(Animation c, Recipient r)
	{
		Vector<Recipient> v = new Vector<Recipient>();
		v.add(r);
		startAnimation(c, v);
	}

	final public void startAnimation(Animation a, Animation t, Collection <Recipient> r)
	{
		// initialize the new animation.
		a.initialize();

		synchronized (animationRecipients)
		{
		
			// store the new animations in CompletableRecipients
			animationRecipients.put(a, new AnimationRecipients(r));
			// store each transition in the RecipientCompletable
			Iterator <Recipient>rItr = r.iterator();
			while (rItr.hasNext())
			{
				RecipientState rState = recipientStates.get(rItr.next());
				if (rState == null)
				{
					// if there was no animation, no transition is required.
					recipientStates.put(rItr.next(), new RecipientState(a));
				}else
				{
					// initialize the transition
					t.initialize();
					// store the transitions in CompletableRecipients.
					animationRecipients.put(t, new AnimationRecipients(r, true));
					// update the state of the recipient to "transitioning"
					rState.transition = t;
					rState.target = a;
				}
			}
		}
		System.out.println("transition to animation " + a + " using transition " + t);
		this.printState(System.out);
	}

	final public void startAnimation(Animation c, Animation t, Recipient r)
	{
		Vector<Recipient> v = new Vector<Recipient>();
		v.add(r);
		startAnimation(c, t, v);
	}

	final public void addListener(AnimationListener listener)
	{
		listeners.add(listener);
	}

	final public void removeListener(AnimationListener listener)
	{
		listeners.remove(listener);
	}
	final public void killOff(Animation a)
	{
		a.cleanUp();
		synchronized (animationRecipients)
		{
			boolean isTransition = animationRecipients.get(a).isTransition;
			animationRecipients.remove(a);
			if (!isTransition)
			{
				Iterator<AnimationListener> list = listeners.iterator();
				while (list.hasNext())
				{
					list.next().completed(a);
				}
			}
		}
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

	final public Animation getCurrentAnimation(Recipient r)
	{
		synchronized (animationRecipients){
			return recipientStates.get(r).current;
		}
	}


	final public void run()
	{
		long startTime;
		while (isRunning)
		{
			startTime = System.currentTimeMillis();

			synchronized (animationRecipients){
	
				// see which animations and transitions are done.
				Iterator<Animation> doneItr = animationRecipients.keySet().iterator();
				while (doneItr.hasNext())
				{
					Animation animation = doneItr.next();
					AnimationRecipients ar = animationRecipients.get(animation);
					if (animation.isDone())
					{
						// find all recipients that were using this transition.
						Iterator<Recipient> recipients = ar.recipients.iterator();
						while (recipients.hasNext())
						{
							Recipient recipient = recipients.next();
							RecipientState rState = recipientStates.get(recipients);
							if (ar.isTransition){
								// kill off the animation we transitioned from.
								rState.current.cleanUp();
								killOff(rState.current);
								// set their state to current = target and transition = null.
								rState.transition = null;
								rState.current = rState.target;
							}else
							{
								// any recipient using this animation (and not transitioning)
								// needs to change it's state to either removed from the state list
								if (rState.transition == null)
								{
									recipientStates.remove(recipient);
								}
							}
						}
						// kill off the animation.
						killOff(animation);
					}else
					{
						ar.latestFrame = animation.getFrame();
					}
				}
	
				Iterator <Recipient> recipients = recipientStates.keySet().iterator();
				while (recipients.hasNext())
				{
					Recipient recipient = recipients.next();
					RecipientState state = recipientStates.get(recipient);
					if (state.transition == null)
					{
						recipient.sync(animationRecipients.get(state.current).latestFrame);
					}else
					{
						// THIS WILL BREAK if the animation you were transitioning to
						// ended before the transition did!
	//					recipient.sync(state.current == null ? null : showRecipients.get(state.current).latestFrame,
	//									showRecipients.get(state.transition).latestFrame,
	//									showRecipients.get(state.target).latestFrame);

						// haven't implemented the actual transition yet.  instead, show the target.
						// THIS WILL ALSO BREAK if the animation you were transitioning to
						// ended before the transition did!
						recipient.sync(animationRecipients.get(state.target).latestFrame);
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

	public void printState(PrintStream os)
	{
		os.println("Live animations and there assigned recipients:");
		os.println("==============================================");

		Iterator<Animation> animations = animationRecipients.keySet().iterator();
		while (animations.hasNext())
		{
			Animation a = animations.next();
			os.println("Animation: " + a + "\t" + animationRecipients.get(a));
		}
		os.println();
		os.println("Recipients running shows, and their transition states:");
		os.println("======================================================");
		
		Iterator<Recipient> recipients = recipientStates.keySet().iterator();
		while (recipients.hasNext())
		{
			Recipient r = recipients.next();
			os.println("Recipient: " + r.getID() + "\t" + recipientStates.get(r));
		}
	}
}
/**
 * Stores all recipients that require the next frame from this animation, 
 * whether or not this is a transition object, and buffers the latest frame.
 * @author geilfuss
 * 
 */
class AnimationRecipients
{
	protected Collection<Recipient> recipients;
	protected boolean isTransition = false;
	protected Raster latestFrame;

	public AnimationRecipients(Collection<Recipient>r)
	{
		this.recipients = r;
	}

	public AnimationRecipients(Collection<Recipient>r, boolean isTransition)
	{
		this.recipients = r;
		this.isTransition = isTransition;
	}
	public String toString()
	{
		return "AnimationRecipients[isTransition=" + isTransition + ", " + recipients + "]";
	}
}
/**
 * Stores what animation and optionally what transition and target animation
 * are running on each specific Recipient.
 * @author geilfuss
 *
 */
class RecipientState
{
	protected Animation current;
	protected Animation transition;
	protected Animation target;

	public RecipientState(Animation current)
	{
		this.current = current;
	}
	public RecipientState(Animation current, Animation transition, Animation target)
	{
		this.current = current;
		this.transition = transition;
		this.target = target;
	}
	public String toString()
	{
		return "RecipientState [current=" + current + ", transition=" + transition + ", target=" + target + "]";
	}
}