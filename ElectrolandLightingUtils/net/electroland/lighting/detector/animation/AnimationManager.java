package net.electroland.lighting.detector.animation;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.log4j.Logger;

import net.electroland.lighting.detector.DetectorManagerJPanel;
import net.electroland.lighting.detector.Recipient;

public class AnimationManager implements Runnable 
{
	private static Logger logger = Logger.getLogger(AnimationManager.class);

	private DetectorManagerJPanel dmp;
	private Thread thread;

	// This object contains all live animations and transitions.  That includes
	// animations that are being transitioned to.
	// animationRecipients hashes all the recipients that are playing an Animation
	// to the Animation. Each AnimationRecipients object lets you know the list of
	// recipients, whether or not the Animation is being used as a transition, and
	// also buffers the latest frame from that Animation.
	private ConcurrentHashMap<Animation, AnimationRecipients>animationRecipients;

	// recipientStates stores the state of every Recipient that is running an
	// Animation.  It does NOT include Recipients that are not running anything.
	// the RecipientState object keeps track of whether or not the object is
	// transitioning.
	private ConcurrentHashMap<Recipient, RecipientState>recipientStates;
	private CopyOnWriteArrayList<AnimationListener>listeners;
	private boolean isRunning = false;
	private long delay; // frame delay to achieve optimal fps

	public AnimationManager(int fps)
	{
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
		animationRecipients = new ConcurrentHashMap<Animation, AnimationRecipients>();
		recipientStates = new ConcurrentHashMap<Recipient, RecipientState>();
		listeners = new CopyOnWriteArrayList<AnimationListener>();
	}

	final public void reapRecipient(Collection <Recipient> recipients)
	{
		synchronized (animationRecipients)
		{
			Iterator <Animation> currentItr = animationRecipients.keySet().iterator();
			while (currentItr.hasNext())
			{
				Animation currentAnimation = currentItr.next();
				AnimationRecipients ar = animationRecipients.get(currentAnimation);
				// if any of the recipient we are targetting currently is allocated
				// to another animation, take it from the other animation.
				ar.recipients.remove(recipients);
				// if the other animation has no recipients left, kill it off.
				if (ar.recipients.isEmpty())
				{
					animationRecipients.remove(currentAnimation);
				}
			}
		}
	}

	final public Recipient reapRecipient(Recipient r)
	{
		CopyOnWriteArrayList<Recipient> v = new CopyOnWriteArrayList<Recipient>();
		v.add(r);
		reapRecipient(v);
		return r;
	}

	// no transition = kill existing show on any overlapping recipient
	final public void startAnimation(Animation a, Collection <Recipient> r)
	{
		if (a == null)
		{
			throw new RuntimeException("Animation object is null.");
		}else if (r == null || r.isEmpty())
		{
			throw new RuntimeException("Recipient list is null or empty.");
		}

		synchronized (animationRecipients)
		{
			Iterator <Animation> currentItr = animationRecipients.keySet().iterator();
			while (currentItr.hasNext())
			{
				Animation currentAnimation = currentItr.next();
				AnimationRecipients ar = animationRecipients.get(currentAnimation);
				// if any of the recipients we are targetting currently is allocated
				// to another animation, take it from the other animation.
				ar.recipients.removeAll(r);
				// if the other animation has no recipients left, kill it off.
				if (ar.recipients.isEmpty())
				{
					animationRecipients.remove(currentAnimation);
				}
			}
			// store the show to recipient mappings
			animationRecipients.put(a, new AnimationRecipients(r));

			// set the current states
			Iterator <Recipient> recipientsItr = r.iterator();
			while (recipientsItr.hasNext())
			{
				recipientStates.put(recipientsItr.next(), new RecipientState(a));
			}
		}

		logger.info("starting animation " + a);
		this.printState();
	}

	final public void startAnimation(Animation c, Recipient r)
	{
		CopyOnWriteArrayList<Recipient> v = new CopyOnWriteArrayList<Recipient>();
		v.add(r);
		startAnimation(c, v);
	}

	final public void startAnimation(Animation a, Animation t, Collection <Recipient> r)
	{
		if (a == null)
		{
			throw new RuntimeException("Animation object is null.");
		}else if (t == null)
		{
			throw new RuntimeException("Transition object is null.");
		}else if (r == null || r.isEmpty())
		{
			throw new RuntimeException("Recipient list is null or empty.");
		}

		synchronized (animationRecipients)
		{
			animationRecipients.put(a, new AnimationRecipients(r));
			animationRecipients.put(t, new AnimationRecipients(r, true));
			// store each transition in the RecipientCompletable
			Iterator <Recipient>rItr = r.iterator();
			while (rItr.hasNext())
			{
				Recipient recip = rItr.next();
				RecipientState rState = recipientStates.get(recip);
				if (rState == null)
				{
					// if there was no animation, no transition is required.
					recipientStates.put(rItr.next(), new RecipientState(a));
				}else
				{
					rState.transition = t;
					rState.target = a;//<-- does this need to go in animationRecipients??
				}
			}
		}
		logger.info("transition to animation " + a + " using transition " + t);
		this.printState();
	}

	final public void startAnimation(Animation c, Animation t, Recipient r)
	{
		CopyOnWriteArrayList<Recipient> v = new CopyOnWriteArrayList<Recipient>();
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

	// this is the problem:
	// this method sees if the animation is in transition, then removes it
	// for all recipients that are currently showing it, and alerts any
	// listeners that it is done.  it should ONLY every be called once
	// per animation.  however, we call it in a couple loops.

	// more importantly, there's a design problem here.  an animation could
	// end for one recipient, but not for others.  for instance, if you start
	// an animation up on one of those recipients, but not others.
	final private void notifyCompletionListeners(Animation a)
	{
		synchronized (animationRecipients)
		{
			Iterator<AnimationListener> list = listeners.iterator();
			while (list.hasNext())
			{
				list.next().completed(a);
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
	final public void stop()
	{
		isRunning = false;
	}

	final public Animation getCurrentAnimation(Recipient r)
	{
		synchronized (animationRecipients){
			RecipientState rs = recipientStates.get(r);
			return rs == null ? null : rs.current;
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

					// the AnimationRecipient may have been removed in a prior
					// execution of this loop.
					if (ar != null)
					{
						if (animation.isDone())
						{
							if (ar.isTransition)
							{
								// if an animation used as a transition just ended
								// check each recipient that the animation was running on:
								Iterator<Recipient> recipients = ar.recipients.iterator();
								while (recipients.hasNext())
								{
									Recipient recipient = recipients.next();
									RecipientState rState = recipientStates.get(recipient);
									// for each recipient
									// 1.) remove the Recipient from the list of recipients
									//		that Animation is playing on
									ar.recipients.remove(recipient);
									// 2.) if the Animation you transitioned from isn't playing for anyone else, kill it.
									if (ar.recipients.isEmpty())
									{
										animationRecipients.remove(ar);
									}
									// 3.) set the Recipient's current state to it's target and forget the transition
									rState.transition = null;
									rState.current = rState.target;
								}
							}else
							{
								// if a standard animation just ended on it's own,
								// check each recipient that the animation was running on:
								Iterator<Recipient> recipients = ar.recipients.iterator();
								while (recipients.hasNext())
								{
									Recipient recipient = recipients.next();
									RecipientState rState = recipientStates.get(recipient);
									// for each recipient that was transitioning:
									if (rState.target != null){
										// 1.) jump straight to the new Animation
										rState.transition = null;
										rState.current = rState.target;
									}else
									// for each animation that was not transitioning:
									{
										// 1.) remove the recipient from the state list.
										recipientStates.remove(recipient);
									}
								}
								// 2.) remove the Animation from the list of running animations
								animationRecipients.remove(ar);
								// 3.) notify listeners.
								this.notifyCompletionListeners(animation);
							}
						}else
						{
							ar.latestFrame = animation.getFrame();
						}
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
						recipient.sync(state.current == null ? null : animationRecipients.get(state.current).latestFrame,
								state.transition == null ? null : animationRecipients.get(state.transition).latestFrame,
								state.target == null ? null : animationRecipients.get(state.target).latestFrame);
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
				logger.error("error attempting to put animation manager to sleep", e);
			}
		}
		thread = null;
	}

	public void printState()
	{
		logger.info("Live animations and there assigned recipients:");
		logger.info("==============================================");

		Iterator<Animation> animations = animationRecipients.keySet().iterator();
		while (animations.hasNext())
		{
			Animation a = animations.next();
			logger.info("Animation: " + a + "\t" + animationRecipients.get(a));
		}
		logger.info("");
		logger.info("Recipients running shows, and their transition states:");
		logger.info("======================================================");
		
		Iterator<Recipient> recipients = recipientStates.keySet().iterator();
		while (recipients.hasNext())
		{
			Recipient r = recipients.next();
			logger.info("Recipient: " + r.getID() + "\t" + recipientStates.get(r));
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
		this.recipients = new CopyOnWriteArrayList<Recipient>();
		this.recipients.addAll(r);
	}

	public AnimationRecipients(Collection<Recipient>r, boolean isTransition)
	{
		this.recipients = new CopyOnWriteArrayList<Recipient>();
		this.recipients.addAll(r);
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