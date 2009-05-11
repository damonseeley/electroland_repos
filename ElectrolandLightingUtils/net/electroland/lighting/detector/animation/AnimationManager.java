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

	public static final int ALL_START_ANIMATION = 0;
	public static final int ALL_TARGET_ANIMATION = 255;

	private DetectorManagerJPanel dmp;
	private Thread thread;
	private ConcurrentHashMap<Animation, AnimationRecipients>animationRecipients; // Hashtable?
	private ConcurrentHashMap<Recipient, RecipientState>recipientStates;
	private CopyOnWriteArrayList<AnimationListener>listeners;	// why not just Vector?
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
					this.killOff(currentAnimation);
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
	final public void startAnimation(Animation a, Collection <Recipient> recipients)
	{
		synchronized (animationRecipients)
		{
			Iterator <Animation> currentItr = animationRecipients.keySet().iterator();
			while (currentItr.hasNext())
			{
				Animation currentAnimation = currentItr.next();
				AnimationRecipients ar = animationRecipients.get(currentAnimation);
				// if any of the recipients we are targetting currently is allocated
				// to another animation, take it from the other animation.
				ar.recipients.removeAll(recipients);
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
					// store the transitions in CompletableRecipients.
					animationRecipients.put(t, new AnimationRecipients(r, true));
					// update the state of the recipient to "transitioning"
					rState.transition = t;
					rState.target = a;
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
	final private void killOff(Animation a)
	{
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
	// for now, same as pause.
	final public void stop()
	{
		isRunning = false;
	}
	// stop all animation
	final public void pause()
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
					if (animation.isDone())
					{
						// find all recipients that were using this transition.
						Iterator<Recipient> recipients = ar.recipients.iterator();
						while (recipients.hasNext())
						{
							Recipient recipient = recipients.next();
							RecipientState rState = recipientStates.get(recipient);
							if (ar.isTransition){
								// kill off the animation we transitioned from.
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