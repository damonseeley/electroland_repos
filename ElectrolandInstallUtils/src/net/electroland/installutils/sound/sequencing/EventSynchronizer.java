package net.electroland.installutils.sound.sequencing;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import net.electroland.installutils.sound.sequencing.test.TestRepeatingTimeSyncedEvent;
import net.electroland.installutils.sound.sequencing.test.TestTimeSyncedEvent;

/**
 * Quantizes events to a constant tempo.  The tempo is assigned in milliseconds.
 * It supports a queued and a non-queued version.  If you allow queing, a new
 * event on a channel with existing events queued up has to wait until all
 * previous events are processed at one event per tempo period.  In the non-
 * queing version, if you play an event on a channel with an existing event, the
 * existing even is simply removed from the queue.
 * 
 * There are two types of events that can be played: TimeSyncedEvent and
 * RepeatingTimeSyncedEvent.  A TimeSyncedEvent plays once.  A repeating one
 * has a counter, and will play for the requested number of counts.  If you are
 * queing events, all repeating instances of a single repeatign event will 
 * play before the next event in the queue.  A RepeatingTimeSyncedEvent with
 * a repeat count of -1 will play forever.
 * 
 * @author bradley
 *
 */
public class EventSynchronizer implements Runnable{

    private int tempo;
    private Map <String, Object> context;
    private Map <String, ConcurrentLinkedQueue<TimeSyncedEvent>> channels;
    private boolean queueEvents = false;

    private Thread thread = null;
    private long next;
    final static private long sleep = 5;

    // unit testing (doesn't test asyncronous calls)
    public static void main(String args[])
    {
        EventSynchronizer s = new EventSynchronizer(300, null, true);
        s.playEvent("repeater", new TestRepeatingTimeSyncedEvent("repeater", 5));
        s.playEvent("non1", new TestTimeSyncedEvent("non1"));
        s.playEvent("non1", new TestTimeSyncedEvent("non1"));
        s.playEvent("non2", new TestTimeSyncedEvent("non2"));
        s.playEvent("non2", new TestTimeSyncedEvent("non2"));
        s.playEvent("non2", new TestRepeatingTimeSyncedEvent("repeater2", 3));
        s.start();
        s.stop();
        s.start();
        s.start();
    }

    /**
     * Creates an instance of an EventSynchronizer.
     * 
     * @param tempo - milliseconds for each beat.
     * @param context - a map of arbitrary objects made available to each event
     * @param queueEvents - true if events can be queued. false otherwise.
     */
    public EventSynchronizer(int tempo, Map<String, Object> context, boolean queueEvents){

        if (tempo < sleep * 2)
        {
            throw new RuntimeException("Tempo cannot be less than " + sleep * 2);
        }else{
            this.tempo =tempo;
            this.context = context;
            this.queueEvents = queueEvents;
            channels = Collections.synchronizedMap(new HashMap<String, ConcurrentLinkedQueue<TimeSyncedEvent>>());
        }
    }

    /**
     * queue an event on a specified channel.  Channels are just arbitrary
     * names.  If EventSynchronizer is not set to allow queing, this method
     * will remove any existing unplayed event on the same channel.
     * 
     * @param channelName
     * @param event
     */
    public void playEvent(String channelName, TimeSyncedEvent event)
    {
        ConcurrentLinkedQueue<TimeSyncedEvent> q = channels.get(channelName);
        if (q == null){
            q = new ConcurrentLinkedQueue<TimeSyncedEvent>();
            channels.put(channelName, q);
        }
        if (!queueEvents){
            q.clear();
        }
        q.add(event);
    }

    /**
     * Call this to start playing events.
     */
    public void start()
    {
        if (thread == null)
        {
            thread = new Thread(this);
            next = System.currentTimeMillis();
            thread.start();
        }
    }

    /**
     * Call this to stop playing events.
     */
    public void stop()
    {
        thread = null;
    }

    @Override
    public final void run() {
        while (thread != null)
        {
            if (System.currentTimeMillis() > next){

                for (String channel : channels.keySet())
                {
                    Queue<TimeSyncedEvent> queue = channels.get(channel);
                    TimeSyncedEvent e = queue.peek();
                    if (e != null){
                        e.doEvent(context);

                        if (e instanceof RepeatingTimeSyncedEvent)
                        {
                            // checking for equals 0 rather than < 1 so that -1 can 
                            // be used for infinite repeat.
                            if (--((RepeatingTimeSyncedEvent)e).repeats == 0){
                                queue.remove();
                            }
                        }else{
                            queue.remove();
                        }
                    }
                }
                next += tempo;
            }

            try {
                Thread.sleep(sleep);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}