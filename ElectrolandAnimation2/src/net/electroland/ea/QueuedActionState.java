package net.electroland.ea;

/**
 * represents the state of a queued tween/message/delay.
 * 
 * Name is a little inacurate, since sometimes we're not tweening- we're a
 * delay or a message.
 * 
 * @author production
 *
 */
public class QueuedActionState {
    final public static int DELETE_CHILDREN = -2;
    final public static int DELETE          = -1;
    final public static int DELAY           = 0;
    final public static int CHANGE          = 1;
    final public static int MESSAGE         = 2;

    protected int     type;
    protected Tween   change;
    protected long    duration     = 0;
    protected long    delay        = 0;
    protected long    startTime;
    protected long    endTime;
    protected boolean started   = false;
}