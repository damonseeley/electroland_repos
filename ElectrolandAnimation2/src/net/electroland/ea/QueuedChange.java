package net.electroland.ea;

/**
 * represents the state of a queued change.
 * @author production
 *
 */
public class QueuedChange {
    final public static int DELETE_CHILDREN = -2;
    final public static int DELETE = -1;
    final public static int DELAY = 0;
    final public static int CHANGE = 1;
    protected int type;
    protected Change change;
    protected long duration = 0;
    protected long delay = 0;
    protected long startTime;
    protected long endTime;
    protected boolean started = false;
}