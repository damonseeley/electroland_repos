package net.electroland.ea;

public class QueuedChange {
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