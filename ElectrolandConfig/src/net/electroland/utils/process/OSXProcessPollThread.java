package net.electroland.utils.process;

public class OSXProcessPollThread extends ProcessPollThread {

    public OSXProcessPollThread(ProcessItem item,
            ProcessExitedListener listener, long period) {
        super(item, listener, period);
    }

    @Override
    public boolean processExited() {
        // "ps -e | grep 9999"
        return false;
    }

}
