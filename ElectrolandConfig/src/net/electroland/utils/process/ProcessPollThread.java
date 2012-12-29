package net.electroland.utils.process;

abstract public class ProcessPollThread extends Thread {

    protected ProcessItem item;
    protected ProcessExitedListener listener;
    private long period;

    public ProcessPollThread(ProcessItem item, ProcessExitedListener listener, long period){
        this.item = item;
        this.listener = listener;
        this.period = period;
    }

    abstract public boolean processExited();

    public void run(){
        boolean running = true;
        while (running){
            if (processExited()){
                running = false;
                if (listener != null){
                    listener.exited(item);
                }
            }
            try {
                Thread.sleep(period);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}