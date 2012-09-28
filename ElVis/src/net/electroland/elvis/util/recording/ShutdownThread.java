package net.electroland.elvis.util.recording;

public class ShutdownThread extends Thread {

    private Shutdownable s;

    public ShutdownThread(Shutdownable s){
        this.s = s;
    }

    public void run(){
        s.shutdown();
    }
}