package net.electroland.norfolk.core;

import org.apache.log4j.Logger;

public class GlobalShow implements Runnable {

    private static Logger logger = Logger.getLogger(GlobalShow.class);
    private ClipPlayer player;
    private long nextPlayTime = 0;
    private Thread thread;

    public GlobalShow(ClipPlayer player){
        this.player = player;
    }

    public void run(){
        while (thread != null){
            if (System.currentTimeMillis() > nextPlayTime){

                /***  play whateve you want here!  Pick from a random list, ***/
                /***  or whatever.                                          ***/
                logger.info("playing 60 seconds show...");
                player.play("timedShow");

                /**************************************************************/
                nextPlayTime = System.currentTimeMillis() + 60 * 1000; // 60 seconds
            }
            try {
                Thread.sleep(1000); // poll every second instead of sleeping for 60,
                                    // so we can more easily turn this thread off.
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void stop(){
        thread = null;
    }

    public void start(){
        if (thread == null){
            thread = new Thread(this);
            thread.start();
        }
    }
}