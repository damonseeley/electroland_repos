package net.electroland.utils.process;

import java.io.File;
import java.io.IOException;

import org.apache.log4j.Logger;

public class MonitoredProcess implements Runnable {

    static Logger logger = Logger.getLogger(MonitoredProcess.class);

    private String command, name;
    private File runDir;
    private long startDelayMillis;
    private boolean restartOnTermination, firstRun = true;
    private InputToOutputThread pOut, pErr;
    private Process running;

    public MonitoredProcess(String name, String command, File runDir, long startDelayMillis, boolean restartOnTermination){
        this.name                 = name;
        this.command              = command;
        this.runDir               = runDir;
        this.startDelayMillis     = startDelayMillis;
        this.restartOnTermination = restartOnTermination;
    }

    public String getName(){
        return name;
    }

    @Override
    public void run() {

        while (restartOnTermination || firstRun){

            try {

                if (!firstRun) {
                    logger.info("starting " + name + " in " + startDelayMillis + " millis.");
                    Thread.sleep(startDelayMillis);
                }

                logger.info(" starting " + name + " now.");
                logger.info(" exec " + runDir + "\\" + command);
                // TODO: corner case: if kill(false) gets called while the previous sleep is underway,
                //       the wrong running process (old one) will be killed.
                //       there should probable be a second check to see if the
                //       restart is still valid here.  Or maybe just sync running?
                running = Runtime.getRuntime().exec(command, null, runDir);

                firstRun = false;

                pOut = new InputToOutputThread(running.getInputStream(), Logger.getLogger(command));
                pOut.startPiping();

                pErr = new InputToOutputThread(running.getErrorStream(), Logger.getLogger(command));
                pErr.startPiping();

                logger.info("monitoring " + name);
                running.waitFor();
                logger.info("process died. ");
                if (restartOnTermination){
                    logger.info(" restart requested.");
                } else {
                    logger.info(" and NO restart requested.");
                    // TODO: if there are no processes left: System.exit(0);
                }

            } catch (IOException e) {
                restartOnTermination = false;
                e.printStackTrace();
                System.exit(-1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void startProcess(){
        new Thread(this).start();
    }

    public void kill(boolean restartOnTermination) {
        logger.info("kill called.");
        this.restartOnTermination = restartOnTermination;
        if (restartOnTermination){
            logger.info(" restart requested.");
        }

        if (running != null){
            running.destroy();
        }
    }
}