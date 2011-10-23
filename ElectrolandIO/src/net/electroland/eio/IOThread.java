package net.electroland.eio;

import java.util.Collection;

import org.apache.log4j.Logger;

import net.electroland.eio.devices.IODevice;

public class IOThread implements Runnable {

    private static Logger logger = Logger.getLogger(IOThread.class);
    protected int rate = 1;
    private Thread thread = null;
    private Collection<IODevice> devices;
    private boolean isRunning = false;

    public IOThread(Collection<IODevice> devices, int rate)
    {
        this.devices = devices;
        this.rate = rate;
    }

    @Override
    public void run() {
    
        logger.info("IOThread: starting");
    	// connect all
        for (IODevice device: devices)
        {
            device.connect();
        }
    
        while (isRunning){

            long start = System.currentTimeMillis();

            // sync states
            for (IODevice device: devices)
            {
                device.readInput();
                device.sendOutput();
            }
            // TODO: add code to get measured FPS
            long duration = System.currentTimeMillis() - start;
            long delay = (long)(1000.0/rate);

            if (duration < delay)
                delay -= duration;
            else
                delay = 0;

            try {
                Thread.sleep(delay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // close all
        for (IODevice device: devices)
        {
            device.close();
        }
        thread = null;
        logger.info("IOThread: stopped");
    }

    public void start()
    {
        isRunning = true;
        thread = new Thread(this);
        thread.start();
    }
    
    public void stop()
    {
        isRunning = false;
    }
}