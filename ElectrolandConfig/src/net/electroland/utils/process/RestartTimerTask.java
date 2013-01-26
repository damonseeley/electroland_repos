package net.electroland.utils.process;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimerTask;

public class RestartTimerTask extends TimerTask {

    public final static int HOURLY = 1;
    public final static int DAILY  = 2;
    public final static int WEEKLY = 3;

    private RestartDaemon daemon;
    private int type;
    private String dateParam;

    public RestartTimerTask(int type, String dateParam, RestartDaemon daemon) {
        this.type = type;
        this.dateParam = dateParam;
        this.daemon = daemon;
        this.scheduleRestart();
    }

    public static void main(String args[]) throws ParseException{
        System.out.println(new SimpleDateFormat("mm", Locale.ENGLISH).parse("05"));
    }

    public void scheduleRestart() {
        Date nextRestart = null;

        // get the current time

        // add the appropriate delay

        try{
            switch (type){
                case(HOURLY):
                    // TODO: need to get the rest of the date?
                    nextRestart = new SimpleDateFormat("mm", Locale.ENGLISH).parse(dateParam);
                    break;
                case(DAILY):
                    nextRestart = new SimpleDateFormat("h:mm a", Locale.ENGLISH).parse(dateParam);
                    break;
                case(WEEKLY):
                    nextRestart = new SimpleDateFormat("EEE, h:mm a", Locale.ENGLISH).parse(dateParam);
                    break;
            }
        }catch(ParseException e){
            throw new RuntimeException(e);
        }

        if (nextRestart != null){
            synchronized(daemon.getTimer()){
                daemon.getTimer().schedule(this, nextRestart);
            }
        }
    }
    
    @Override
    public void run() {
        daemon.restart();
        this.scheduleRestart();
    }
}