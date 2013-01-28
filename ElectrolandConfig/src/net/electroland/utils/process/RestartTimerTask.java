package net.electroland.utils.process;

import java.util.Calendar;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

import org.apache.log4j.Logger;

public class RestartTimerTask extends TimerTask {

    static Logger logger = Logger.getLogger(RestartTimerTask.class);

    private MonitoredProcess process;
    private RestartDateTime referenceStartTime;
    private Timer timer;

    public RestartTimerTask(String repeatRate, String repeatDayTime, MonitoredProcess process, Timer timer) {

        this.process = process;
        this.timer = timer;
        this.referenceStartTime = new RestartDateTime(repeatRate, repeatDayTime);

        this.scheduleRestart();
    }
    
    public RestartTimerTask(RestartTimerTask original){
        this.process            = original.process;
        this.referenceStartTime = original.referenceStartTime;
        this.timer              = original.timer;
    }

    public void scheduleRestart(){
        Date nextRestart = getNextStartDateTime(referenceStartTime).getTime();
        logger.info("scheduling restart of " + process.getName() + " at " + nextRestart);
        timer.schedule(this, nextRestart);
    }

    public static Calendar getNextStartDateTime(RestartDateTime referenceStartTime){

        Calendar now     = Calendar.getInstance();
        Calendar nextRun = Calendar.getInstance();

        // make sure comparisons during the same minute don't botch up.
        now.set(Calendar.SECOND, 0);
        now.set(Calendar.MILLISECOND, 0);
        nextRun.set(Calendar.SECOND, 0);
        nextRun.set(Calendar.MILLISECOND, 0);

        // all restarts happen on a specified minute.
        nextRun.set(Calendar.MINUTE, referenceStartTime.getMinutes());

        if (referenceStartTime.isHourly()){

            if (nextRun.before(now) || nextRun.equals(now)){
                nextRun.add(Calendar.HOUR, 1);
            }

        } else if (referenceStartTime.isDaily()){

            int hour = referenceStartTime.getHour();
            nextRun.set(Calendar.HOUR, hour);
            nextRun.set(Calendar.AM_PM, referenceStartTime.getAmPm());

            if (nextRun.before(now) || nextRun.equals(now)){
                nextRun.add(Calendar.DATE, 1);
            }

        } else if (referenceStartTime.isWeekly()){

            int hour = referenceStartTime.getHour();
            nextRun.set(Calendar.HOUR, hour);
            nextRun.set(Calendar.AM_PM, referenceStartTime.getAmPm());

            while (nextRun.get(Calendar.DAY_OF_WEEK) != referenceStartTime.getDay() ||
                   nextRun.get(Calendar.DATE) == now.get(Calendar.DATE)){

                nextRun.add(Calendar.DATE, 1);
            }

        }

        return nextRun;
    }

    public RestartDateTime getReferenceStartDateTime(){
        return referenceStartTime;
    }

    @Override
    public void run() {
        process.kill(true);
        RestartTimerTask nextTask = new RestartTimerTask(this);
        nextTask.scheduleRestart();
    }
}