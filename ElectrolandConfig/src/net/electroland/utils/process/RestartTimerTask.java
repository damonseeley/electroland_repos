package net.electroland.utils.process;

import java.util.Calendar;
import java.util.TimerTask;

public class RestartTimerTask extends TimerTask {

    private RestartDaemon daemon;
    private RestartDateTime referenceStartTime;

    public RestartTimerTask(String repeatRate, String repeatDayTime, RestartDaemon daemon) {

        this.daemon = daemon;
        this.referenceStartTime = new RestartDateTime(repeatRate, repeatDayTime);
        Calendar nextStart = getNextStartDateTime(referenceStartTime);
        daemon.scheduleRestart(this, nextStart.getTime());
    }
    
    public RestartTimerTask(RestartTimerTask original){
        this.daemon             = original.daemon;
        this.referenceStartTime = original.referenceStartTime;
    }

    public static void main(String[] args){
        System.out.println(getNextStartDateTime(new RestartDateTime(RestartDateTime.DAILY, "8:08 PM")).getTime());
        System.out.println(getNextStartDateTime(new RestartDateTime(RestartDateTime.WEEKLY, "Mon, 11:08 PM")).getTime());
        System.out.println(getNextStartDateTime(new RestartDateTime(RestartDateTime.HOURLY, "07")).getTime());
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
            if (hour > 11){
                nextRun.set(Calendar.AM_PM, Calendar.PM);
            }
            if (nextRun.before(now) || nextRun.equals(now)){
                nextRun.add(Calendar.DATE, 1);
            }
            System.out.println(nextRun.getTime());
        } else if (referenceStartTime.isWeekly()){

            int hour = referenceStartTime.getHour();
            nextRun.set(Calendar.HOUR, hour);
            if (hour > 11){
                nextRun.set(Calendar.AM_PM, Calendar.PM);
            }

            while (nextRun.get(Calendar.DAY_OF_WEEK) != referenceStartTime.getDay() ||
                   nextRun.get(Calendar.DATE) == now.get(Calendar.DATE)){

                nextRun.add(Calendar.DATE, 1);
            }
        }

        return nextRun;
    }

    @Override
    public void run() {
        daemon.restart();
        daemon.scheduleRestart(new RestartTimerTask(this), getNextStartDateTime(referenceStartTime).getTime());
    }
}