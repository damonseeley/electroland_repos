package net.electroland.utils.process;

import java.util.Calendar;
import java.util.TimerTask;

public class RestartTimerTask extends TimerTask {

    private RestartDaemon daemon;
    private RestartDateTime referenceStartTime;

    public RestartTimerTask(String repeatRate, String repeatDayTime, RestartDaemon daemon) {

        this.daemon = daemon;

        this.referenceStartTime = new RestartDateTime(repeatRate, repeatDayTime);

        this.scheduleRestart();
    }

    public void scheduleRestart() {
        synchronized(daemon.getTimer()){
            daemon.getTimer().schedule(this, getNextStartDateTime().getTime());
        }
    }

    public Calendar getNextStartDateTime(){

        Calendar nextRun = Calendar.getInstance();

        nextRun.set(Calendar.MINUTE, referenceStartTime.getMinutes());

        if (referenceStartTime.isHourly()){
            nextRun.add(Calendar.HOUR, 1);
        } else if (referenceStartTime.isDaily()){
            nextRun.add(Calendar.DATE, 1);
        } else if (referenceStartTime.isWeekly()){
            nextRun.set(Calendar.HOUR, referenceStartTime.getHour());
            while (nextRun.get(Calendar.DAY_OF_WEEK) != referenceStartTime.getDay()
                   && nextRun.before(Calendar.getInstance())){
                nextRun.add(Calendar.DATE, 1);
            }
        }

        return nextRun;
    }

    @Override
    public void run() {
        daemon.restart();
        this.scheduleRestart();
    }
}