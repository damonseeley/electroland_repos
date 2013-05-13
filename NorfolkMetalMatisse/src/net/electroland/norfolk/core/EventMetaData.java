package net.electroland.norfolk.core;

import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

public class EventMetaData {

    public Queue<NorfolkEvent> history;
    public Map<String, Long>lastCues =  new HashMap<String, Long>();
    private long historyMaxLengthMillis;

    public EventMetaData(int historyMaxLengthMillis){
        this.historyMaxLengthMillis = historyMaxLengthMillis;
        history = new LinkedBlockingQueue<NorfolkEvent>();
    }

    public static void main(String args[]) throws InterruptedException{
        EventMetaData meta = new EventMetaData(60000);
        meta.addEvent(new SensorEvent());
        Thread.sleep(1000);
        meta.addEvent(new SensorEvent());
        Thread.sleep(1000);
        meta.addEvent(new SensorEvent());
    }

    public int totalEventsPast(long millis){
        int total = 0;
        long current = System.currentTimeMillis();
        for (NorfolkEvent evt : history){
            if (current - evt.eventTime < millis){
                total++;
            }
        }
        return total;
    }

    public int totalSensorsEvents(long millis){
        int total = 0;
        long current = System.currentTimeMillis();
        for (NorfolkEvent evt : history){
            if (current - evt.eventTime < millis &&
                evt instanceof SensorEvent){
                    total++;
            }
        }
        return total;
    }

    public int totalCueEvents(long millis){
        int total = 0;
        long current = System.currentTimeMillis();
        for (NorfolkEvent evt : history){
            if (current - evt.eventTime < millis &&
                evt instanceof CueEvent){
                    total++;
            }
        }

        return total;
    }

    public long getTimeOfLastCue(Cue cue){
        Long time = lastCues.get(cue.getClass().getName());
        return time == null ? -1 : time;
    }

    public long getTimeOfLastNonScreenSaverCue(){
        long overallLast = 0;
        for (String cueName : lastCues.keySet()){
            if (!cueName.equals(ScreenSaverCue.class.getName())){
                long cueLast = lastCues.get(cueName);
                if (cueLast > overallLast){
                    overallLast = cueLast;
                }
            }
        }
        return overallLast;
    }

    public void addCue(Cue cue){
        lastCues.put(cue.getClass().getName(), System.currentTimeMillis());
    }

    public void addEvent(NorfolkEvent evt){

        history.add(evt);

        if (headIsTooldestEventIsTooOld(history, historyMaxLengthMillis)){
            history.remove();
        }
    }

    public static boolean headIsTooldestEventIsTooOld(Queue<NorfolkEvent> history, long maxAgeMillis){
        if (history.size() == 0){
            return false;
        }else{
            return System.currentTimeMillis() - history.peek().eventTime > maxAgeMillis;
        }
    }
}