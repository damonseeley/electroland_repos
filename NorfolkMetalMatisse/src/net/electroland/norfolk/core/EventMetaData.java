package net.electroland.norfolk.core;

import java.util.Map;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

public class EventMetaData {

    public Queue<NorfolkEvent> history;
    public Map<String, Long>lastCues;
    private long historyMaxLengthMillis;

    public EventMetaData(int historyMaxLengthMillis){
        this.historyMaxLengthMillis = historyMaxLengthMillis;
        history = new LinkedBlockingQueue<NorfolkEvent>();
    }

    public int totalEventsPast(long millis){
        int total = 0;
        for (NorfolkEvent evt : history){
            if (System.currentTimeMillis() - evt.eventTime < millis){
                total++;
            }else{
                break;
            }
        }
        return total;
    }

    public int totalSensorsEvents(long millis){
        int total = 0;
        for (NorfolkEvent evt : history){
            if (System.currentTimeMillis() - evt.eventTime < millis){
                if (evt instanceof SensorEvent){
                    total++;
                }
            }else{
                break;
            }
        }
        return total;
    }

    public int totalAnimationEvents(long millis){
        int total = 0;
        for (NorfolkEvent evt : history){
            if (System.currentTimeMillis() - evt.eventTime < millis){
                if (evt instanceof CueEvent){
                    total++;
                }
            }else{
                break;
            }
        }
        return total;
    }

    public long getTimeOfLastCue(Class<NorfolkEvent> cue){
        Long time = lastCues.get(cue.toString());
        return time == null ? -1 : time;
    }

    public void addEvent(NorfolkEvent evt){

        history.add(evt);

        if (evt instanceof CueEvent){
            lastCues.put(evt.getClass().toString(), evt.eventTime);
        }

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